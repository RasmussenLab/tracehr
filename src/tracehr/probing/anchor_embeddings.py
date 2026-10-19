# src/tracehr/probing/anchor_embeddings.py
"""Pipeline 05's embedding extraction: one row per (patient, anchor) holding the frozen
transformer's embedding of the history before the anchor, the targets, and the
bag-of-words and demographic columns the comparison arms read.

The cohort is read in ONE ordered pass (anchor_windows.iter_patient_windows): DuckDB
encodes tokens/types and applies the anchor cutoff, windows are filled into preallocated
buffers, and a producer thread keeps the model fed while peak memory stays bounded by
chunk_windows rather than by the cohort. With a prebuilt window store
(outcome.mode: save_intermediate) the windows are read memory-mapped instead.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..dataset.cohort import ANCHOR_SKIPPED, anchors_by_patient, report_missing_anchors
from ..dataset.meds_reader import ABS_COL, AGE_COL
from ..dataset.vocabulary import load_vocab, type_id_map, vocab_fingerprint
from ..dataset.windows import preamble_spec
from .anchor_windows import (
    alloc_window_buffers,
    anchor_window_store_mismatch,
    fill_patient_windows,
    iter_patient_windows,
    open_anchor_window_store,
    patient_record,
)
from .embeddings import (
    combine_window_embeddings,
    extract_sequence_embedding,
    validate_extraction_settings,
)


# ============================================================================ #
# Embedding + sequence extraction at a per-patient time anchor
# ============================================================================ #
def _embed_buffer(model, buf: dict, n: int, batch_size: int, pooling: str) -> np.ndarray:
    """Forward-pass the first n rows of a window buffer; return [n, D] embeddings.

    Slices the preallocated buffers straight into batches - no per-window tuple rebuild.
    """
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    out = []
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        out.append(extract_sequence_embedding(
            model,
            input_ids=torch.from_numpy(buf["input_ids"][lo:hi]).to(torch.int64),
            attention_mask=torch.from_numpy(buf["attention_mask"][lo:hi]).to(torch.int64),
            time_age=(torch.from_numpy(buf["time_age"][lo:hi])
                      if model.needs_time_age else None),
            time_abs=(torch.from_numpy(buf["time_abs"][lo:hi])
                      if buf["time_abs"] is not None else None),
            tok_type=(torch.from_numpy(buf["type"][lo:hi]).to(torch.int64)
                      if buf["type"] is not None else None),
            pooling=pooling).cpu().numpy())
    return np.concatenate(out, axis=0)


def _patient_rows(records: list[dict], embs: np.ndarray, *, combine_windows: str,
                  itos, id_col, save_sequences: bool) -> list[dict]:
    """Reduce each patient's windows to one row. `embs` is this flush's windows in order.

    Window order is chronological (oldest -> newest) by construction, which
    combine_window_embeddings' recency weighting depends on: it weights index n-1 highest.
    """
    rows, off = [], 0
    for rec in records:
        n_win = rec["n_windows"]
        w = embs[off:off + n_win]
        if len(w) != n_win:
            raise RuntimeError(
                f"patient {rec['patient_id']!r}: expected {n_win} window embeddings, got "
                f"{len(w)}. A patient's windows must never be split across a flush.")
        off += n_win
        # "none" (most-recent window only) is one of the combiner's methods, so there is no
        # special case here to drift from it.
        combined = combine_window_embeddings(w, combine_windows)

        row = {
            id_col: rec["patient_id"],
            "anchor_age": rec["anchor_age"],
            "target": rec["target"],
            # Survival only; absent for every other task, so the frame has a duration
            # column only when one was requested.
            **({"duration": rec["duration"]} if "duration" in rec else {}),
            "split": rec["split"],
            "Sex": rec["Sex"],
            "last_token_age": rec["last_token_age"],
            "Embedding": combined.astype(np.float32),
            "bow_tokens": rec["bow_tokens"],
            "bow_token_counts": rec["bow_token_counts"],
            "n_windows": n_win,
        }
        if save_sequences:
            # The full truncated pre-anchor history, sex token removed (Sex has its own
            # column) - the same span bow_tokens counts, in chronological order.
            row["Sequence"] = [itos[int(t)] for t in rec["_hist_ids"]]
            row["Age"] = [float(a) for a in rec["_hist_ages"]]
        rows.append(row)
    if off != len(embs):
        raise RuntimeError(f"{len(embs) - off} window embeddings were not consumed.")
    return rows


@torch.no_grad()
def extract_anchor_embeddings(
    model,
    parquet_path: str,
    vocab_path: str,
    anchors: pd.DataFrame,
    patient_ids: set,
    split_name: str,
    *,
    pooling: str = "mean",
    overlap_percent: float = 0.0,
    start_from_end: bool = True,
    combine_windows: str = "recency_weighted",
    max_seq_len: int | None = None,
    time_before_event: float = 0.0,
    remove_cutoff_token: bool = True,
    use_time_abs: bool = True,
    use_type: bool = True,
    batch_size: int = 64,
    min_seq_len: int = 1,
    add_head_window: bool = False,
    id_col: str = "subject_id",
    tok_col: str = "code",
    time_age_col: str = AGE_COL,
    time_abs_col: str = ABS_COL,
    type_col: str = "type",
    outcome_col: str = "target",
    outcome_duration_col: str | None = None,
    outcome_patient_col: str | None = None,
    outcome_age_col: str | None = None,
    show_progress: bool = True,
    split_by_pid: dict | None = None,
    chunk_windows: int = 2048,
    windows_dir: str | None = None,
    save_sequences: bool = False,
    event_times=None,
) -> pd.DataFrame:
    """One row per (patient, anchor): the transformer embedding of the pre-anchor history.

    `anchors` is the outcomes table as load_anchors returns it; a patient with several
    anchors gets one row per anchor, all cut from one read of the patient's history.
    Truncation keeps events with Age < (anchor - time_before_event), both in days since the
    patient's birth (load_event_times, load_anchors), and is applied inside DuckDB so
    truncated rows never reach Python. Patients absent from the outcomes table, or with
    too little pre-anchor history to form a window, are dropped.
    outcome_patient_col/outcome_age_col name the outcomes-table columns; None falls back
    to the sequence names (id_col/time_age_col). outcome_duration_col is survival-only:
    naming it adds a "duration" column (time from the anchor to the event or to
    censoring) alongside "target", which then carries the 0/1 event indicator. Left None,
    the frame has no duration column.

    Windowing: each patient's truncated history is split into windows of width
    max_seq_len (overlap_percent, start_from_end) and reduced to one embedding by
    combine_windows (mean|max|recency_weighted); combine_windows="none" keeps only the
    most-recent window. `pooling` is the within-window token pooling. Output stays one row
    per (patient, anchor); n_windows records how many windows fed it.

    min_seq_len is the window floor and should be the value the model was trained with (the
    pipeline reads it off the checkpoint). It gates BOTH whole patients - too little
    pre-anchor history to fill one trainable window - and individual spans, so the
    right-aligned plan cannot emit a narrow head window that the model never saw in training
    and that combine_windows would then weight like any other. Left at 1 nothing is gated.

    The cohort is read in ONE ordered pass: windows are built into preallocated buffers
    and embedded in flushes of ~chunk_windows, so peak memory is bounded by the chunk
    rather than the cohort.

    split_by_pid ({pid: "train"|"test"}) covers both splits in that single pass; without
    it every patient in patient_ids is labelled split_name.
    windows_dir reads windows from a prebuilt store (build_anchor_window_store) instead of
    streaming the parquet. event_times (load_event_times) is passed to the reader.
    save_sequences adds the Sequence/Age inspection columns; they cost one Python object
    per token per patient, so they are off by default.
    """
    validate_extraction_settings(combine_windows, pooling, overlap_percent)

    if max_seq_len is None:
        max_seq_len = model.max_seq_len
    model.eval()

    stoi, itos, stats = load_vocab(vocab_path)
    sos_id, con_id, pad_id = int(stoi["<SOS>"]), int(stoi["<CON>"]), int(stoi["<PAD>"])
    preamble = preamble_spec(stats)
    type_to_id = type_id_map(stats) if use_type else None

    outcome_patient_col = outcome_patient_col or id_col
    outcome_age_col = outcome_age_col or time_age_col

    # Targets (and the survival duration) are looked up by (patient id, anchor age), which
    # load_anchors makes unique. The split label comes from split_by_pid, by patient.
    oc = anchors
    keys = list(zip(oc[outcome_patient_col], oc[outcome_age_col].astype(float)))
    target_by_key = dict(zip(keys, oc[outcome_col])) if outcome_col in oc.columns else {}
    # Survival only. Missing here is a config error, not an absent optional column: the
    # caller asked for a duration by naming it, so fail before the expensive extraction.
    if outcome_duration_col is not None and outcome_duration_col not in oc.columns:
        raise KeyError(
            f"Anchors table missing the survival duration column "
            f"{outcome_duration_col!r}. Available: {list(oc.columns)}")
    duration_by_key = (dict(zip(keys, oc[outcome_duration_col]))
                       if outcome_duration_col is not None else None)

    if split_by_pid is None:
        split_by_pid = {pid: split_name for pid in patient_ids}
        label = split_name
    else:
        label = "+".join(sorted(set(split_by_pid.values())))
    cohort = oc[oc[outcome_patient_col].isin(set(split_by_pid))]
    report_missing_anchors(cohort[outcome_age_col], ANCHOR_SKIPPED, label=label)
    cohort_anchors = anchors_by_patient(cohort, outcome_patient_col, outcome_age_col)

    def finish(rec, work):
        """Complete a patient_record row in place and return it: the target, the survival
        duration, the split and, with save_sequences, the truncated history. The target is
        a float, not an int, so continuous regression targets stay intact; binary 0/1
        becomes 0.0/1.0, which every metric, curve and StratifiedKFold treats identically."""
        key = (rec["patient_id"], rec["anchor_age"])
        rec["target"] = float(target_by_key.get(key, np.nan))
        if duration_by_key is not None:
            rec["duration"] = float(duration_by_key.get(key, np.nan))
        rec["split"] = split_by_pid.get(rec["patient_id"])
        if save_sequences and work is not None:
            # The FULL truncated history, the span the embedding covers under every
            # combine_windows except "none" - not one window, and not the window as the
            # model saw it: no SOS/CON, and no front block (Sex has its own column).
            rec["_hist_ids"] = work.toks.tolist()
            rec["_hist_ages"] = work.ages.tolist()
        return rec

    row_kwargs = dict(combine_windows=combine_windows, itos=itos, id_col=id_col,
                      save_sequences=save_sequences)

    if windows_dir is not None:
        # The store's geometry decides the windows, so a caller asking for different
        # geometry would silently get the store's. Fail instead. (run_downstream_stage
        # already validates before building, but this function is public.)
        mismatch = anchor_window_store_mismatch(
            windows_dir,
            max_seq_len=int(max_seq_len), overlap_percent=float(overlap_percent),
            start_from_end=bool(start_from_end),
            # The floor decides both which patients survive and how wide the head window
            # is, so a store built under a different one holds different windows.
            min_seq_len=int(min_seq_len),
            # Same reason: it decides whether the sub-floor head gets a window at all.
            add_head_window=bool(add_head_window),
            time_before_event=float(time_before_event),
            remove_cutoff_token=bool(remove_cutoff_token),
            use_time_abs=bool(use_time_abs), use_type=bool(use_type),
            # The vocabulary records the front block and the time origin the windows used.
            vocab_hash=vocab_fingerprint(vocab_path))
        if mismatch is not None:
            raise ValueError(
                f"the window store at {windows_dir} does not match the requested "
                f"extraction: {mismatch}. Rebuild it with build_anchor_window_store, or "
                f"drop windows_dir to stream.")
        rows = _rows_from_store(
            model, windows_dir, finish=finish, batch_size=batch_size,
            chunk_windows=chunk_windows, pooling=pooling, combine_windows=combine_windows,
            save_sequences=save_sequences, row_kwargs=row_kwargs,
            show_progress=show_progress, label=label)
    else:
        rows = _rows_from_stream(
            model, parquet_path, finish=finish, stoi=stoi, itos=itos,
            preamble=preamble, type_to_id=type_to_id, anchors=cohort_anchors,
            max_seq_len=max_seq_len, overlap_percent=overlap_percent,
            start_from_end=start_from_end, time_before_event=time_before_event,
            remove_cutoff_token=remove_cutoff_token, min_seq_len=min_seq_len,
            add_head_window=add_head_window,
            id_col=id_col, tok_col=tok_col, time_age_col=time_age_col,
            time_abs_col=time_abs_col if use_time_abs else None,
            type_col=type_col if use_type else None,
            use_time_abs=use_time_abs, use_type=use_type, pad_id=pad_id,
            sos_id=sos_id, con_id=con_id, batch_size=batch_size,
            chunk_windows=chunk_windows, pooling=pooling, row_kwargs=row_kwargs,
            show_progress=show_progress, label=label,
            # "none" needs only the most-recent window, so don't build or embed the rest.
            single_final=combine_windows == "none", event_times=event_times)

    if not rows:
        raise ValueError(
            f"[downstream] No anchor of split '{label}' kept enough history to embed. Check "
            f"that the outcomes file's patient ids occur in the input and that the anchors "
            f"(minus time_before_event) come after at least min_seq_len={min_seq_len} of the "
            f"patient's events.")
    # min_seq_len gates whole patients as well as spans, so raising it shrinks the scored
    # cohort. That is a change to WHO is measured, not just how, and it would otherwise be
    # invisible in the metrics - report it rather than letting the row count drift quietly.
    if show_progress and len(rows) < len(cohort):
        print(f"[downstream] embedded {len(rows)}/{len(cohort)} cohort anchors "
              f"(min_seq_len={min_seq_len}); {len(cohort) - len(rows)} dropped for "
              f"a missing/NaN anchor, no events in the input, or fewer than "
              f"{min_seq_len} pre-anchor clinical event(s).")
    return pd.DataFrame(rows)


def _rows_from_stream(model, parquet_path, *, finish, stoi, itos, preamble, type_to_id,
                      anchors, max_seq_len, overlap_percent, start_from_end,
                      time_before_event, remove_cutoff_token, min_seq_len, add_head_window,
                      id_col, tok_col,
                      time_age_col, time_abs_col, type_col, use_time_abs,
                      use_type, pad_id, sos_id, con_id, batch_size, chunk_windows, pooling,
                      row_kwargs, show_progress, label, single_final,
                      event_times=None) -> list[dict]:
    """One ordered pass: a producer thread builds windows while the model embeds them.

    The producer only ever emits whole patients, so every patient is embedded and combined
    inside a single flush - the invariant _patient_rows checks, and the reason multi-window
    recombination stays correct no matter where the chunk boundaries land.
    """
    q: queue.Queue = queue.Queue(maxsize=2)
    SENTINEL = object()
    stop = threading.Event()

    def put(item) -> bool:
        """Block for a free slot like q.put, but wake if the consumer has given up.

        The consumer is the producer's only source of free slots, so a plain put() on the
        bounded queue parks forever once the consumer leaves - and the join() below would
        then wait on it forever. False means the pass is over: abandon it.
        """
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        """The producer thread: fill whole patients' windows into buffers and queue them as
        (buffer, n_windows, records) chunks; a failure is queued as the exception, and the
        sentinel ends the queue."""
        try:
            buf = alloc_window_buffers(chunk_windows, max_seq_len, pad_id,
                                       use_time_abs, use_type)
            recs, k = [], 0
            for work in iter_patient_windows(
                    parquet_path, stoi=stoi, preamble=preamble,
                    type_to_id=type_to_id, anchors=anchors, max_seq_len=max_seq_len,
                    overlap_percent=overlap_percent, start_from_end=start_from_end,
                    time_before_event=time_before_event,
                    remove_cutoff_token=remove_cutoff_token, min_seq_len=min_seq_len,
                    add_head_window=add_head_window,
                    id_col=id_col, tok_col=tok_col, time_age_col=time_age_col,
                    time_abs_col=time_abs_col, type_col=type_col,
                    single_final=single_final, event_times=event_times):
                n_win = len(work.spans)
                if k and k + n_win > chunk_windows:
                    if not put((buf, k, recs)):
                        return                  # consumer gone: stop building windows
                    # A fresh buffer per flush: the consumer still owns the previous one.
                    buf = alloc_window_buffers(chunk_windows, max_seq_len, pad_id,
                                               use_time_abs, use_type)
                    recs, k = [], 0
                if n_win > chunk_windows:
                    # One patient wider than a chunk: grow just for this patient rather
                    # than splitting it, so the whole-patient invariant always holds.
                    buf = alloc_window_buffers(n_win, max_seq_len, pad_id,
                                               use_time_abs, use_type)
                k = fill_patient_windows(buf, k, work, sos_id, con_id)
                recs.append(finish(patient_record(work, itos), work))
            if k:
                put((buf, k, recs))
        except BaseException as exc:            # surface producer failures to the consumer
            put(exc)                            # via put: a full queue must not wedge this
        finally:
            put(SENTINEL)

    thread = threading.Thread(target=produce, name="anchor-window-producer", daemon=True)
    thread.start()

    rows, bar = [], None
    if show_progress:
        bar = tqdm(desc=f"[{label}] embedding windows", unit="win", mininterval=2.0,
                   leave=False)
    try:
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            buf, k, recs = item
            embs = _embed_buffer(model, buf, k, batch_size, pooling)
            rows.extend(_patient_rows(recs, embs, **row_kwargs))
            if bar is not None:
                bar.update(k)
    finally:
        if bar is not None:
            bar.close()
        # Release a producer parked on a full queue before waiting on it. Every early exit
        # here - an OOM in _embed_buffer, _patient_rows' whole-patient check, Ctrl-C - ends
        # the draining the producer is blocked on, so without this the join deadlocks and
        # an hours-long stage stalls silently instead of reporting the error. The timeout
        # covers the producer being mid-read rather than at a put(); it honours the stop at
        # its next one, and the thread is a daemon.
        stop.set()
        thread.join(timeout=30.0)
    return rows


def _rows_from_store(model, windows_dir, *, finish, batch_size, chunk_windows, pooling,
                     combine_windows, save_sequences, row_kwargs, show_progress,
                     label) -> list[dict]:
    """Embed from a prebuilt window store, memory-mapped so RAM stays flat.

    open_anchor_window_store validates the per-patient layout, so windows arrive grouped
    by patient and ordered oldest -> newest exactly as the combiner requires.
    """
    if save_sequences:
        raise ValueError(
            "save_sequences is not available when reading a prebuilt window store; the "
            "store keeps windows, not the token strings. Re-run in streaming mode.")
    mm, patients = open_anchor_window_store(windows_dir)
    n_win_per = patients["n_windows"].to_numpy()
    # Chunk on patient boundaries: offsets[i] is where patient i's windows start.
    offsets = np.concatenate(([0], np.cumsum(n_win_per)))
    keep_final_only = combine_windows == "none"
    names = mm.dtype.names or ()

    rows, i, n_pat = [], 0, len(patients)
    bar = (tqdm(total=n_pat, desc=f"[{label}] embedding from store", unit="pat",
                mininterval=2.0, leave=False) if show_progress else None)
    try:
        while i < n_pat:
            j, taken = i, 0
            while j < n_pat and (taken == 0 or taken + n_win_per[j] <= chunk_windows):
                taken += n_win_per[j]
                j += 1
            block = mm[offsets[i]:offsets[j]]
            if keep_final_only:
                block = block[block["is_final_window"] == 1]
            # np.array (not ascontiguousarray): a structured-array field view keeps the
            # record stride, which torch.from_numpy rejects. This forces a real copy.
            buf = {
                "input_ids": np.array(block["input_ids"]),
                "attention_mask": np.array(block["attention_mask"]),
                "time_age": np.array(block["time_age"]),
                "time_abs": np.array(block["time_abs"]) if "time_abs" in names else None,
                "type": np.array(block["type"]) if "type" in names else None,
            }
            embs = _embed_buffer(model, buf, len(block), batch_size, pooling)
            recs = []
            for r in patients.iloc[i:j].to_dict("records"):
                if keep_final_only:
                    r = {**r, "n_windows": 1}
                recs.append(finish(r, None))
            rows.extend(_patient_rows(recs, embs, **row_kwargs))
            if bar is not None:
                bar.update(j - i)
            i = j
    finally:
        if bar is not None:
            bar.close()
    return rows
