# src/tracehr/probing/anchor_windows.py
"""Anchor-truncated window production for downstream outcome prediction, sized for
millions of patients.

One shared producer (iter_patient_windows) feeds BOTH modes, so streaming and
save_intermediate can never drift apart:

  streaming          producer -> chunk buffers -> forward passes, nothing persisted
  save_intermediate  producer -> anchor_windows.npy (+ sidecars) -> forward passes

The heavy lifting is pushed into DuckDB: token/type encoding is a hash join and the
anchor cutoff is a join predicate, so per-event Python work is zero and truncated rows
never leave the database. Windows are filled into preallocated buffers rather than
allocating five arrays each.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from ..dataset.cohort import cutoff_index
from ..dataset.meds_reader import (
    ABS_COL,
    AGE_COL,
    canonical_sort,
    event_source,
    normalized_sql,
    sql_in,
    stream_encoded_patient_sequences,
)
from ..dataset.vocabulary import json_digest, load_vocab, type_id_map, vocab_fingerprint
from ..dataset.windows import (
    PreambleSpec,
    extract_preamble_tokens,
    history_window_spans,
    preamble_spec,
    validate_window_floor,
    window_geometry,
    write_window,
)

WINDOWS_FILE = "anchor_windows.npy"
PATIENTS_FILE = "anchor_patients.parquet"
META_FILE = "anchor_store_meta.json"


# ============================================================================ #
# The unit of work
# ============================================================================ #
@dataclass
class PatientWork:
    """One (patient, anchor): the history truncated at that anchor plus its chronological
    window plan. A patient with several anchors yields one PatientWork per anchor."""
    pid: object
    spans: list[dict]      # oldest -> newest; spans[-1] is the prediction point
    toks: np.ndarray
    ages: np.ndarray
    abss: np.ndarray | None
    typs: np.ndarray | None
    preamble: tuple        # the front block, as extract_preamble_tokens returns it
    sex_id: int | None
    anchor_age: float


# ============================================================================ #
# The shared producer
# ============================================================================ #
def iter_anchor_histories(
    parquet_path: str,
    *,
    stoi: dict[str, int],
    preamble: PreambleSpec,
    type_to_id: dict[str, int] | None,
    anchors: dict,
    time_before_event: float = 0.0,
    remove_cutoff_token: bool = True,
    id_col: str = "subject_id",
    tok_col: str = "code",
    time_age_col: str = AGE_COL,
    time_abs_col: str | None = ABS_COL,
    type_col: str | None = "type",
    event_times=None,
):
    """Yield (pid, anchor_age, toks, ages, abss, typs, preamble, sex_id) once per anchor,
    in patient-id order and oldest anchor first: the history before that anchor, in
    canonical order, with the sex and static tokens taken out into the front block
    (preamble and sex_id, as extract_preamble_tokens returns them).

    anchors maps each patient id to a list of anchor ages; missing ones are skipped. Each
    patient is read once: SQL truncates at the patient's LATEST cutoff (anchor_age -
    time_before_event), and an earlier anchor's history is a prefix of that read - events
    sort by age first, so the ones with Age < cutoff (<= when remove_cutoff_token is False)
    are exactly the first searchsorted(cutoff) of them. A one-anchor patient therefore gets
    the full SQL result.
    event_times (load_event_times) is passed to the reader.
    """
    by_pid = {pid: sorted(float(a) for a in ages if not pd.isna(a))
              for pid, ages in anchors.items()}
    by_pid = {pid: ages for pid, ages in by_pid.items() if ages}
    latest = {pid: ages[-1] - float(time_before_event) for pid, ages in by_pid.items()}

    for pid, toks, ages, abss, typs in stream_encoded_patient_sequences(
            parquet_path, id_col, tok_col, time_age_col, time_abs_col, type_col,
            stoi=stoi, unk_id=int(stoi["<UNK>"]), type_to_id=type_to_id,
            anchors=latest, remove_cutoff_token=remove_cutoff_token, event_times=event_times):
        toks, ages, abss, typs = canonical_sort(toks, ages, abss, typs)
        for anchor_age in by_pid[pid]:
            n = cutoff_index(ages, anchor_age - float(time_before_event), remove_cutoff_token)
            yield (pid, anchor_age, *extract_preamble_tokens(
                toks[:n], ages[:n],
                abss[:n] if abss is not None else None,
                typs[:n] if typs is not None else None, preamble))


def iter_patient_windows(
    parquet_path: str,
    *,
    stoi: dict[str, int],
    preamble: PreambleSpec,
    type_to_id: dict[str, int] | None,
    anchors: dict,
    max_seq_len: int,
    overlap_percent: float = 0.0,
    start_from_end: bool = True,
    time_before_event: float = 0.0,
    remove_cutoff_token: bool = True,
    min_seq_len: int = 1,
    add_head_window: bool = False,
    id_col: str = "subject_id",
    tok_col: str = "code",
    time_age_col: str = AGE_COL,
    time_abs_col: str | None = ABS_COL,
    type_col: str | None = "type",
    single_final: bool = False,
    event_times=None,
):
    """Yield one PatientWork per surviving (patient, anchor), in patient-id order and
    oldest anchor first. anchors maps each patient id to a list of anchor ages.
    event_times (load_event_times) is passed to the reader.

    Truncation happens in SQL and per anchor (see iter_anchor_histories), so the arrays
    that arrive here are already cut at anchor_age - time_before_event.

    single_final=True plans only the most-recent window (what combine_windows="none"
    needs, and all it needs). The saved store always uses False so that one store serves
    every combine_windows, pooling, batch_size and checkpoint; "none" then becomes a
    read-time filter on is_final_window, which selects the identical window.
    """
    K, stride = window_geometry(max_seq_len, overlap_percent, preamble.slots)
    min_seq_len = validate_window_floor(min_seq_len, K)

    for pid, anchor_age, toks, ages, abss, typs, front, sex_id in iter_anchor_histories(
            parquet_path, stoi=stoi, preamble=preamble, type_to_id=type_to_id,
            anchors=anchors, time_before_event=time_before_event,
            remove_cutoff_token=remove_cutoff_token, id_col=id_col, tok_col=tok_col,
            time_age_col=time_age_col, time_abs_col=time_abs_col, type_col=type_col,
            event_times=event_times):
        if len(toks) < min_seq_len:
            continue  # too little pre-anchor history to form a window
        spans = history_window_spans(len(toks), K, stride, start_from_end,
                                     single_final=single_final, min_seq_len=min_seq_len,
                                     add_head_window=add_head_window)
        yield PatientWork(pid, spans, toks, ages, abss, typs, front, sex_id, anchor_age)


# ============================================================================ #
# Allocation-free window building
# ============================================================================ #
def alloc_window_buffers(chunk: int, max_seq_len: int, pad_id: int,
                         use_abs: bool, use_type: bool) -> dict:
    """Preallocate the (chunk, max_seq_len) buffers a flush writes into.

    build_window allocates five arrays per window; at millions of windows that allocator
    traffic is pure overhead. Optional feature buffers are skipped entirely when the model
    does not use them.
    """
    buf = {
        "input_ids": np.full((chunk, max_seq_len), pad_id, dtype=np.int32),
        "time_age": np.zeros((chunk, max_seq_len), dtype=np.float64),   # times are float64
        "attention_mask": np.zeros((chunk, max_seq_len), dtype=np.int8),
        "pad_id": pad_id,
    }
    buf["time_abs"] = np.zeros((chunk, max_seq_len), dtype=np.float64) if use_abs else None
    buf["type"] = np.zeros((chunk, max_seq_len), dtype=np.int32) if use_type else None
    return buf


def fill_window(buf: dict, k: int, chunk_toks, chunk_ages, chunk_abss, chunk_typs,
                start_tok: int, preamble: tuple) -> None:
    """Write one window into row k of the buffers: the window build_window returns."""
    ids, age, mask = buf["input_ids"], buf["time_age"], buf["attention_mask"]
    absb, typb = buf["time_abs"], buf["type"]

    # Row k may be reused across flushes, so reset it first.
    ids[k] = buf["pad_id"]
    age[k] = 0.0
    mask[k] = 0
    if absb is not None:
        absb[k] = 0.0
    if typb is not None:
        typb[k] = 0
    write_window(ids[k], age[k], absb[k] if absb is not None else None,
                 typb[k] if typb is not None else None, mask[k], chunk_toks, chunk_ages,
                 chunk_abss, chunk_typs, start_tok, preamble)


def fill_patient_windows(buf: dict, k: int, work: PatientWork,
                         sos_id: int, con_id: int) -> int:
    """Write all of one patient's windows starting at row k; return the next free row.

    Spans are written oldest -> newest, the order combine_window_embeddings' recency
    weighting depends on.
    """
    for w in work.spans:
        ws, we = w["start"], w["end"]
        fill_window(
            buf, k, work.toks[ws:we], work.ages[ws:we],
            work.abss[ws:we] if work.abss is not None else None,
            work.typs[ws:we] if work.typs is not None else None,
            sos_id if w["first"] else con_id, work.preamble)
        k += 1
    return k


# ============================================================================ #
# Per-patient sidecar row (everything the windows cannot express)
# ============================================================================ #
def patient_record(work: PatientWork, itos: list[str]) -> dict:
    """Sidecar fields for one (patient, anchor): target-side scalars plus full-history
    bag-of-words.

    The BoW counts span the FULL truncated history, not just the most-recent window - the
    same span the embedding covers under mean/max/recency_weighted. (One documented gap: a
    right-aligned plan drops a head span narrower than min_seq_len, and with
    add_head_window=False - the default - nothing replaces it. Those oldest
    (len(toks) mod stride) tokens therefore reach BoW but no window, and so no embedding.
    That asymmetry is the deliberate trade: covering them costs a near-duplicate window
    that combine_windows would weight like a real one. add_head_window=True closes the gap
    and pays that cost.) The front block's sex and static tokens are
    re-added with count 1 because extract_preamble_tokens pulled them out of the history but
    they are genuine patient features. SOS/CON are deliberately absent: they belong to the
    window, not the history.

    bow_tokens holds token NAMES, not ids, so the column is readable without a vocabulary
    and is directly comparable with pipeline 04's column of the same name.
    """
    sex_id = work.sex_id if work.sex_id is not None else -1
    pre_ids, pre_ages = work.preamble[0], work.preamble[1]
    hist_ids, hist_counts = np.unique(work.toks, return_counts=True)
    return {
        "patient_id": work.pid,
        "anchor_age": float(work.anchor_age),
        "Sex": itos[sex_id] if sex_id >= 0 else None,
        "sex_id": sex_id,
        "sex_age": float(pre_ages[pre_ids == sex_id][0]) if sex_id >= 0 else 0.0,
        "last_token_age": float(work.ages[-1]) if len(work.ages) else np.nan,
        "bow_tokens": [itos[int(t)] for t in hist_ids] + [itos[int(t)] for t in pre_ids],
        "bow_token_counts": [int(c) for c in hist_counts] + [1] * len(pre_ids),
        "n_windows": len(work.spans),
    }


# ============================================================================ #
# Fingerprints + staleness
# ============================================================================ #
def anchor_fingerprint(anchors: dict, split_by_pid: dict | None = None) -> str:
    """Stable 16-hex digest of the derived cohort: sorted (pid, anchor_age, split), one
    triple per anchor. anchors maps each patient id to a list of anchor ages.

    Fingerprinting what was derived rather than the files themselves catches a changed
    anchor age, a changed target cohort or a re-split, and stays cheap because these
    tables hold one row per anchor. A one-anchor-per-patient cohort hashes exactly as it
    did when anchors were one per patient. The digest is json_digest's.
    """
    split_by_pid = split_by_pid or {}
    triples = sorted(
        (str(pid), None if age is None or pd.isna(age) else float(age),
         split_by_pid.get(pid))
        for pid, ages in anchors.items() for age in ages)
    return json_digest(triples)


def store_meta(windows_dir) -> dict | None:
    """The store's meta json (META_FILE) as a dict, or None when the store has none."""
    path = Path(windows_dir) / META_FILE
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def anchor_window_store_mismatch(windows_dir, **expected) -> str | None:
    """Compare a saved store against the current config.

    Returns None when it matches, else a human-readable description of the first
    mismatch, so the caller can print the reason and regenerate - the same contract as
    window_npz_geometry_mismatch.
    """
    d = Path(windows_dir)
    for name in (WINDOWS_FILE, PATIENTS_FILE, META_FILE):
        if not (d / name).is_file():
            return f"missing {name}"
    meta = store_meta(d)
    if meta is None:
        return f"missing {META_FILE}"
    for key, value in expected.items():
        if key not in meta:
            return f"{key} is absent from the store (built by an older version)"
        if meta[key] != value:
            return f"{key}={meta[key]!r} but the config expects {value!r}"
    return None


# ============================================================================ #
# The on-disk store
# ============================================================================ #
def _store_dtype(max_seq_len: int, use_abs: bool, use_type: bool) -> np.dtype:
    """The structured dtype of one store row: the window's arrays (time_abs and type only
    when the model reads them), the patient's row in PATIENTS_FILE, the window's
    chronological position and the final-window flag."""
    fields = [
        ("input_ids", np.int32, (max_seq_len,)),
        ("attention_mask", np.int8, (max_seq_len,)),
        ("time_age", np.float64, (max_seq_len,)),
    ]
    if use_abs:
        fields.append(("time_abs", np.float64, (max_seq_len,)))
    if use_type:
        fields.append(("type", np.int32, (max_seq_len,)))
    fields += [
        ("patient_idx", np.int32),      # row index into anchor_patients.parquet
        ("window_pos", np.int32),       # 0-based chronological position within the patient
        ("is_final_window", np.int8),   # window_pos == n_windows - 1
    ]
    return np.dtype(fields)


def _count_windows(parquet_path, cutoffs, preamble_slots, *, id_col, tok_col, time_age_col,
                   max_seq_len, overlap_percent, start_from_end, min_seq_len,
                   remove_cutoff_token, add_head_window=False, event_times=None) -> int:
    """Exact window total from one bounded DuckDB aggregate - no rows materialised.

    Mirrors create_sequence_windows' sizing pass, with the anchor join added so the count
    reflects the truncated history. cutoffs maps each patient id to a list of cutoffs; each
    one is its own row of the join (aid), so every anchor is counted on its own history.
    Sex and static tokens are matched as strings because the raw parquet is unencoded;
    extract_preamble_tokens removes all of them, so the effective length is (events - sex
    and static events). It reads the same event source as the build (event_times), so both
    see the same patients.
    """
    K, stride = window_geometry(max_seq_len, overlap_percent, preamble_slots)
    min_seq_len = validate_window_floor(min_seq_len, K)
    op = "<" if remove_cutoff_token else "<="
    pairs = [(pid, float(c)) for pid, cs in cutoffs.items() for c in cs]
    con = duckdb.connect()
    try:
        source = event_source(con, parquet_path, event_times, id_col, tok_col, None)
        con.register("_anchors", pa.table({
            id_col: pa.array([pid for pid, _ in pairs]),
            "cutoff": pa.array([c for _, c in pairs], type=pa.float64()),
            "aid": pa.array(range(len(pairs)), type=pa.int64()),
        }))
        is_pre = sql_in(normalized_sql(f'e."{tok_col}"'), event_times.preamble_tokens)
        rows = con.execute(
            f'SELECT COUNT(*) AS n, COUNT(*) FILTER (WHERE {is_pre}) AS n_pre '
            f'FROM {source} AS e '
            f'JOIN _anchors a ON e."{id_col}" = a."{id_col}" '
            f'AND e."{time_age_col}" {op} a.cutoff '
            f'GROUP BY a.aid').fetchnumpy()
    finally:
        con.close()

    total = 0
    for n, n_pre in zip(rows["n"].tolist(), rows["n_pre"].tolist()):
        n_eff = int(n) - int(n_pre)
        if n_eff < min_seq_len:
            continue
        # Same floor the producer uses, or the sizing pass and the build disagree and
        # build_anchor_window_store raises on the count mismatch.
        total += len(history_window_spans(n_eff, K, stride, start_from_end,
                                          single_final=False, min_seq_len=min_seq_len,
                                          add_head_window=add_head_window))
    return total


def build_anchor_window_store(
    windows_dir,
    parquet_path: str,
    vocab_path: str,
    *,
    anchors: dict,
    targets: dict,
    split_by_pid: dict,
    max_seq_len: int,
    use_time_abs: bool,
    use_type: bool,
    overlap_percent: float = 0.0,
    start_from_end: bool = True,
    time_before_event: float = 0.0,
    remove_cutoff_token: bool = True,
    min_seq_len: int = 1,
    add_head_window: bool = False,
    id_col: str = "subject_id",
    tok_col: str = "code",
    time_age_col: str = AGE_COL,
    time_abs_col: str | None = ABS_COL,
    type_col: str | None = "type",
    show_progress: bool = True,
    event_times=None,
) -> str:
    """Write anchor_windows.npy + anchor_patients.parquet + anchor_store_meta.json.

    anchors maps each patient id to a list of anchor ages, and targets maps each
    (patient id, anchor age) to its outcome. The sidecar holds one record per (patient,
    anchor) - patient_idx in the windows file indexes those records.

    Sized exactly up front, then streamed straight to a memory-mapped file, so neither the
    build nor a later read ever holds the cohort in RAM. event_times (load_event_times) is
    passed to the reader.
    """
    from tqdm import tqdm

    windows_dir = Path(windows_dir)
    windows_dir.mkdir(parents=True, exist_ok=True)

    stoi, itos, stats = load_vocab(vocab_path)
    sos_id, con_id, pad_id = int(stoi["<SOS>"]), int(stoi["<CON>"]), int(stoi["<PAD>"])
    preamble = preamble_spec(stats)
    type_to_id = type_id_map(stats) if use_type else None

    cutoffs = {pid: [float(a) - float(time_before_event) for a in ages if not pd.isna(a)]
               for pid, ages in anchors.items()}
    n_windows = _count_windows(
        parquet_path, cutoffs, preamble.slots, id_col=id_col, tok_col=tok_col,
        time_age_col=time_age_col, max_seq_len=max_seq_len,
        overlap_percent=overlap_percent, start_from_end=start_from_end,
        min_seq_len=min_seq_len, remove_cutoff_token=remove_cutoff_token,
        add_head_window=add_head_window, event_times=event_times)

    dtype = _store_dtype(max_seq_len, use_time_abs, use_type)
    mm = np.lib.format.open_memmap(windows_dir / WINDOWS_FILE, mode="w+",
                                   dtype=dtype, shape=(n_windows,))
    # A one-row view reuses fill_window unchanged, so the store and the streaming path
    # cannot produce different windows.
    one = alloc_window_buffers(1, max_seq_len, pad_id, use_time_abs, use_type)

    records, k = [], 0
    it = iter_patient_windows(
        parquet_path, stoi=stoi, preamble=preamble, type_to_id=type_to_id,
        anchors=anchors, max_seq_len=max_seq_len, overlap_percent=overlap_percent,
        start_from_end=start_from_end, time_before_event=time_before_event,
        remove_cutoff_token=remove_cutoff_token, min_seq_len=min_seq_len,
        add_head_window=add_head_window,
        id_col=id_col, tok_col=tok_col, time_age_col=time_age_col,
        time_abs_col=time_abs_col if use_time_abs else None,
        type_col=type_col if use_type else None, event_times=event_times)
    if show_progress:
        it = tqdm(it, desc="[anchor-windows] building", unit="pat", mininterval=2.0)

    for work in it:
        pidx = len(records)
        n_win = len(work.spans)
        for pos, w in enumerate(work.spans):
            ws, we = w["start"], w["end"]
            fill_window(one, 0, work.toks[ws:we], work.ages[ws:we],
                        work.abss[ws:we] if work.abss is not None else None,
                        work.typs[ws:we] if work.typs is not None else None,
                        sos_id if w["first"] else con_id, work.preamble)
            mm["input_ids"][k] = one["input_ids"][0]
            mm["attention_mask"][k] = one["attention_mask"][0]
            mm["time_age"][k] = one["time_age"][0]
            if use_time_abs:
                mm["time_abs"][k] = one["time_abs"][0]
            if use_type:
                mm["type"][k] = one["type"][0]
            mm["patient_idx"][k] = pidx
            mm["window_pos"][k] = pos
            mm["is_final_window"][k] = 1 if pos == n_win - 1 else 0
            k += 1
        rec = patient_record(work, itos)
        rec["target"] = float(targets.get((work.pid, work.anchor_age), np.nan))
        rec["split"] = split_by_pid.get(work.pid)
        records.append(rec)
    mm.flush()

    # Loud failure beats zero-filled records silently becoming garbage embeddings.
    if k != n_windows:
        raise RuntimeError(
            f"anchor window store sizing mismatch: the count pass allocated {n_windows} "
            f"windows but the build wrote {k}. The store at {windows_dir} is unusable.")

    pd.DataFrame(records).to_parquet(windows_dir / PATIENTS_FILE, index=False)
    meta = {
        "max_seq_len": int(max_seq_len),
        "overlap_percent": float(overlap_percent),
        "stride": int(window_geometry(max_seq_len, overlap_percent, preamble.slots)[1]),
        "start_from_end": bool(start_from_end),
        "min_seq_len": int(min_seq_len),
        # Part of the geometry: it decides whether the sub-floor head is dropped or covered
        # by an extra window, so a store built under the other setting holds a different
        # window count for the same cohort.
        "add_head_window": bool(add_head_window),
        "time_before_event": float(time_before_event),
        "remove_cutoff_token": bool(remove_cutoff_token),
        "use_time_abs": bool(use_time_abs),
        "use_type": bool(use_type),
        # Covers the front block too (PREAMBLE_KEYS), so a store of another one is rebuilt.
        "vocab_hash": vocab_fingerprint(vocab_path),
        "anchor_hash": anchor_fingerprint(anchors, split_by_pid),
        "n_windows": int(n_windows),
        "n_anchors": int(len(records)),
    }
    with open(windows_dir / META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[anchor-windows] wrote {n_windows} windows for {len(records)} anchors "
          f"-> {windows_dir}")
    return str(windows_dir)


def open_anchor_window_store(windows_dir) -> tuple[np.memmap, pd.DataFrame]:
    """Memory-map the store and validate its per-patient window layout.

    The layout checks are what make multi-window recombination verifiable rather than
    assumed: a partially written or reordered store can never reach the model.
    """
    windows_dir = Path(windows_dir)
    mm = np.load(windows_dir / WINDOWS_FILE, mmap_mode="c")
    patients = pd.read_parquet(windows_dir / PATIENTS_FILE)

    n_expected = int(patients["n_windows"].sum())
    if len(mm) != n_expected:
        raise ValueError(
            f"{windows_dir}: store holds {len(mm)} windows but the sidecar expects "
            f"{n_expected}.")
    pidx, wpos = mm["patient_idx"], mm["window_pos"]
    if len(mm):
        if pidx[0] != 0 or np.any(np.diff(pidx) < 0):
            raise ValueError(f"{windows_dir}: patient_idx is not non-decreasing from 0.")
        # window_pos must restart at 0 for each patient and step by exactly 1 within it.
        starts = np.concatenate(([0], np.nonzero(np.diff(pidx))[0] + 1))
        if np.any(wpos[starts] != 0):
            raise ValueError(f"{windows_dir}: a patient's windows do not start at pos 0.")
        step = np.diff(wpos)
        same = np.diff(pidx) == 0
        if np.any(step[same] != 1):
            raise ValueError(f"{windows_dir}: window_pos is not contiguous within a patient.")
        counts = np.bincount(pidx, minlength=len(patients))
        if not np.array_equal(counts, patients["n_windows"].to_numpy()):
            raise ValueError(f"{windows_dir}: window counts disagree with the sidecar.")
        finals = np.bincount(pidx, weights=mm["is_final_window"], minlength=len(patients))
        if np.any(finals != 1):
            raise ValueError(f"{windows_dir}: every patient must have exactly one final window.")
    return mm, patients
