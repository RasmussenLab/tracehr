# src/tracehr/probing/embeddings.py
"""Sequence embeddings from a trained NTPTransformer: the embedding of a batch of windows
(pooled by pooling.py), the combination of a patient's windows into one vector, and the
anchor-grid extraction of pipeline 04 with the run-settings sidecar it shares with
pipeline 05."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from ..dataset.cohort import cutoff_index, validate_cutoff
from ..dataset.meds_reader import ABS_COL, AGE_COL, canonical_sort, stream_patient_sequences
from ..dataset.vocabulary import encode_tokens, encode_types, load_vocab, type_id_map
from ..dataset.windows import (
    build_window,
    extract_preamble_tokens,
    history_window_spans,
    preamble_spec,
    validate_window_floor,
    window_geometry,
)
from .pooling import pool_hidden_states

# ============================================================================ #
# Pooling a window's hidden states
# ============================================================================ #
# Token poolings of extract_sequence_embedding; the finetuning head adds "attention".
POOLING_METHODS = ("mean", "max", "last", "combined")


@torch.no_grad()
def extract_sequence_embedding(
    model,
    input_ids: torch.Tensor,
    time_age: torch.Tensor | None = None,
    time_abs: torch.Tensor | None = None,
    tok_type: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    pooling: str = "mean"
) -> torch.Tensor:
    """[B, D] embeddings of a batch of windows ([B, 3D] for "combined"): the model's own
    forward pass up to the final hidden states, then the token pooling `pooling` over
    attention_mask (all ones when None). time_age, time_abs and tok_type join the batch
    only when given."""
    model.eval()
    device = next(model.parameters()).device

    # Build a batch and reuse the model's own forward pass, so extracted
    # embeddings always match exactly what the trained model computes
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask.to(device)}
    if time_age is not None:
        batch["time_age"] = time_age.to(device)
    if time_abs is not None:
        batch["time_abs"] = time_abs.to(device)
    if tok_type is not None:
        batch["type"] = tok_type.to(device)

    hidden_states = model(batch, hidden_only=True)["hidden_states"]  # [B, T, D]
    return pool_hidden_states(hidden_states, batch["attention_mask"], pooling)  # [B, D] ([B, 3D] combined)


# ============================================================================ #
# Combining a patient's windows
# ============================================================================ #
# The complete set of window-combination methods, and the one every caller validates
# against - extract_anchor_embeddings and extract_anchor_grid_embeddings both reject
# anything outside it, so the function accepts exactly what a config may ask for.
COMBINE_METHODS = ("mean", "max", "recency_weighted", "none")


def combine_window_embeddings(
        embeddings: np.ndarray,
        method: str = "mean",
) -> np.ndarray:
    """Reduce a patient's [n_windows, D] embeddings to one [D] vector.

    Windows arrive oldest -> newest (see fill_patient_windows), which is what "none" and
    the recency weighting below both depend on.
    """
    # Validated BEFORE the single-window shortcut: otherwise a misspelled method sails
    # through for every patient who happens to have exactly one window, and only starts
    # raising on the first patient with two.
    if method not in COMBINE_METHODS:
        raise ValueError(f"combine_windows must be one of {'|'.join(COMBINE_METHODS)}, "
                         f"got {method!r}.")

    n_windows = embeddings.shape[0]

    # Single window - no combination needed (and [0] is also [-1], so "none" agrees)
    if n_windows == 1:
        return embeddings[0]

    if method == "none":       # the most-recent window only; nothing is combined
        return embeddings[-1]
    if method == "mean":
        return embeddings.mean(axis=0)
    if method == "max":
        return embeddings.max(axis=0)
    # recency_weighted: exponential weighting, later (more recent) windows weigh more.
    # Window 0 (earliest) -> weight ~ 1.0, window -1 (most recent) -> ~ 7.4.
    weights = np.exp(np.linspace(0, 2, n_windows))
    weights = weights / weights.sum()
    return np.average(embeddings, axis=0, weights=weights)


def validate_extraction_settings(combine_windows: str, pooling: str, overlap_percent: float) -> None:
    """Check the window settings both embedding extractors take (pipelines 04 and 05)."""
    if combine_windows not in COMBINE_METHODS:
        raise ValueError(f"combine_windows must be one of {'|'.join(COMBINE_METHODS)}, "
                         f"got {combine_windows!r}.")
    if pooling not in POOLING_METHODS:
        raise ValueError(f"pooling must be one of {'|'.join(POOLING_METHODS)}, got {pooling!r}.")
    if not (0 <= overlap_percent < 100):
        raise ValueError(f"overlap_percent must be in [0, 100), got {overlap_percent}.")


# ============================================================================ #
# Run-settings sidecar, shared by pipelines 04 and 05
# ============================================================================ #
EXTRACTION_META_FILE = "embedding_extraction_meta.json"


def write_extraction_meta(output_dir, emb_df: pd.DataFrame | None, **settings) -> Path:
    """Write the settings that describe the whole extraction beside its parquet.

    These were columns once (token_pooling, window_scope, window_aggregation, embedding_dim,
    hidden_state_source), repeated identically on every row. They describe the RUN, not the
    patient, so one file per output beats one value per row. Both embedding pipelines call
    this so the two sidecars have the same name and shape.

    embedding_dim is read off the produced embeddings rather than the model config, which
    keeps it correct for poolings that change the width (e.g. "combined"). Pipeline 04
    streams its rows to disk instead of holding them, so it passes emb_df=None with n_rows
    and embedding_dim in settings, as counted while writing.
    """
    meta = dict(settings)
    meta["window_scope"] = ("final_window" if settings.get("combine_windows") == "none"
                            else "all_windows")
    meta["hidden_state_source"] = "final_layer_hidden_states"
    if emb_df is not None:
        meta["embedding_dim"] = int(len(emb_df["Embedding"].iloc[0])) if len(emb_df) else None
        meta["n_rows"] = int(len(emb_df))
    path = Path(output_dir) / EXTRACTION_META_FILE
    # default=str: the settings dicts carry Paths and enums that json refuses on its own.
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[embeddings] extraction settings -> {path}")
    return path


# ============================================================================ #
# Anchor-grid embedding extraction (pipeline 04)
# ============================================================================ #
def _embed_windows(model, win_specs, n_tasks, *, batch_size, pooling, use_abs, use_type,
                   desc=None):
    """Per-task lists of pooled embeddings of a flat list of (task_index, built_window), in
    chronological window order: stacks each batch and runs extract_sequence_embedding,
    passing time_abs and the types only when the model reads them."""
    task_embs = [[] for _ in range(n_tasks)]
    # `desc` labels the forward-pass bar; None keeps callers that never showed one silent.
    for lo in tqdm(range(0, len(win_specs), batch_size), desc=desc, unit="batch",
                   mininterval=2.0, disable=desc is None, leave=False):
        batch = win_specs[lo:lo + batch_size]
        wins = [w for _, w in batch]
        emb = extract_sequence_embedding(
            model,
            input_ids=torch.from_numpy(np.stack([w[0] for w in wins])).to(torch.int64),
            attention_mask=torch.from_numpy(np.stack([w[4] for w in wins])).to(torch.int64),
            time_age=torch.from_numpy(np.stack([w[1] for w in wins])) if model.needs_time_age else None,
            time_abs=torch.from_numpy(np.stack([w[2] for w in wins])) if use_abs else None,
            tok_type=torch.from_numpy(np.stack([w[3] for w in wins])).to(torch.int64) if use_type else None,
            pooling=pooling).cpu().numpy()
        for (ti, _w), e in zip(batch, emb):
            task_embs[ti].append(e)
    return task_embs


def _append_rows(writer, path, rows):
    """Append one chunk of output rows to the parquet file at path, opening the writer on the
    first chunk; returns the writer. That chunk fixes the column types, except Sex, which is
    always a string column: a chunk in which no patient has a sex token would otherwise give
    a null column, and the writer would then refuse every later chunk that does have one."""
    if writer is None:
        schema = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False).schema
        schema = schema.set(schema.get_field_index("Sex"), pa.field("Sex", pa.string()))
        writer = pq.ParquetWriter(path, schema)
    writer.write_table(pa.Table.from_pandas(pd.DataFrame(rows), schema=writer.schema,
                                            preserve_index=False))
    return writer


def _anchor_tasks(patient: tuple, anchor0, *, timestep: float | None, time_before_event: float,
                  remove_cutoff_token: bool, min_seq_len: int):
    """The tasks one anchor yields, in grid order: patient's (toks, ages, abss, typs, front,
    sex_id, pid) followed by (anchor_age, cutoff, step_index, cut index, is_final_step).

    A missing (NaN) anchor yields one task over the full history (the history pipeline 03
    generates from). Otherwise each grid anchor comes from the closed float64 formula; a step
    whose cut keeps fewer than min_seq_len events yields nothing and the grid continues. The
    walk stops at the first cut that takes the whole history - the same test is_final_step
    records, so the two cannot disagree. A stop step solved in advance rounds differently
    from the anchors on a fractional grid (0.1 days).
    """
    toks, ages = patient[0], patient[1]
    if pd.isna(anchor0):
        yield (*patient, float("nan"), float("nan"), 0, len(toks), True)
        return
    anchor0 = float(anchor0)
    grid = bool(timestep)
    s = 0
    while True:
        anchor_s = np.float64(anchor0) + np.float64(s) * np.float64(timestep or 0.0)
        cutoff = float(anchor_s) - float(time_before_event)
        idx = cutoff_index(ages, cutoff, remove_cutoff_token)
        if idx >= min_seq_len:
            yield (*patient, float(anchor_s), cutoff, s, idx, idx == len(toks))
        if not grid or idx == len(toks):
            break
        s += 1


def _task_row(t: tuple, embedding: np.ndarray, n_windows: int, *, itos: list[str], id_col: str,
              timestep: float | None, save_sequences: bool) -> dict:
    """The output row of task t (see _anchor_tasks) and its combined embedding.

    t[0]/t[1] are the patient's full arrays and t[10] is the cut index, so t[0][:t[10]] is
    exactly this step's truncated history - the same span the embedding covers unless
    combine_windows is "none". t[4] is extract_preamble_tokens' front block and t[5] the sex
    token id, so neither the sex nor the static tokens are in t[0].
    """
    grid = bool(timestep)
    hist_toks, hist_ages = t[0][:t[10]], t[1][:t[10]]
    hist_ids, hist_counts = np.unique(hist_toks, return_counts=True)
    sex_id, front_ids = t[5], t[4][0]
    row = {
        id_col: t[6],
        "anchor_age": t[7],
        # The newest surviving event, NOT the truncation boundary: the boundary is
        # anchor_age - time_before_event, is pure config arithmetic, and is always
        # strictly greater than this (the gap is however long the patient went quiet).
        "last_token_age": float(hist_ages[-1]),
        "Sex": itos[sex_id] if sex_id is not None else None,
        "Embedding": embedding.astype(np.float32),
        # Token names rather than ids, and the front block re-added with count 1 - the same
        # recipe as anchor_windows.patient_record, so the two pipelines' BoW columns are
        # directly comparable without a vocabulary to decode them.
        "bow_tokens": [itos[int(i)] for i in hist_ids] + [itos[int(i)] for i in front_ids],
        "bow_token_counts": [int(c) for c in hist_counts] + [1] * len(front_ids),
        "n_windows": n_windows,
        # Grid-only provenance; the single-anchor mode leaves them at step 0.
        "step_index": t[9],
        "timestep": float(timestep) if grid else None,
        "anchor_offset": float(t[9]) * float(timestep) if grid else 0.0,
        "is_final_step": bool(t[11]),
    }
    if save_sequences:
        row["Sequence"] = [itos[int(x)] for x in hist_toks]
        row["Age"] = [float(a) for a in hist_ages]
    return row


@torch.no_grad()
def extract_anchor_grid_embeddings(
    model,
    parquet_path: str,
    vocab_path: str,
    anchors_by_pid: dict,
    *,
    timestep: float | None = None,
    pooling: str = "mean",
    overlap_percent: float = 0.0,
    start_from_end: bool = True,
    combine_windows: str = "recency_weighted",
    batch_size: int = 64,
    time_before_event: float = 0.0,
    remove_cutoff_token: bool = True,
    save_sequences: bool = False,
    min_seq_len: int = 1,
    add_head_window: bool = False,
    id_col: str = "subject_id",
    tok_col: str = "code",
    time_age_col: str = AGE_COL,
    time_abs_col: str = ABS_COL,
    type_col: str = "type",
    out_path: str | None = None,
    event_times=None,
) -> pd.DataFrame | dict:
    """One embedding per (patient, anchor, grid step): anchors advance as anchor_0 + s*timestep (DAYS).

    anchors_by_pid maps each patient id to a list of anchor ages (anchors_by_patient); every
    anchor gets its own row, or its own grid, cut from one read of the patient's history. A
    missing (NaN) anchor gives one row over the full history - anchor_age NaN, step_index 0,
    is_final_step True - in grid mode too, since a grid needs a starting age.
    DAYS CONTRACT: anchors, timestep and time_age are all in DAYS.
    Cutoff (shared contract with pipelines 03/05/06): kept events satisfy
    Age < (anchor - time_before_event); the event exactly AT the anchor age is
    EXCLUDED. remove_cutoff_token=False flips to <=.
    Grid: timestep None/0 -> only step 0. Positive timestep -> steps 0,1,2,...,
    each anchor computed from the closed float64 formula (no accumulation drift);
    stall steps (no new events) are preserved; the run stops at the FIRST grid
    anchor whose cutoff consumed the complete history (emitted, is_final_step=True).
    Steps whose truncation leaves zero clinical events are skipped; the grid continues.

    Windowing: each grid step's truncated history is split into windows of width
    model.max_seq_len (overlap_percent, start_from_end) and reduced to one embedding by
    combine_windows (mean|max|recency_weighted); combine_windows="none" keeps only the
    single most-recent window. `pooling` is the within-window token pooling. The output
    always has one row per (patient, anchor, grid step); n_windows records how many windows fed it.

    min_seq_len is the window floor, and it must be the value the model was trained with
    (the pipeline reads it off the checkpoint). It drops grid steps whose truncated history
    is shorter than one trainable window, and stops the right-aligned plan from emitting a
    narrow head window that the model never saw during training and that would otherwise be
    combined into the row's embedding at full weight. At 1 every span is kept, the narrow
    head window included.

    add_head_window covers what happens to that dropped head. False (the default) leaves the
    oldest (n mod stride) tokens in no window at all; True restores them by prepending a
    full-width [0:K) window, which necessarily re-reads tokens the next window already
    covers and gives that near-duplicate the same weight under combine_windows. Off is the
    better trade at the usual geometry - see history_window_spans - so it is the default.
    Right-aligned plans only; start_from_end=False ignores it.

    Columns match pipeline 05's downstream_embeddings.parquet wherever they mean the same
    thing (id, anchor_age, last_token_age, Sex, Embedding, bow_tokens, bow_token_counts,
    n_windows, and the opt-in Sequence/Age); step_index/timestep/anchor_offset/is_final_step
    are the grid-only extras. Settings that describe the run rather than the patient live in
    the sidecar the pipeline writes, not in a column repeated on every row.

    save_sequences writes the full truncated history per row. In grid mode that is one copy
    per step, so it is off by default.

    out_path streams the rows to that parquet file a chunk at a time instead of holding them
    all, and returns {"n_rows", "embedding_dim"} (what the sidecar needs) in place of the
    DataFrame. The file is written as <out_path>.partial and renamed only once complete, so
    a failed run never leaves a truncated file under the real name. Left at None, the rows
    come back as a DataFrame.
    """
    if timestep is not None and not (timestep >= 0 and np.isfinite(timestep)):
        raise ValueError(f"timestep must be a finite number >= 0 (DAYS), got {timestep}.")
    grid = bool(timestep)  # None or 0 -> single-anchor mode

    # Windowing / combination controls. combine_windows == "none" keeps only the single
    # most-recent window; the other methods tile the whole truncated history into windows
    # and reduce them to one patient-level embedding.
    validate_extraction_settings(combine_windows, pooling, overlap_percent)
    time_before_event, remove_cutoff_token = validate_cutoff(time_before_event, remove_cutoff_token)

    model.eval()
    stoi, itos, stats = load_vocab(vocab_path)
    sos_id, con_id, pad_id = stoi["<SOS>"], stoi["<CON>"], stoi["<PAD>"]
    unk_id = int(stoi["<UNK>"])
    preamble = preamble_spec(stats)

    use_abs = model.use_time_abs
    use_type = model.use_type
    type_to_id = type_id_map(stats) if use_type else {}

    # Window width after the start token and the front block; max_seq_len always from the model
    K, stride = window_geometry(model.max_seq_len, overlap_percent, preamble.slots)
    min_seq_len = validate_window_floor(min_seq_len, K)
    n_anchors = sum(len(a) for a in anchors_by_pid.values())  # every anchor asked for
    n_found, dropped = 0, 0  # anchors of patients in the input; of those, anchors with no row

    def _task_spans(idx):
        """The window spans of a history cut at idx, from the shared planner: the single
        most-recent window for combine_windows "none", else every window of [0:idx]."""
        return history_window_spans(idx, K, stride, start_from_end,
                                    combine_windows == "none", min_seq_len=min_seq_len,
                                    add_head_window=add_head_window)

    # Windows are built, embedded and reduced to rows a chunk at a time as the stream
    # advances, so peak memory is ~32 forward batches of windows (plus the output rows)
    # rather than the whole cohort. A task is never split across chunks, so every row
    # still combines all of its windows; rows keep the stream order.
    flush_at = 32 * batch_size
    tasks, win_specs, rows = [], [], []
    repeats = set()    # chunk tasks that repeat the previous task's cut (see _add_task)
    last_cut = None    # (patient, cut index) of the previous task
    last_emb = None    # (embedding, n_windows) of the previous row, which a repeat reuses
    n_rows, writer = 0, None
    partial_path = f"{out_path}.partial"

    def _add_task(t):
        """Queue task t = (toks, ages, abss, typs, front, sex_id, pid, anchor_age, cutoff,
        step_index, cut index, is_final_step) and its built windows, as (task_index, window)
        pairs in chronological order per task; flush once the chunk is full."""
        nonlocal last_cut
        ti = len(tasks)
        tasks.append(t)
        # Same patient and cut index as the previous task - a grid step with no new event, or
        # two anchors cutting at the same event. Its windows would be identical, so its row
        # reuses the previous row's embedding instead of embedding them again.
        cut = (t[6], t[10])
        if cut == last_cut:
            repeats.add(ti)
        else:
            last_cut = cut
            for w in _task_spans(t[10]):
                ws, we = w["start"], w["end"]
                win = build_window(
                    t[0][ws:we], t[1][ws:we],
                    t[2][ws:we] if t[2] is not None else None,
                    t[3][ws:we] if t[3] is not None else None,
                    sos_id if w["first"] else con_id,
                    t[4], model.max_seq_len, pad_id)
                win_specs.append((ti, win))
        # A repeat adds a row but no window, so the row count bounds a chunk as well.
        if len(win_specs) >= flush_at or len(tasks) >= flush_at:
            _flush()

    def _flush():
        """Embed the queued windows, reduce each task to its output row, append the rows to
        the file (out_path) or keep them, and empty the queue."""
        nonlocal last_emb, n_rows, writer
        # Batch-embed every window; gather per task in chronological order.
        task_embs = _embed_windows(
            model, win_specs, len(tasks),
            batch_size=batch_size, pooling=pooling, use_abs=use_abs, use_type=use_type)

        # One row per task: reduce its windows to a single patient-level embedding.
        for ti, t in enumerate(tasks):
            if ti in repeats:
                combined, n_win = last_emb
            else:
                embs = np.stack(task_embs[ti])
                n_win = int(embs.shape[0])
                # "none" goes through the combiner too, which returns the most-recent window.
                # _task_spans plans exactly one window for "none", so this is that window; were
                # the plan ever to hold more, "none" would still mean the most recent one.
                combined = combine_window_embeddings(embs, combine_windows)
            last_emb = (combined, n_win)
            rows.append(_task_row(t, combined, n_win, itos=itos, id_col=id_col,
                                  timestep=timestep, save_sequences=save_sequences))
        n_rows += len(tasks)
        if out_path is not None and rows:
            writer = _append_rows(writer, partial_path, rows)
            rows.clear()
        tasks.clear()
        win_specs.clear()
        repeats.clear()

    # Stream the anchored patients' events ordered by (id, age, token), then encode them and
    # put them in canonical order (event_times: load_event_times' births and dropped patients).
    for pid, tok_vals, ages, abss, typs in stream_patient_sequences(
            parquet_path, id_col, tok_col, time_age_col,
            time_abs_col if use_abs else None, type_col if use_type else None,
            ids=sorted(anchors_by_pid.keys(), key=str), event_times=event_times):
        # A missing time arrives as NaN and would make every embedding that includes the
        # event NaN, so refuse it here, before the patient reaches the model.
        for col, vals in ((time_age_col, ages), (time_abs_col, abss)):
            if vals is not None and not np.isfinite(vals).all():
                raise ValueError(
                    f"Patient {pid!r} has {int((~np.isfinite(vals)).sum())} event(s) with a "
                    f"missing or non-finite '{col}' in {parquet_path}. Fix or drop those rows "
                    f"before extracting embeddings.")
        patient_anchors = anchors_by_pid[pid]
        n_found += len(patient_anchors)
        toks = encode_tokens(tok_vals, stoi, unk_id)
        typs = encode_types(typs, type_to_id)
        toks, ages, abss, typs = canonical_sort(toks, ages, abss, typs)
        toks, ages, abss, typs, front, sex_id = extract_preamble_tokens(toks, ages, abss, typs, preamble)
        if len(toks) < min_seq_len:
            # Too little history to fill one trainable window. At the default floor of 1
            # this drops only the patients with no clinical events at all.
            dropped += len(patient_anchors)
            continue
        for anchor0 in patient_anchors:
            emitted = False
            for task in _anchor_tasks((toks, ages, abss, typs, front, sex_id, pid), anchor0,
                                      timestep=timestep, time_before_event=time_before_event,
                                      remove_cutoff_token=remove_cutoff_token,
                                      min_seq_len=min_seq_len):
                _add_task(task)
                emitted = True
            if not emitted:
                dropped += 1  # every step's cut left fewer than min_seq_len events
    _flush()  # the last, partial chunk

    missing = n_anchors - n_found
    print(f"[embeddings] {n_rows} rows from {n_anchors - missing - dropped} of {n_anchors} anchors "
          f"({missing} dropped: patient not in the input; {dropped} dropped: fewer than "
          f"min_seq_len={min_seq_len} clinical event(s) before the cutoff); "
          f"mode={'grid(timestep=' + str(timestep) + ' days)' if grid else 'single-anchor'}")
    # Zero rows would otherwise give a frame (and parquet) with no columns at all.
    empty = pd.DataFrame(columns=[id_col, "anchor_age", "last_token_age", "Sex", "Embedding",
                                  "bow_tokens", "bow_token_counts", "n_windows", "step_index",
                                  "timestep", "anchor_offset", "is_final_step"]
                                 + (["Sequence", "Age"] if save_sequences else []))
    if out_path is None:
        return pd.DataFrame(rows) if rows else empty
    if writer is None:
        empty.to_parquet(partial_path, index=False)  # no rows: an empty file with the row columns
    else:
        writer.close()
    os.replace(partial_path, out_path)
    return {"n_rows": n_rows, "embedding_dim": len(last_emb[0]) if last_emb else None}
