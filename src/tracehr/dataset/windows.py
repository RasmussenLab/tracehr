# src/tracehr/dataset/windows.py
"""Model windows: the front block, the window geometry and planning, building one window,
and the on-disk window store.

Every window is [SOS or CON, front block (sex token, static tokens), events..., pads]. The
same functions plan and build the windows of every path - the streaming datasets, the
window store (create_sequence_windows), the anchor-window store of pipeline 05 and the
embedding extraction of pipeline 04 - so they all produce identical windows.
"""

from typing import NamedTuple

import duckdb
import numpy as np

from .cohort import load_patient_id_filter
from .meds_reader import (
    ABS_COL,
    AGE_COL,
    canonical_sort,
    event_source,
    normalized_sql,
    register_ids,
    sql_in,
    stream_encoded_patient_sequences,
)
from .vocabulary import load_vocab, vocab_fingerprint

# The front block of every window: the SOS/CON token, the sex token and the static tokens.
# At most this many positions per patient, so at most MAX_PREAMBLE_LEN - 2 = 8 static tokens;
# a patient with more keeps the first 8 alphabetically, and the run reports it. A new
# vocabulary records this value, and every later run uses the recorded one (with a notice
# when this constant differs).
MAX_PREAMBLE_LEN = 10

_NOTICED = set()


# ============================================================================ #
# The front block
# ============================================================================ #
class PreambleSpec(NamedTuple):
    """The front block a vocabulary defines (see MAX_PREAMBLE_LEN)."""
    sex_ids: np.ndarray     # the vocabulary's sex token ids
    static_ids: np.ndarray  # its static token ids (no sex token among them)
    slots: int              # S: front-block positions every window reserves after SOS/CON
    max_static: int         # static tokens one patient may carry


def preamble_spec(stats: dict) -> PreambleSpec:
    """The front block recorded in a vocabulary's stats. Its recorded limit is used, with a
    notice when the current MAX_PREAMBLE_LEN differs."""
    if "preamble_slots" not in stats:
        raise ValueError("The vocabulary records no front block: it was built before MEDS input. "
                         "Delete it and let pipeline 01 rebuild it.")
    limit = int(stats["max_preamble_len"])
    if limit != MAX_PREAMBLE_LEN and limit not in _NOTICED:
        _NOTICED.add(limit)
        print(f"[windows] NOTICE: the vocabulary was built with MAX_PREAMBLE_LEN {limit}, not this "
              f"module's {MAX_PREAMBLE_LEN}; the vocabulary's value is used. Rebuild the "
              f"vocabulary to use the new one.")
    return PreambleSpec(np.asarray(stats.get("sex_token_ids", []), dtype=np.int64),
                        np.asarray(stats.get("static_token_ids", []), dtype=np.int64),
                        int(stats["preamble_slots"]), limit - 2)


def extract_preamble_tokens(toks, times_age, times_abs, types, spec: PreambleSpec):
    """Remove every sex and static token from one patient's arrays.

    Returns (toks, times_age, times_abs, types, preamble, sex_id): the filtered arrays; the
    front block as (ids, time_age, time_abs, types) arrays of length k; and the patient's sex
    token id (None without one). The front block holds the sex token first (the first
    alphabetically, the lowest id, when there are several), then the distinct static tokens
    sorted by id: at most spec.max_static static tokens and spec.slots tokens in all, the sex
    token kept first. Each keeps the time and type of its first occurrence; the readers give
    these rows the birth moment.
    """
    pre_ids = np.concatenate([spec.sex_ids, spec.static_ids])
    mask = np.isin(toks, pre_ids)
    found, first = np.unique(toks[mask], return_index=True)     # sorted ids, first rows
    rows = np.flatnonzero(mask)[first]
    is_sex = np.isin(found, spec.sex_ids)
    sex_rows = rows[is_sex][:min(1, spec.slots)]
    room = max(0, min(spec.max_static, spec.slots - len(sex_rows)))
    keep = np.concatenate([sex_rows, rows[~is_sex][:room]])      # sex first, then static by id
    preamble = (toks[keep], times_age[keep],
                times_abs[keep] if times_abs is not None else None,
                types[keep] if types is not None else None)
    sex_id = int(toks[sex_rows[0]]) if len(sex_rows) else None
    if not mask.any():
        return toks, times_age, times_abs, types, preamble, sex_id
    rest = ~mask
    return (toks[rest], times_age[rest], times_abs[rest] if times_abs is not None else None,
            types[rest] if types is not None else None, preamble, sex_id)


# ============================================================================ #
# Window geometry and planning
# ============================================================================ #
def window_capacity(max_seq_len: int, preamble_slots: int) -> int:
    """Events one window holds: max_seq_len minus the SOS/CON token and the front block's
    reserved positions. The one place for this formula."""
    return max_seq_len - 1 - preamble_slots


def window_geometry(max_seq_len: int, overlap_percent: float,
                    preamble_slots: int) -> tuple[int, int]:
    """(K, stride): the events one window holds (window_capacity) and the step between
    consecutive windows (K at 0% overlap). The one place for the stride formula."""
    K = window_capacity(max_seq_len, preamble_slots)
    stride = max(1, int(K * (1 - overlap_percent / 100.0)))
    return K, stride


def plan_windows(
    n_tokens: int,
    real_len: int,
    stride: int,
    min_seq_len: int = 1,
    add_final_window: bool = True,
    start_from_end: bool = False,
) -> list[dict]:
    """Plan the window spans covering a sequence of n_tokens.

    Returns a list of dicts with keys: start, end, first (SOS vs CON),
    is_final (chronologically last window), is_extra (extra coverage window).
    Pure function of the sequence length, shared by the window-store and streaming paths.

    `stride` (the step between consecutive windows, real_len at 0% overlap) is
    honoured in both alignments; only the anchor differs — left-aligned walks
    forward from token 0, right-aligned walks backward from the sequence end.
    """
    windows: list[dict] = []

    if start_from_end:
        # Right-aligned: walk backwards so the last window is always full-length,
        # stepping back by `stride` so overlap_percent shifts consecutive windows
        # exactly as it does left-aligned (stride == real_len is the 0%-overlap case).
        # The walk stops once a window reaches the start of the sequence.
        spans: list[tuple[int, int]] = []
        end = n_tokens
        while end > 0:
            start = max(0, end - real_len)
            if end - start >= min_seq_len:
                spans.append((start, end))
            if start == 0:
                break
            end -= stride
        spans.reverse()  # Chronological order

        # Extra head window covering the first real_len tokens if uncovered (also when the
        # floor dropped the head and left a single span)
        has_extra = add_final_window and len(spans) > 0 and spans[0][0] != 0
        if has_extra:
            spans.insert(0, (0, min(real_len, n_tokens)))

        for i, (s, e) in enumerate(spans):
            windows.append({
                "start": s, "end": e,
                # SOS means "this window begins the history", so the flag reads the span's
                # own start, not its position in the list. At min_seq_len > 1 the walk can
                # drop the span that reaches token 0 — a history of K < n < K + min_seq_len
                # plans one window starting at n - K — and a positional flag would then put
                # SOS on a window whose earliest tokens the model cannot see. At most one
                # span per plan starts at 0, so this marks exactly one window when a head
                # window exists and none when it does not, which is also what
                # history_window_spans' single_final branch reports for the same history.
                "first": s == 0,
                "is_final": i == len(spans) - 1,
                "is_extra": has_extra and i == 0,
            })
        return windows

    # Left-aligned: slide forward with the given stride
    start = 0
    first = True
    last_end_unclamped = 0
    while start < n_tokens:
        end = min(start + real_len, n_tokens)
        if end - start < min_seq_len:
            start += stride
            first = False
            continue
        windows.append({"start": start, "end": end, "first": first, "is_final": False, "is_extra": False})
        first = False
        last_end_unclamped = start + real_len
        if last_end_unclamped >= n_tokens:
            break
        start += stride

    # Extra right-aligned tail window when the stride pattern did not land
    # exactly on the sequence end; otherwise the last regular window is final.
    # A single window gets one only when it stops short of the end (the floor
    # dropped the tail); one that reaches the end already covers everything.
    if (add_final_window and last_end_unclamped != n_tokens and len(windows) > 0
            and (len(windows) > 1 or windows[-1]["end"] < n_tokens)):
        windows.append({
            "start": max(0, n_tokens - real_len), "end": n_tokens,
            "first": False, "is_final": True, "is_extra": True,
        })
    elif windows:
        windows[-1]["is_final"] = True
    return windows


def history_window_spans(n_tokens, max_window, stride, start_from_end, single_final,
                         min_seq_len=1, add_head_window=False):
    """The window spans of a truncated history (pipelines 04 and 05): dicts with start, end
    and first (True when the window begins the history: SOS, else CON).

    single_final plans the single most-recent window, of width min(n_tokens, max_window);
    callers gate n_tokens >= min_seq_len beforehand, so it never falls below the floor.
    Otherwise every window of the history [0:n_tokens] comes from plan_windows, the planner
    the training and window-store paths use, so the window counts can never drift apart.

    min_seq_len is the span floor and must be the value the model was trained with. A
    right-aligned walk leaves a head of width r = (n_tokens mod stride); when r is below the
    floor that span is dropped, and add_head_window=True replaces it with a full-width
    [0:max_window) head window. That rescue is off by default because it buys the head
    with duplication: it re-reads (max_window - r) tokens the next span already covers, and
    combine_windows then weights the near-duplicate window like a distinct one, skewing the
    patient embedding. With it off, those r oldest tokens reach bag-of-words but no window
    (see anchor_windows.patient_record). The flag governs the right-aligned head window
    only: a left-aligned plan always keeps its extra tail window.
    """
    if single_final:
        start = n_tokens - min(n_tokens, max_window)
        return [{"start": start, "end": n_tokens, "first": n_tokens <= max_window}]
    return plan_windows(n_tokens, max_window, stride, min_seq_len=min_seq_len,
                        add_final_window=add_head_window if start_from_end else True,
                        start_from_end=start_from_end)


def validate_window_floor(min_seq_len: int, max_window: int) -> int:
    """min_seq_len as an int, after checking it is at least 1 and fits the window.

    A floor wider than the window itself makes plan_windows drop every span, so every
    patient would plan zero windows and the failure would only surface as an empty
    extraction far downstream. Both embedding pipelines (04, 05) check it once, up front,
    against the model's own geometry; it fires only on a hand-assembled mismatch, since
    training built its windows from the same max_seq_len.
    """
    if int(min_seq_len) < 1:
        raise ValueError(f"min_seq_len must be at least 1, got {min_seq_len}.")
    if int(min_seq_len) > int(max_window):
        raise ValueError(
            f"min_seq_len={min_seq_len} exceeds the usable window width "
            f"(max_seq_len - 1 - the front block's slots = {max_window}): every window would "
            f"be dropped and no patient could be embedded. The model was trained with a "
            f"window floor wider than the context it is being used with.")
    return int(min_seq_len)


# ============================================================================ #
# Building one window
# ============================================================================ #
def write_window(ids, time_age, time_abs, types, mask, chunk_toks, chunk_times_age,
                 chunk_times_abs, chunk_types, start_tok: int, preamble: tuple) -> None:
    """Write one window into the given 1-D rows: [start_tok, front block..., chunk tokens...].

    The rows must hold an empty window (ids = <PAD>, everything else 0). preamble is the
    (ids, time_age, time_abs, types) front block from extract_preamble_tokens. time_abs and
    types may be None for a buffer that does not carry them; a None chunk or front-block
    array leaves its row at 0. The one window writer: build_window allocates fresh rows,
    anchor_windows.fill_window writes into preallocated buffers.
    """
    pre_ids, pre_age, pre_abs, pre_type = preamble
    data_start = 1 + len(pre_ids)
    n = len(chunk_toks)
    width = len(ids)
    if n > width - data_start:
        raise ValueError(f"A {n}-event chunk does not fit a {width}-wide window after "
                         f"{data_start} start positions: plan windows with window_geometry().")

    # Position 0: start token (SOS/CON), then the front block, then the real tokens
    ids[0] = start_tok
    mask[0] = 1
    ids[1:data_start] = pre_ids
    time_age[1:data_start] = pre_age
    if time_abs is not None and pre_abs is not None:
        time_abs[1:data_start] = pre_abs
    if types is not None and pre_type is not None:
        types[1:data_start] = pre_type
    mask[1:data_start] = 1

    stop = data_start + n
    ids[data_start:stop] = chunk_toks
    time_age[data_start:stop] = chunk_times_age
    mask[data_start:stop] = 1
    if time_abs is not None and chunk_times_abs is not None:
        time_abs[data_start:stop] = chunk_times_abs
    if types is not None and chunk_types is not None:
        types[data_start:stop] = chunk_types


def build_window(
    chunk_toks,
    chunk_times_age,
    chunk_times_abs,
    chunk_types,
    start_tok: int,
    preamble: tuple,
    max_seq_len: int,
    pad_id: int,
):
    """Build one padded window: [start_tok, front block..., chunk tokens..., pads].

    preamble is the (ids, time_age, time_abs, types) front block from
    extract_preamble_tokens. Returns (ids int32, time_age f64, time_abs f64, type int32,
    mask int8), each of length max_seq_len. Both times stay float64 into the model: float32
    holds adult Ages and absTimes only to ~3-6 minutes, coarser than the 1-minute resolution.
    """
    win_ids = np.full(max_seq_len, pad_id, dtype=np.int32)
    win_time_age = np.zeros(max_seq_len, dtype=np.float64)
    win_time_abs = np.zeros(max_seq_len, dtype=np.float64)
    win_type = np.zeros(max_seq_len, dtype=np.int32)
    win_mask = np.zeros(max_seq_len, dtype=np.int8)
    write_window(win_ids, win_time_age, win_time_abs, win_type, win_mask, chunk_toks,
                 chunk_times_age, chunk_times_abs, chunk_types, start_tok, preamble)
    return win_ids, win_time_age, win_time_abs, win_type, win_mask


# ============================================================================ #
# The window store: one memory-mapped structured .npy (one record per window)
# ============================================================================ #
# Written incrementally and read memory-mapped, so neither the build nor training ever holds
# the whole windowed set in RAM.

# Logical keys exposed to readers (internal geometry/orig_len fields stay hidden).
_STORE_LOGICAL_KEYS = (
    "input_ids", "attention_mask", "time_age", "time_abs", "type",
    "is_first_window", "is_final_window", "is_extra_window", "preamble_len",
    "patient_id", "meta", "patient_original_lengths",
)


def _window_dtype(max_seq_len: int, pid_dtype) -> np.dtype:
    """The store's record: one padded window plus its per-window bookkeeping. The geometry
    constants are repeated per row (negligible next to the (T,) arrays), so the store is
    fully self-describing in a single file."""
    return np.dtype([
        ("input_ids", np.int32, (max_seq_len,)),
        ("attention_mask", np.int8, (max_seq_len,)),
        ("time_age", np.float64, (max_seq_len,)),   # Exact times, as build_window returns them
        ("time_abs", np.float64, (max_seq_len,)),
        ("type", np.int32, (max_seq_len,)),
        ("is_first_window", np.int8),
        ("is_final_window", np.int8),
        ("is_extra_window", np.int8),
        ("preamble_len", np.int8),
        ("patient_id", pid_dtype),
        ("orig_len", np.int32),
        ("_stride", np.int32),
        ("_min_seq_len", np.int32),
        ("_add_final", np.int8),
        ("_start_from_end", np.int8),
        ("_vocab_hash", "U16"),      # vocab_fingerprint (detects a stale vocabulary)
        ("_vocab_size", np.int32),
        ("_preamble_slots", np.int8),    # S, the front-block positions after SOS/CON
        ("_has_types", np.int8),         # 1: real type ids (type column + type map); 0: all 0
    ])


class _WindowStore:
    """NpzFile-like read view over the single structured .npy store, so every
    existing window reader works unchanged after the loader swap."""

    def __init__(self, mm: np.memmap):
        """mm is the memory-mapped structured array of the store."""
        self._mm = mm
        self._names = set(mm.dtype.names or ())
        self.files = [k for k in _STORE_LOGICAL_KEYS
                      if k in self._names or k in ("meta", "patient_original_lengths")]

    def __contains__(self, key):
        """Whether key is one of the store's logical keys."""
        return key in self.files

    def __getitem__(self, key):
        """A logical key's array: a record field, or the rebuilt "meta" /
        "patient_original_lengths" dict (wrapped in a 1-element object array, as np.load
        would return it from an npz)."""
        g = self._mm
        if key == "meta":
            T = g["input_ids"].shape[1] if len(g) else 0
            has_vocab = "_vocab_hash" in self._names
            return np.array([{
                "max_seq_len": int(T),
                "stride": int(g["_stride"][0]) if len(g) else 0,
                "min_seq_len": int(g["_min_seq_len"][0]) if len(g) else 0,
                "add_final_window": bool(g["_add_final"][0]) if len(g) else False,
                "start_from_end": bool(g["_start_from_end"][0]) if len(g) else False,
                "labels_included": False,
                # "" when the field is absent (store predates the vocab fingerprint) so
                # window_npz_geometry_mismatch treats it as stale and regenerates.
                "vocab_hash": (str(g["_vocab_hash"][0]) if has_vocab and len(g) else ""),
                "vocab_size": (int(g["_vocab_size"][0]) if has_vocab and len(g) else 0),
                # None when the field is absent (store predates it): treated as stale.
                "preamble_slots": (int(g["_preamble_slots"][0])
                                   if "_preamble_slots" in self._names and len(g) else None),
                # None when the field is absent (store predates it): unknown, so a run that
                # uses types treats the store as stale.
                "has_types": (bool(g["_has_types"][0])
                              if "_has_types" in self._names and len(g) else None),
            }], dtype=object)
        if key == "patient_original_lengths":
            d = dict(zip(g["patient_id"].tolist(), g["orig_len"].tolist())) if len(g) else {}
            return np.array([d], dtype=object)
        return g[key]


def open_window_store(path):
    """Open a window store: a single structured .npy (memory-mapped, copy-on-write), or an
    .npz written by an older version, which the store checks then report as stale.
    Returns an NpzFile-like object exposing `.files` and `store["key"]`, used by every
    window reader."""
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic == b"\x93NUMPY":
        return _WindowStore(np.load(path, mmap_mode="c"))
    return np.load(path, allow_pickle=True)


def infer_num_types_from_npz(npz_path: str) -> int:
    """Number of type ids a window store's type field needs: its largest id + 1 (0 is the
    pad/missing type), at least 1. A bounded reduction over the memory-mapped field."""
    d = open_window_store(npz_path)
    if "type" not in d.files or d["type"].size == 0:
        return 1
    return max(1, int(np.max(d["type"])) + 1)


def window_npz_geometry_mismatch(
    npz_path: str,
    max_seq_len: int,
    overlap_percent: int,
    min_seq_len: int = 1,
    add_final_window: bool = True,
    start_from_end: bool = False,
    vocab_path: str | None = None,
    *,
    preamble_slots: int,
    uses_types: bool = False,
) -> str | None:
    """Compare an existing window store against the expected window geometry.

    Returns a human-readable description of the first mismatch, or None when the
    file matches. Files without the meta/preamble fields (old format) count as
    mismatched so callers regenerate them instead of feeding stale windows to a
    model configured differently. preamble_slots is the vocabulary's S.

    When `vocab_path` is given, the store's vocabulary fingerprint is also checked:
    a store built against a different vocabulary (stale token ids), or one predating
    the fingerprint field, counts as mismatched. This catches the silent case where
    the vocab was rebuilt without regenerating the windows.

    With `uses_types` (the run embeds types), a store without type ids - built without a
    type column, its type field all 0, or built before stores recorded it - counts as
    mismatched, so a model with types never trains on type 0 for every event.
    """
    d = open_window_store(npz_path)
    if "meta" not in d.files or "is_first_window" not in d.files or "preamble_len" not in d.files:
        return "missing meta/preamble fields (old window-store format)"

    meta = d["meta"][0]
    _, stride = window_geometry(max_seq_len, overlap_percent, preamble_slots)
    expected = {
        "max_seq_len": max_seq_len,
        "preamble_slots": preamble_slots,
        "stride": stride,
        "min_seq_len": min_seq_len,
        "add_final_window": add_final_window,
        "start_from_end": start_from_end,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            return f"{key}={meta.get(key)!r} but the config expects {value!r}"

    if d["input_ids"].shape[1] != max_seq_len:
        return f"window width {d['input_ids'].shape[1]} but the config expects {max_seq_len}"

    if uses_types and not meta.get("has_types"):
        return ("the store holds no type ids (built without a type column, or before stores "
                "recorded it) but this run uses types")

    if vocab_path is not None:
        expected_fp = vocab_fingerprint(vocab_path)
        stored_fp = meta.get("vocab_hash", "")
        if stored_fp != expected_fp:
            return (f"vocab fingerprint {stored_fp!r} but the current vocabulary is "
                    f"{expected_fp!r} (vocabulary changed since these windows were built; "
                    f"token ids may be stale)")
    return None


def window_store_rescoring(npz_path: str) -> str | None:
    """Why a window store would score some tokens more than once, or None if it partitions
    every history (each token in exactly one window).

    Metrics averaged over a store must see each held-out token once. Two geometries break
    that, both useful for TRAINING and both wrong for measurement: overlapping windows
    (stride < tokens per window) and the extra add_final_window tail/head window. The
    re-scored copies sit near the start of a window with little context, so they bias the
    metrics and over-weight long histories. Read from the store's own build metadata.
    """
    d = open_window_store(npz_path)
    if d["input_ids"].shape[0] == 0:
        return None
    if "meta" not in d.files:
        return "it has no build metadata (old format), so its window geometry cannot be verified"
    meta = d["meta"][0]
    real_len = window_capacity(int(meta["max_seq_len"]), int(meta["preamble_slots"]))
    reasons = []
    if int(meta["stride"]) < real_len:
        reasons.append(f"consecutive windows overlap (stride {int(meta['stride'])} < "
                       f"{real_len} tokens per window)")
    if meta.get("add_final_window"):
        reasons.append("it was built with add_final_window=True, whose extra window re-covers "
                       "tokens an earlier window already scored")
    return "; ".join(reasons) or None


# ============================================================================ #
# Writing the window store
# ============================================================================ #
def create_sequence_windows(
    parquet_path: str,
    vocab_path: str,                            # Expects stoi with <SOS>, <CON>, <PAD>, <UNK>
    output_path: str,
    max_seq_len: int = 512,
    overlap_percent: int = 0,                   # 0..99 (% of the window capacity)
    ids_path: str | None = None,                # Optional one-column parquet of patient ids to include
    id_col: str = "subject_id",
    tok_col: str = "code",
    time_age_col: str = AGE_COL,                # the computed time columns (resolve_columns)
    time_abs_col: str | None = ABS_COL,         # Optional, None = don't use
    type_col: str | None = "type",
    sos_token: str = "<SOS>",
    con_token: str = "<CON>",
    pad_token: str = "<PAD>",
    min_seq_len: int = 1,                       # Minimum real-token length for sequences/fragments
    add_final_window: bool = True,              # Add extra window covering last real_len tokens
    start_from_end: bool = False,               # If True, create windows right-aligned (start from end)
    event_times=None,                           # load_event_times: the run's checked time basis
    unk_token: str = "<UNK>",
    ) -> str:
    """Write the window store of the input (a MEDS file or folder) straight from the event
    view, the tokens encoded in SQL: the same events, order and front block as the
    streaming path, so both give byte-identical windows."""
    # Load vocab + special token ids
    stoi, itos, stats = load_vocab(vocab_path)
    vocab_hash = vocab_fingerprint(vocab_path)
    vocab_size = len(stoi)
    sos_id = int(stoi[sos_token])
    con_id = int(stoi[con_token])
    pad_id = int(stoi[pad_token])
    spec = preamble_spec(stats)

    real_len, stride = window_geometry(max_seq_len, overlap_percent, spec.slots)

    ids = load_patient_id_filter(ids_path)

    # ── Pass 1: count windows + resolve patient_id dtype (cheap DuckDB aggregation) ──
    # length after the front block = events - sex/static rows (extract_preamble_tokens removes
    # ALL of them), so plan_windows(length) gives each patient's exact window count.
    con = duckdb.connect()
    try:
        source = event_source(con, parquet_path, event_times, id_col, tok_col, type_col)
        where = register_ids(con, ids, id_col)
        is_pre = sql_in(normalized_sql(f'e."{tok_col}"'), event_times.preamble_tokens)
        agg = (f'SELECT "{id_col}" AS pid, COUNT(*) AS n, SUM(CASE WHEN {is_pre} THEN 1 ELSE 0 END) AS n_pre '
               f'FROM {source} AS e{where} GROUP BY "{id_col}"')
        summary = con.execute(agg).fetchnumpy()
    finally:
        con.close()

    pids = summary["pid"]
    n_windows = 0
    for n, n_pre in zip(summary["n"].tolist(), summary["n_pre"].tolist()):
        n_windows += len(plan_windows(int(n) - int(n_pre), real_len, stride,
                                      min_seq_len, add_final_window, start_from_end))
    pid_dtype = (pids.dtype if np.issubdtype(pids.dtype, np.number)
                 else f"U{max((len(str(x)) for x in pids.tolist()), default=1)}")

    # ── Allocate the single memory-mapped store ───────────────────────────────
    dtype = _window_dtype(max_seq_len, pid_dtype)
    mm = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=(n_windows,))

    # ── Pass 2: stream patients, build windows, write straight to disk ────────
    f_ids, f_mask = mm["input_ids"], mm["attention_mask"]
    f_age, f_abs, f_type = mm["time_age"], mm["time_abs"], mm["type"]
    f_first, f_final, f_extra = mm["is_first_window"], mm["is_final_window"], mm["is_extra_window"]
    f_preamble, f_pid, f_orig = mm["preamble_len"], mm["patient_id"], mm["orig_len"]

    k = 0
    for pid, toks, times_age, times_abs, types in stream_encoded_patient_sequences(
            parquet_path, id_col, tok_col, time_age_col, time_abs_col, type_col,
            stoi=stoi, unk_id=int(stoi[unk_token]), type_to_id=stats.get("type_to_id") or {},
            ids=ids, event_times=event_times):
        # Rows arrive grouped by patient but not ordered within one: sort by the canonical
        # key, which also resolves genuine ties exactly as the streaming path does.
        toks, times_age, times_abs, types = canonical_sort(toks, times_age, times_abs, types)
        orig_len = len(toks)
        toks, times_age, times_abs, types, preamble, _ = extract_preamble_tokens(
            toks, times_age, times_abs, types, spec)
        if len(toks) < min_seq_len:
            continue
        for w in plan_windows(len(toks), real_len, stride, min_seq_len, add_final_window, start_from_end):
            ws, we = w["start"], w["end"]
            win_ids, win_time_age, win_time_abs, win_type, win_mask = build_window(
                toks[ws:we],
                times_age[ws:we],
                times_abs[ws:we] if times_abs is not None else None,
                types[ws:we] if types is not None else None,
                sos_id if w["first"] else con_id,
                preamble,
                max_seq_len,
                pad_id,
            )
            f_ids[k] = win_ids
            f_mask[k] = win_mask
            f_age[k] = win_time_age
            f_abs[k] = win_time_abs
            f_type[k] = win_type
            f_first[k] = 1 if w["first"] else 0
            f_final[k] = 1 if w["is_final"] else 0
            f_extra[k] = 1 if w["is_extra"] else 0
            f_preamble[k] = 1 + len(preamble[0])
            f_pid[k] = pid
            f_orig[k] = orig_len
            k += 1

    # Geometry constants (repeated per row) let window_npz_geometry_mismatch read
    # back the build config from the store itself.
    mm["_stride"][:] = stride
    mm["_min_seq_len"][:] = min_seq_len
    mm["_add_final"][:] = 1 if add_final_window else 0
    mm["_start_from_end"][:] = 1 if start_from_end else 0
    mm["_vocab_hash"][:] = vocab_hash
    mm["_vocab_size"][:] = vocab_size
    mm["_preamble_slots"][:] = spec.slots
    mm["_has_types"][:] = 1 if type_col is not None and stats.get("type_to_id") else 0
    mm.flush()

    return output_path
