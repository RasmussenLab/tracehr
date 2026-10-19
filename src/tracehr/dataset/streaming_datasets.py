# src/tracehr/dataset/streaming_datasets.py
"""The streaming datasets: windows built on the fly from the MEDS input, no window store.

RawStreamingNTPDataset yields the next-token training windows of every patient (pipelines
01 and 02); StreamingAnchorsDataset yields the windows of histories cut at anchors, with
their outcome labels (pipelines 03 and 06). Both read through meds_reader's event view and
build windows with windows.py's functions, so a streamed window equals the stored one.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

from .cohort import cutoff_index, load_patient_id_filter, validate_cutoff
from .meds_reader import (
    ABS_COL,
    AGE_COL,
    PatientEventReader,
    canonical_sort,
    duckdb_default_memory_limit,
    parquet_files,
    parquet_id_class,
    stream_patient_sequences,
    warn_precluster,
)
from .ntp_batching import (
    build_ntp_labels,
    compute_time_deltas,
    mask_preamble_targets,
    shuffle_simultaneous_events,
    t2e_delta_mask_len,
)
from .vocabulary import encode_tokens, encode_types, load_vocab, type_id_map
from .windows import (
    build_window,
    extract_preamble_tokens,
    plan_windows,
    preamble_spec,
    window_geometry,
)


def _pandas_id_class(series):
    """Classify a pandas id column as 'numeric', 'string', or None, mirroring
    parquet_id_class so the input's id and the outcomes-file id can be checked
    for a join-breaking dtype-class mismatch."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return "numeric"
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    val = non_null.iloc[0]
    if isinstance(val, str):
        return "string"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return "numeric"
    return None


def _sorted_unique_patient_ids(parquet_path: str, id_col: str) -> np.ndarray:
    """Every patient id of the input, sorted (by their string form), for a run without an
    id file."""
    values = set()
    dataset = ds.dataset(parquet_files(parquet_path), format="parquet")   # the readers' files
    scanner = dataset.scanner(columns=[id_col])
    for batch in scanner.to_batches():
        values.update(batch.column(id_col).to_pylist())
    return np.asarray(sorted(values, key=lambda value: str(value)))


def _worker_memory_limit() -> str | None:
    """DuckDB memory limit for one DataLoader worker's stream: an equal share of DuckDB's own
    default for one connection (duckdb_default_memory_limit), so the workers together stay
    within the memory the job may use - the RAM, or a SLURM allocation or container limit when
    that is smaller - and spill to disk instead. None outside a worker, with a single worker, or
    when the default can't be read (each worker then gets DuckDB's default)."""
    worker = get_worker_info()
    if worker is None or worker.num_workers <= 1:
        return None
    total = duckdb_default_memory_limit()
    if total is None:
        return None
    share = total // worker.num_workers // 2 ** 20
    if worker.id == 0:
        print(f"[memory] DuckDB limit per DataLoader worker: {share} MB "
              f"(DuckDB's default {total / 2 ** 30:.1f} GiB over {worker.num_workers} workers)")
    return f"{share}MB"


class RawStreamingNTPDataset(IterableDataset):
    """Next-token training windows streamed from the MEDS input, with no window store.

    Each patient's events are read through the run's event view (event_times), encoded with
    the vocabulary, sorted canonically, split into windows (max_seq_len, overlap_percent,
    min_seq_len, add_final_window, start_from_end) and turned into the items NTPDataset
    yields for the same windows. DataLoader workers each stream their own share of the
    patients; shuffle_buffer_windows shuffles within a buffer of that many windows, and
    shuffle_simultaneous reorders the events that share a time_age (training only).
    """

    def __init__(
        self,
        parquet_path: str,
        vocab_path: str,
        ids_path: str | None = None,
        max_seq_len: int = 512,
        overlap_percent: int = 0,
        id_col: str = "subject_id",
        tok_col: str = "code",
        time_age_col: str = AGE_COL,
        time_abs_col: str | None = ABS_COL,
        type_col: str | None = "type",
        sos_token: str = "<SOS>",
        con_token: str = "<CON>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        min_seq_len: int = 1,
        add_final_window: bool = True,
        start_from_end: bool = False,
        return_metadata: bool = False,
        compute_time_deltas: bool = False,
        shuffle_buffer_windows: int = 0,
        seed: int = 0,
        shuffle_simultaneous: bool = False,  # Train-time: reorder events sharing the same Age per window
        event_times=None,                    # load_event_times: the run's checked time basis
    ):
        """Load the vocabulary and resolve the window geometry and the patient list (the id
        file, or every patient of the input); the events are read only when iterating."""
        if compute_time_deltas:
            print("[t2e] RawStreamingNTPDataset time_deltas from time_age")

        stoi, itos, stats = load_vocab(vocab_path)
        for token in (sos_token, con_token, pad_token, unk_token):
            if token not in stoi:
                raise KeyError(f"Required token {token!r} is missing from vocabulary {vocab_path}.")

        self.parquet_path = parquet_path
        self.id_col = id_col
        self.tok_col = tok_col
        self.time_age_col = time_age_col
        self.time_abs_col = time_abs_col
        self.type_col = type_col
        self.event_times = event_times
        self.stoi = stoi
        self.sos_id = int(stoi[sos_token])
        self.con_id = int(stoi[con_token])
        self.pad_id = int(stoi[pad_token])
        self.unk_id = int(stoi[unk_token])
        self.preamble = preamble_spec(stats)

        self.max_seq_len = max_seq_len
        self.real_len, self.stride = window_geometry(max_seq_len, overlap_percent,
                                                     self.preamble.slots)
        self.min_seq_len = min_seq_len
        self.add_final_window = add_final_window
        self.start_from_end = start_from_end
        self.return_metadata = return_metadata
        self.compute_time_deltas = compute_time_deltas
        self.shuffle_buffer_windows = max(0, int(shuffle_buffer_windows))
        self.shuffle_simultaneous = shuffle_simultaneous
        self._rng = None       # Lazy: one RNG per process (main or DataLoader worker)
        self._rng_seed = None  # The torch.initial_seed() it was built from; see _iter_items
        self._shuffle_epoch = 0  # Mixed into the shuffle seed; see _shuffle_items
        self.seed = int(seed)
        warn_precluster(parquet_path)

        self.patient_ids = (
            load_patient_id_filter(ids_path)
            if ids_path else _sorted_unique_patient_ids(parquet_path, id_col)
        )

        if self.type_col is not None:
            self.type_to_id = type_id_map(stats)
            if not self.type_to_id:
                raise ValueError(
                    "Vocabulary stats are missing 'type_to_id'. Rebuild the vocabulary with build_vocab(...) "
                    "before streaming with a type column."
                )
            self.num_types = len(self.type_to_id) + 1
        else:
            self.type_to_id = {}
            self.num_types = 0

        self._length: int | None = None
        self.patient_original_lengths = {}

    def __len__(self) -> int:
        """The number of windows of one epoch, counted once by a pass over the data."""
        if self._length is None:
            self._length = self._count_windows()
        return self._length

    def __iter__(self) -> Iterator[dict]:
        """This worker's items, through the shuffle buffer."""
        patient_ids = self._worker_patient_ids()
        items = self._iter_items(patient_ids)
        yield from self._shuffle_items(items)

    def _worker_patient_ids(self) -> np.ndarray:
        """The patients this DataLoader worker streams (all of them outside a worker)."""
        worker = get_worker_info()
        if worker is None:
            return self.patient_ids
        return self.patient_ids[worker.id::worker.num_workers]

    def _iter_patient_arrays(self, patient_ids: np.ndarray):
        """(pid, toks, age, abs, types) per patient: one ordered DuckDB stream over these ids,
        encoded per patient and sorted canonically, as the window store's build does."""
        for pid, tok_vals, age, abs_time, type_vals in stream_patient_sequences(
                self.parquet_path, self.id_col, self.tok_col, self.time_age_col,
                self.time_abs_col, self.type_col, ids=np.asarray(patient_ids),
                event_times=self.event_times, memory_limit=_worker_memory_limit()):
            toks = encode_tokens(tok_vals, self.stoi, self.unk_id)
            types = encode_types(type_vals, self.type_to_id)
            yield (pid, *canonical_sort(toks, age, abs_time, types))

    def _count_windows(self) -> int:
        """The number of windows of all patients (a pass that plans every patient's windows)."""
        count = 0
        for pid, toks, times_age, times_abs, types in self._iter_patient_arrays(self.patient_ids):
            self.patient_original_lengths[pid] = len(toks)
            toks, times_age, times_abs, types, _, _ = extract_preamble_tokens(
                toks, times_age, times_abs, types, self.preamble)
            if len(toks) < self.min_seq_len:
                continue
            count += len(plan_windows(
                len(toks),
                self.real_len,
                self.stride,
                self.min_seq_len,
                self.add_final_window,
                self.start_from_end,
            ))
        return count

    def _iter_items(self, patient_ids: np.ndarray) -> Iterator[dict]:
        """The training items of these patients' windows, in order."""
        window_index = 0
        for pid, toks, times_age, times_abs, types in self._iter_patient_arrays(patient_ids):
            self.patient_original_lengths[pid] = len(toks)
            toks, times_age, times_abs, types, preamble, _ = extract_preamble_tokens(
                toks, times_age, times_abs, types, self.preamble)

            if len(toks) < self.min_seq_len:
                continue

            for window in plan_windows(
                len(toks),
                self.real_len,
                self.stride,
                self.min_seq_len,
                self.add_final_window,
                self.start_from_end,
            ):
                start, end = window["start"], window["end"]
                win_ids, win_time_age, win_time_abs, win_type, win_mask = build_window(
                    toks[start:end],
                    times_age[start:end],
                    times_abs[start:end] if times_abs is not None else None,
                    types[start:end] if types is not None else None,
                    self.sos_id if window["first"] else self.con_id,
                    preamble,
                    self.max_seq_len,
                    self.pad_id,
                )

                preamble_len = 1 + len(preamble[0])

                # Train-time augmentation: reorder events sharing the same Age
                # (each token keeps its own features; age deltas are invariant)
                if self.shuffle_simultaneous:
                    # Seeded and rebuilt as in NTPDataset.__getitem__ (reproducible under a
                    # fixed seed, new tie orders every epoch), with the dataset seed mixed in
                    # like the shuffle buffer's. The trailing 1 gives it a stream of its own,
                    # so it never repeats the shuffle buffer's draws.
                    seed = torch.initial_seed()
                    if self._rng is None or self._rng_seed != seed:
                        self._rng = np.random.default_rng([(seed + self.seed) % (2**32), 1])
                        self._rng_seed = seed
                    win_ids, win_time_age, win_time_abs, win_type = shuffle_simultaneous_events(
                        win_ids, win_time_age, win_time_abs, win_type,
                        preamble_len, int(win_mask.sum()), self._rng)

                ids = torch.from_numpy(win_ids).to(torch.int64)
                attn = torch.from_numpy(win_mask).to(torch.uint8)
                time_age = torch.from_numpy(win_time_age)        # float64, as are all times
                time_abs = torch.from_numpy(win_time_abs)
                item = {
                    "input_ids": ids,
                    "attention_mask": attn.to(torch.int8),
                    "labels": mask_preamble_targets(build_ntp_labels(ids, attn), preamble_len),
                    "time_age": time_age,
                    "time_abs": time_abs,
                    "type": torch.from_numpy(win_type).to(torch.int64),
                }
                if self.compute_time_deltas:
                    mask_len = t2e_delta_mask_len(preamble_len)
                    item["time_deltas"] = compute_time_deltas(time_age, attn, mask_len)
                if self.return_metadata:
                    item["sample_id"] = str(pid)
                    item["window_index"] = window_index

                window_index += 1
                yield item

    def _shuffle_items(self, items: Iterator[dict]) -> Iterator[dict]:
        """items shuffled within a buffer of shuffle_buffer_windows (unchanged at 0 or 1)."""
        if self.shuffle_buffer_windows <= 1:
            yield from items
            return

        # torch.initial_seed() is new every epoch only inside DataLoader workers; in the main
        # process (num_workers=0) it is fixed, so without the epoch count every epoch would
        # replay the same order. Fresh workers start from 0, so their seeds are unchanged.
        seed = (torch.initial_seed() + self.seed + self._shuffle_epoch) % (2**32)
        self._shuffle_epoch += 1
        rng = np.random.default_rng(seed)
        buffer = []
        for item in items:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer_windows:
                idx = int(rng.integers(0, len(buffer)))
                yield buffer.pop(idx)

        rng.shuffle(buffer)
        for item in buffer:
            yield item


class StreamingAnchorsDataset(Dataset):
    """Fine-tuning windows with outcome labels, built on the fly.

    One dataset that streams from the MEDS input plus an anchors table (load_anchors):

    1. Each anchors row is one sample: the patient's events strictly before that row's
       cutoff age minus `time_before_event` (`<=` when remove_cutoff_token=False). A
       patient with several anchors gives several samples, cut from one read of their
       history; patients absent from the table are dropped; NaN cutoffs keep the full
       sequence.
    2. Window the truncated sequence (start_from_end=True by default, so the
       final window holds the most recent events before the prediction point).
    3. Keep only the final window per patient (final_windows_only=True) and
       attach the patient's outcome target as the single "labels" value.

    Items are what NTPOutcomeModel and collate_fn take. The anchors table's anchor time_age
    is computed by load_anchors from the timestamps, in days like the events'.

    outcome_duration_col is survival-only: naming it makes "labels" a two-column
    (duration, event) pair per window instead of a scalar, with outcome_col carrying the
    0/1 event indicator. Left None, the labels are exactly the [N] scalars they were.
    """

    def __init__(
        self,
        parquet_path: str,
        vocab_path: str,
        anchors: pd.DataFrame,                      # load_anchors' table
        patient_ids: set | None = None,             # Train/val/test split filter
        # Truncation at each anchor (step 1 above)
        time_before_event: float = 0,
        remove_cutoff_token: bool = True,
        # Windowing (same knobs as RawStreamingNTPDataset)
        max_seq_len: int = 512,
        overlap_percent: int = 0,
        id_col: str = "subject_id",
        tok_col: str = "code",
        time_age_col: str = AGE_COL,
        time_abs_col: str | None = ABS_COL,
        type_col: str | None = "type",
        sos_token: str = "<SOS>",
        con_token: str = "<CON>",
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        min_seq_len: int = 1,
        add_final_window: bool = True,
        start_from_end: bool = True,
        final_windows_only: bool = True,
        # Outcomes table columns
        outcome_patient_col: str = "subject_id",
        outcome_age_col: str = AGE_COL,
        outcome_col: str = "target",
        outcome_duration_col: str | None = None,
        # Progress: a label turns the index-pass bar on, None keeps the scan silent
        progress_desc: str | None = None,
        event_times=None,                    # load_event_times: the run's checked time basis
    ):
        """Index the samples with one streamed pass over the selected patients (cut at each
        anchor, front block extracted, windows planned); the events of a window are fetched
        again when it is read."""
        time_before_event, remove_cutoff_token = validate_cutoff(time_before_event, remove_cutoff_token)

        stoi, itos, stats = load_vocab(vocab_path)
        self.stoi = stoi
        self.sos_id = int(stoi[sos_token])
        self.con_id = int(stoi[con_token])
        self.pad_id = int(stoi[pad_token])
        self.unk_id = int(stoi[unk_token])
        self.preamble = preamble_spec(stats)
        self.type_to_id = type_id_map(stats) if type_col is not None else {}

        self.parquet_path = parquet_path
        self.id_col, self.tok_col, self.time_age_col = id_col, tok_col, time_age_col
        self.time_abs_col, self.type_col = time_abs_col, type_col
        self.max_seq_len = max_seq_len
        self.time_before_event = time_before_event
        self.remove_cutoff_token = remove_cutoff_token
        warn_precluster(parquet_path)

        # Anchors: one row per sample; the target column is optional so an anchors-only
        # table (patient id + anchor time_age) also works, e.g. for generation
        for col in (outcome_patient_col, outcome_age_col):
            if col not in anchors.columns:
                raise KeyError(f"Anchors table missing column '{col}'. Available: {list(anchors.columns)}")

        # The per-patient join below is a plain dict keyed on native id values, so a
        # numeric-vs-string dtype-class mismatch between the input's id and the
        # outcomes id would never match and silently empty the dataset. DuckDB's own id
        # filter already raises on this; give the dict join the same loud guard.
        raw_class = parquet_id_class(parquet_path, id_col)
        out_class = _pandas_id_class(anchors[outcome_patient_col])
        if raw_class and out_class and raw_class != out_class:
            raise ValueError(
                f"Patient-id dtype mismatch: input column '{id_col}' is {raw_class} "
                f"but anchors column '{outcome_patient_col}' is {out_class}. The "
                f"per-patient join is by exact id, so these never match and every patient "
                f"would be silently dropped. Re-export the outcomes/anchors file with a "
                f"matching id dtype.")

        # Survival only. Naming the column is what asks for it, so a missing column is a
        # config error rather than an absent optional - fail before the indexing scan.
        if outcome_duration_col is not None and outcome_duration_col not in anchors.columns:
            raise ValueError(
                f"Anchors table has no survival duration column {outcome_duration_col!r}. "
                f"Available columns: {list(anchors.columns)}.")
        # Each patient's anchor rows, oldest anchor first, as positions into the
        # per-row arrays below
        anchors = anchors.sort_values(outcome_age_col, kind="stable")
        rows_by_pid = anchors.groupby(outcome_patient_col, sort=False).indices
        cutoff_of_row = anchors[outcome_age_col].to_numpy()
        target_of_row = (anchors[outcome_col].to_numpy() if outcome_col in anchors.columns
                         else np.zeros(len(anchors)))  # dummy labels
        duration_of_row = (anchors[outcome_duration_col].to_numpy()
                           if outcome_duration_col is not None else None)

        real_len, stride = window_geometry(max_seq_len, overlap_percent, self.preamble.slots)

        # Lazy index: streamed metadata pass (truncate -> front-block extract -> plan) with the
        # events fetched per patient at __getitem__ (whole split never resident).
        kept_pids, targets, cutoffs, durations = [], [], [], []
        window_rows = []  # (sample_idx, start, end, first); a sample is one (patient, anchor)
        n_streamed = 0
        n_join_hits = 0  # streamed pids found in the anchors table
        stream = stream_patient_sequences(
            parquet_path, id_col, tok_col, time_age_col, time_abs_col, type_col,
            ids=sorted(patient_ids) if patient_ids is not None else None,
            event_times=event_times)
        # The split filter gives the bar a total; a full-file scan has none up front,
        # so tqdm degrades to a plain counter. mininterval keeps captured (non-TTY)
        # job logs readable: one line every few seconds instead of one per patient.
        stream = tqdm(stream, desc=progress_desc, unit="pat", mininterval=2.0,
                      total=len(patient_ids) if patient_ids is not None else None,
                      disable=progress_desc is None, leave=False)
        for pid, tok_vals, age, absv, type_vals in stream:
            n_streamed += 1
            if pid not in rows_by_pid:
                continue  # patients absent from the anchors table are dropped
            n_join_hits += 1
            toks = encode_tokens(tok_vals, self.stoi, self.unk_id)
            types = encode_types(type_vals, self.type_to_id)
            toks, age, absv, types = canonical_sort(toks, age, absv, types)
            for row in rows_by_pid[pid]:  # one sample per anchor, oldest first
                cutoff = cutoff_of_row[row]
                s_toks, s_age, s_absv, s_types = self._truncate(toks, age, absv, types, cutoff)
                s_toks, s_age, s_absv, s_types, _, _ = extract_preamble_tokens(
                    s_toks, s_age, s_absv, s_types, self.preamble)
                if len(s_toks) < min_seq_len:
                    continue
                plans = plan_windows(len(s_toks), real_len, stride, min_seq_len, add_final_window, start_from_end)
                if final_windows_only:
                    plans = [w for w in plans if w["is_final"]]
                if not plans:
                    continue
                pi = len(kept_pids)
                kept_pids.append(pid)
                targets.append(target_of_row[row])  # native dtype (int class or float value)
                if duration_of_row is not None:
                    durations.append(duration_of_row[row])
                cutoffs.append(np.nan if pd.isna(cutoff) else float(cutoff))
                for w in plans:
                    window_rows.append((pi, w["start"], w["end"], int(w["first"])))

        # Backstop for mismatches the class check can't see (e.g. same-class but
        # disjoint id spaces, or different id encodings): a total join miss is never
        # a legitimate empty dataset, so fail loudly instead of returning length 0.
        if n_streamed > 0 and len(rows_by_pid) > 0 and n_join_hits == 0:
            raise ValueError(
                f"Zero patient-id overlap: {n_streamed} patients streamed from "
                f"{parquet_path} and {len(rows_by_pid)} ids in the anchors table, but "
                f"none matched. The two id columns likely use different dtypes or id "
                f"spaces; the dataset would otherwise be silently empty.")

        # One entry per sample (patient, anchor): the patient id and the anchor age
        self.patient_table = np.asarray(kept_pids)
        self.cutoff_table = np.asarray(cutoffs, dtype=np.float64)
        self.windows = np.asarray(window_rows, dtype=np.int64) if window_rows else np.empty((0, 4), dtype=np.int64)
        self._reader = PatientEventReader(parquet_path, id_col, tok_col, time_age_col, time_abs_col,
                                          type_col, event_times=event_times)

        # Per-window labels: keep the native value (float preserves regression
        # targets AND integer class ids). Survival is the one two-column label: [N, 2]
        # ordered (duration, event) - the contract survival_core owns - which
        # collate_fn stacks to [B, 2] with no change, exactly as it
        # stacks the [N] case to [B].
        per_sample_target = np.asarray(targets, dtype=np.float64)
        if duration_of_row is not None:
            per_sample_target = np.column_stack(
                [np.asarray(durations, dtype=np.float64), per_sample_target])
        self.labels = per_sample_target[self.windows[:, 0]] if len(self.windows) else per_sample_target

        self.patient_ids = self.patient_table[self.windows[:, 0]] if len(self.windows) else self.patient_table
        # Each window's own anchor age (a patient may have several); with patient_ids, its key
        self.anchor_ages = self.cutoff_table[self.windows[:, 0]] if len(self.windows) else self.cutoff_table

    def __len__(self) -> int:
        """The number of windows (one per sample when final_windows_only)."""
        return len(self.windows)

    def _truncate(self, toks, age, absv, types, cutoff):
        """The events before cutoff - time_before_event: strictly (<) when
        remove_cutoff_token, else up to and including it (<=, which keeps the anchor event).
        A NaN cutoff keeps all of them."""
        if cutoff is None or pd.isna(cutoff):
            return toks, age, absv, types
        n = cutoff_index(age, float(cutoff) - float(self.time_before_event),
                         self.remove_cutoff_token)
        return (toks[:n], age[:n],
                absv[:n] if absv is not None else None,
                types[:n] if types is not None else None)

    def _patient_arrays(self, pid, cutoff):
        """One patient's arrays cut at cutoff, as the index pass built them: fetch, encode,
        sort, truncate, extract the front block (deterministic, so the window is identical).
        Returns extract_preamble_tokens' tuple."""
        tok_vals, age, absv, type_vals = self._reader.fetch(pid.item() if hasattr(pid, "item") else pid)
        toks = encode_tokens(tok_vals, self.stoi, self.unk_id)
        types = encode_types(type_vals, self.type_to_id)
        toks, age, absv, types = canonical_sort(toks, age, absv, types)
        toks, age, absv, types = self._truncate(toks, age, absv, types, cutoff)
        return extract_preamble_tokens(toks, age, absv, types, self.preamble)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """The i-th window as an item: input_ids, attention_mask, labels (the sample's
        outcome), time_age, time_abs and type."""
        pi, ws, we, first = self.windows[i]
        toks, times_age, times_abs, types, preamble, _ = self._patient_arrays(
            self.patient_table[pi], self.cutoff_table[pi])

        win_ids, win_time_age, win_time_abs, win_type, win_mask = build_window(
            toks[ws:we],
            times_age[ws:we],
            times_abs[ws:we] if times_abs is not None else None,
            types[ws:we] if types is not None else None,
            self.sos_id if first else self.con_id,
            preamble,
            self.max_seq_len,
            self.pad_id,
        )

        # The dtypes collate_fn stacks
        return {
            "input_ids": torch.from_numpy(win_ids.astype(np.int64)),
            "attention_mask": torch.from_numpy(win_mask.astype(np.int64)),
            "labels": torch.tensor(self.labels[i], dtype=torch.float32),
            "time_age": torch.from_numpy(win_time_age),            # float64, as are all times
            "time_abs": torch.from_numpy(win_time_abs),
            "type": torch.from_numpy(win_type.astype(np.int64)),
        }
