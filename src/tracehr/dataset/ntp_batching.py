# src/tracehr/dataset/ntp_batching.py
"""Next-token training items: labels, the masked front block, waiting times, the shuffle of
simultaneous events, the window-store dataset (NTPDataset) and collate_fn.

The streaming datasets (streaming_datasets.py) build their items with the same functions, so a
window yields the same item whichever path built it.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def build_ntp_labels(ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Next-token labels of one window [T]: the next id where the current AND the next
    positions are real, else -100 (ignored by the loss)."""
    T = ids.shape[0]
    labels = torch.full((T,), -100, dtype=torch.int64)
    a = attn.bool()
    valid = a[:-1] & a[1:]
    labels[:-1][valid] = ids[1:][valid]
    return labels


def shuffle_simultaneous_events(ids, time_age, time_abs, tok_type, preamble_len, real_len, rng):
    """Randomly reorder positions within equal-Age runs in [preamble_len, real_len).

    Sorted data places equal-Age events in token-id order; shuffling the runs at
    training time stops the model from learning that arbitrary tie-break. Returns
    fancy-indexed copies (inputs, incl. shared window-store rows, are never mutated).
    """
    perm = np.arange(len(ids))
    start = preamble_len
    for i in range(preamble_len + 1, real_len + 1):
        if i == real_len or time_age[i] != time_age[start]:
            if i - start > 1:
                rng.shuffle(perm[start:i])   # in-place on the slice view
            start = i
    return (ids[perm], time_age[perm],
            time_abs[perm] if time_abs is not None else None,
            tok_type[perm] if tok_type is not None else None)


def mask_preamble_targets(labels: torch.Tensor, preamble_len: int) -> torch.Tensor:
    """labels with the front block's targets set to -100: the start token and the sex and
    static tokens are re-inserted in every window, so their prediction is never trained."""
    if preamble_len > 1:
        labels[:preamble_len - 1] = -100
    return labels


def t2e_delta_mask_len(preamble_len: int) -> int:
    """Number of leading time-delta positions to mask with the -1 sentinel: the preamble.

    Waiting times are scored from the first REAL event onward, in every window. The
    preamble (SOS/CON start token, then the re-inserted sex and static tokens) carries synthetic times,
    so no waiting time leaving it is a real one:
      * CON windows: the true previous event lives in the preceding window.
      * SOS windows: the preamble sits at birth (start token at time 0, sex/static tokens at Age 0),
        so its target would be birth -> first recorded event. In EHR that span is the time
        before the patient entered the record, not a waiting time between events. Scoring
        it also hurts: at initialisation these few decades-long gaps produce a gradient
        ~9x larger than every other gap combined and ~40x the token loss, and in
        coupled T2E that update runs through the token logits (in an ablation it
        cost ~8 perplexity points after 8 epochs, with no gain on the other gaps).
    The token target of the first event is unaffected (see mask_preamble_targets).
    """
    return preamble_len


def compute_time_deltas(times: torch.Tensor, attn: torch.Tensor, mask_len: int = 0) -> torch.Tensor:
    """Waiting time to the next event at each position [T] (float64 days): the forward time
    difference where the current AND the next positions are real, else the -1 sentinel; the
    first mask_len positions are forced to -1 (see t2e_delta_mask_len). Differenced and kept
    in float64 (pass the float64 Age), so they are exact to well below a minute."""
    T = times.shape[0]
    time_deltas = torch.full((T,), -1.0, dtype=torch.float64)
    a = attn.bool()
    valid = a[:-1] & a[1:]
    if valid.any():
        deltas = torch.clamp(times[1:] - times[:-1], min=0.0).double()  # Sorted data => non-negative
        time_deltas[:-1][valid] = deltas[valid]
    if mask_len > 0:
        time_deltas[:mask_len] = -1.0
    return time_deltas


class NTPDataset(Dataset):
    """Next-token items read from a window store (create_sequence_windows).

    Each item is one stored window with its next-token labels (front block masked), the
    times and types it holds, and, with compute_time_deltas, the T2E waiting times.
    shuffle_simultaneous reorders the events that share a time_age each time a window is
    read (training only); return_metadata adds sample_id and window_index.
    """

    def __init__(
        self,
        npz_path: str,
        return_metadata: bool = False,
        compute_time_deltas: bool = False,
        shuffle_simultaneous: bool = False,  # train-time: reorder events sharing the same Age per fetch
    ):
        """Open the store (memory-mapped) and refuse one this version did not build, or one
        lacking what compute_time_deltas / shuffle_simultaneous need."""
        # Open the window store: a single memory-mapped structured .npy (fields are
        # OS-paged, shared across workers) or a legacy .npz. No full load into RAM.
        from .windows import open_window_store
        d = open_window_store(npz_path)
        # Every store built from MEDS input records its front block (preamble_slots). One
        # without it comes from an older version, whose times or front block may differ, and
        # would feed the model wrong windows with no error (train_model regenerates such stores).
        meta = d["meta"][0] if "meta" in d.files else {}
        if meta.get("preamble_slots") is None:
            raise ValueError(f"{npz_path} was built by an older version (it records no front block). "
                             f"Regenerate the windows with create_sequence_windows(...).")

        # Required fields
        self.input_ids = d["input_ids"]           # [N,T] Int
        self.attn = d["attention_mask"]           # [N,T] Int8 (1=real incl. start, 0=pad)
        self.patient_ids = d["patient_id"]        # [N] Patient IDs

        # Optional features - only load if present
        self.time_age = d["time_age"] if "time_age" in d.files else None
        self.time_abs = d["time_abs"] if "time_abs" in d.files else None
        self.tok_type = d["type"] if "type" in d.files else None

        self.preamble_len = d["preamble_len"] if "preamble_len" in d.files else None
        self.return_metadata = return_metadata
        self.compute_time_deltas = compute_time_deltas
        self.shuffle_simultaneous = shuffle_simultaneous
        self._rng = None       # Lazy: one RNG per process (main or DataLoader worker)
        self._rng_seed = None  # The torch.initial_seed() it was built from; see __getitem__

        # Equal-Age runs are found via time_age, and preamble_len keeps the front block
        # (sex and static tokens) pinned after the start token — both are required to shuffle safely
        if shuffle_simultaneous and (self.time_age is None or self.preamble_len is None):
            raise KeyError(
                "shuffle_simultaneous=True requires 'time_age' and 'preamble_len' in the window store. "
                "Regenerate the windows with create_sequence_windows(...)."
            )

        if compute_time_deltas:
            if self.time_age is None or not np.any(self.time_age):
                raise ValueError(
                    "compute_time_deltas=True but the window store has no usable 'time_age' values. "
                    "Regenerate the windows with create_sequence_windows(...)."
                )
            # Per-window preamble length is needed to mask synthetic waiting times
            if "preamble_len" not in d.files:
                raise KeyError(
                    "The window store is missing 'preamble_len' (needed for T2E delta masking). "
                    "Regenerate the windows with create_sequence_windows(...)."
                )
            print(f"[t2e] NTPDataset time_deltas from time_age ({npz_path})")

    def __len__(self) -> int:
        """The number of windows in the store."""
        return self.input_ids.shape[0]

    def __getitem__(self, i: int) -> dict:
        """The i-th window as a training item (see the class docstring)."""
        # Fetch the rows as numpy (the shuffle, if any, happens before tensors and labels)
        ids_np = self.input_ids[i]
        age_np = self.time_age[i] if self.time_age is not None else None
        abs_np = self.time_abs[i] if self.time_abs is not None else None
        typ_np = self.tok_type[i] if self.tok_type is not None else None
        attn = torch.from_numpy(self.attn[i]).to(torch.uint8)           # [T] 1=real, 0=pad

        # Train-time augmentation: reorder events sharing the same Age (each token
        # keeps its own features). time_deltas (from Age) are invariant under this.
        if self.shuffle_simultaneous:
            # Seeded from torch.initial_seed(): set_seed fixes it in the main process and the
            # DataLoader gives every worker of every epoch a new one, so a fixed seed reproduces
            # the run while the tie orders still change from epoch to epoch (in the main
            # process, num_workers=0, the generator simply carries on). Rebuilt whenever that
            # seed changes, so a generator made in the main process is never copied into, and
            # shared by, the workers.
            seed = torch.initial_seed()
            if self._rng is None or self._rng_seed != seed:
                self._rng, self._rng_seed = np.random.default_rng(seed), seed
            ids_np, age_np, abs_np, typ_np = shuffle_simultaneous_events(
                ids_np, age_np, abs_np, typ_np,
                int(self.preamble_len[i]), int(attn.sum()), self._rng)

        ids = torch.from_numpy(ids_np).to(torch.int64)                  # [T]

        # Next-token labels, built on the fly
        labels = build_ntp_labels(ids, attn)
        if self.preamble_len is not None:
            labels = mask_preamble_targets(labels, int(self.preamble_len[i]))

        item = {
            "input_ids": ids,                     # [T] Int64
            "attention_mask": attn.to(torch.int8),# [T] Int8 (HF expects 1/0)
            "labels": labels,                     # [T] Int64 (-100 where ignored)
        }

        # Add optional Age, absTime, Type features only if they exist. Times stay float64
        # (copied, so an item never shares memory with the store).
        if age_np is not None:
            item["time_age"] = torch.from_numpy(np.array(age_np, dtype=np.float64))

        if abs_np is not None:
            item["time_abs"] = torch.from_numpy(np.array(abs_np, dtype=np.float64))

        if typ_np is not None:
            item["type"] = torch.from_numpy(typ_np).to(torch.int64)

        # Time deltas for T2E prediction, with synthetic preamble waiting times masked
        # out, from the stored float64 Age.
        if self.compute_time_deltas:
            mask_len = t2e_delta_mask_len(int(self.preamble_len[i]))
            item["time_deltas"] = compute_time_deltas(torch.from_numpy(age_np), attn, mask_len)

        if self.return_metadata:
            item["sample_id"] = str(self.patient_ids[i])
            item["window_index"] = i

        return item


def collate_fn(batch):
    """Stack window items into a batch: every tensor field the first item carries, plus the
    sample_id / window_index lists when the dataset returns metadata.

    Serves the next-token windows (NTPDataset, RawStreamingNTPDataset) and the outcome
    windows (StreamingAnchorsDataset, whose labels stack to [B], or to [B, 2] for survival).
    time_deltas must come from the dataset (compute_time_deltas=True), which knows each
    window's front block; there is no fallback here.
    """
    collated = {}
    for key in ("input_ids", "attention_mask", "labels", "time_age", "time_abs", "type", "time_deltas"):
        if key in batch[0]:
            collated[key] = torch.stack([b[key] for b in batch])

    if "sample_id" in batch[0]:
        collated["sample_id"] = [b["sample_id"] for b in batch]
        collated["window_index"] = [b["window_index"] for b in batch]

    return collated
