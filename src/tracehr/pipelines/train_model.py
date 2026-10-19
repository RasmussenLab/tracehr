#!/usr/bin/env python3
# src/tracehr/pipelines/train_model.py
"""Pipeline 01: pretrain the NTPTransformer on next-token prediction (with coupled
time-to-event when model.t2e_weight > 0) from the MEDS input. Builds or reuses the
vocabulary, streams the training windows (data.mode streaming) or builds/reuses window
stores (save_intermediate), validates every epoch, and writes latest.pt, best_model.pt,
the periodic epoch_N.pt, training_history.json and the plots to data.output_dir. Resumes
from data.model_checkpoint_path, else from <output_dir>/latest.pt when it exists."""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from tracehr.dataset.cohort import load_patient_id_filter
from tracehr.dataset.column_mapping import enabled_col, resolve_columns
from tracehr.dataset.meds_reader import EARLIEST_TIMESTAMP, count_rows, load_event_times
from tracehr.dataset.ntp_batching import NTPDataset, collate_fn
from tracehr.dataset.streaming_datasets import RawStreamingNTPDataset
from tracehr.dataset.vocabulary import (
    build_vocab,
    load_vocab,
    resolve_t2e_exclude_ids,
    unk_token_share,
    vocabulary_fingerprint,
)
from tracehr.dataset.windows import (
    MAX_PREAMBLE_LEN,
    create_sequence_windows,
    infer_num_types_from_npz,
    preamble_spec,
    window_capacity,
    window_npz_geometry_mismatch,
)
from tracehr.ntp_model.ntp_transformer import ROPE_TIME_MAX_DAYS, NTPTransformer
from tracehr.training.one_epoch import train_one_epoch
from tracehr.training.plots import (
    plot_accuracy,
    plot_all_batch_losses,
    plot_losses,
    plot_perplexity,
)
from tracehr.training.resume import (
    check_training_resume,
    describe_lr_schedule,
    resolve_resume_checkpoint,
    resume_lr_schedule,
)
from tracehr.training.runtime import (
    count_params,
    prune_epoch_checkpoints,
    resolve_device,
    set_seed,
)
from tracehr.training.validation import validate

# training_history.json's per-epoch series, in file order. The two maps name the metric of
# train_one_epoch / validate each series records (0.0 when the metric is absent).
_TRAIN_SERIES = {
    "train_token_losses": "token_loss",
    "train_t2e_losses": "t2e_loss",
    "train_weighted_t2e_losses": "weighted_t2e_loss",
    "train_t2e_ratios": "t2e_to_token_ratio",
    "train_t2e_fractions": "t2e_loss_fraction",
}
_VAL_SERIES = {
    "val_token_losses": "token_loss",
    "val_t2e_losses": "t2e_loss",
    "val_weighted_t2e_losses": "weighted_t2e_loss",
    "val_t2e_ratios": "t2e_to_token_ratio",
    "val_t2e_fractions": "t2e_loss_fraction",
    "val_t2e_maes": "t2e_mae",
}
# train_loss_diags / val_loss_diags: the objective's diagnostics per epoch ({} for plain CE)
_HISTORY_SERIES = ("train_losses", "val_losses", *_TRAIN_SERIES, *_VAL_SERIES,
                   "val_perplexities", "val_accuracies", "val_selections",
                   "train_loss_diags", "val_loss_diags", "all_batch_losses")


@dataclass
class _RunState:
    """Where this run starts, and what it carries over when it resumes a checkpoint."""
    history_path: Path
    start_epoch: int = 0
    best_val_select: float = float('inf')
    best_epoch: int = 0
    schedule_start: int = 0       # epoch the current LR schedule began at: 0, or where a new phase began
    new_phase: str | None = None  # why the LR warms up again on this resume, if it does
    data_changes: list = field(default_factory=list)  # epochs after which the training/validation data changed
    ckpt: dict | None = None      # the checkpoint resumed, None for a fresh run
    steps_changed: bool = False   # the epoch length differs from the checkpoint's


# ============================================================================ #
# Settings
# ============================================================================ #
def _refuse_removed_settings(model_cfg: dict) -> None:
    """Fail before any vocabulary or dataset work if an old config still carries switches
    that no longer exist. Silently ignoring either would run a different architecture from
    the one the YAML appears to request."""
    removed = sorted({"tie_weights", "t2e_mode"}.intersection(model_cfg))
    if removed:
        raise ValueError(
            f"Removed model setting(s): {removed}. Weights are always tied; remove "
            "tie_weights and use only t2e_weight (0=off, >0=coupled T2E)."
        )
    removed = sorted({"t2e_time_resolution_days", "rope_time_scale_days",
                      "time_daily_cycle"}.intersection(model_cfg))
    if removed:
        raise ValueError(
            f"Removed model setting(s): {removed}. Every model resolves time to the minute: "
            "the T2E grid, the fastest Age-RoPE rotation and Time2Vec's daily cycle are fixed. "
            "Remove them."
        )


def _resolve_t2e_weight(model_cfg: dict) -> float:
    """model.t2e_weight as a float, written back into model_cfg. One T2E source of truth: zero
    disables the architecture and data path; a positive finite value enables coupled
    competing risks and is the exact combined-loss weight."""
    raw_t2e_weight = model_cfg.get("t2e_weight", 0.0)
    if isinstance(raw_t2e_weight, bool) or not isinstance(raw_t2e_weight, (int, float)):
        raise TypeError(f"model.t2e_weight must be a finite number >= 0, got {raw_t2e_weight!r}.")
    t2e_weight = float(raw_t2e_weight)
    if not math.isfinite(t2e_weight) or t2e_weight < 0.0:
        raise ValueError(f"model.t2e_weight must be finite and >= 0, got {raw_t2e_weight!r}.")
    model_cfg["t2e_weight"] = t2e_weight
    return t2e_weight


# ============================================================================ #
# Vocabulary and data
# ============================================================================ #
def _build_training_vocab(data_cfg: dict, seq_cfg: dict, cols: dict, type_col: str | None):
    """Build the vocabulary from the training split and write it to data.vocab_path."""
    return build_vocab(
        parquet_path=data_cfg["parquet_path"],
        ids_path=data_cfg["train_ids"],
        ids_col=cols["id_col"],
        tok_col=cols["tok_col"],
        type_col=type_col,
        special_tokens={"<PAD>": 0, "<SOS>": 1, "<CON>": 2, "<UNK>": 3, "<MASK>": 4},
        unk_token=seq_cfg.get("unk_token", "<UNK>"),   # data's rare-token marker (default <UNK>)
        sex_tokens=seq_cfg.get("sex_tokens", []),      # exact token strings treated as sex ([] = none)
        output_path=str(Path(data_cfg["vocab_path"])),
        static_tokens=seq_cfg.get("static_tokens", []),  # further front-block tokens ([] = none)
        birth_token=seq_cfg.get("birth_token"),       # the code of the rows that give each birth
        max_preamble_len=MAX_PREAMBLE_LEN,
        earliest_timestamp=EARLIEST_TIMESTAMP,
        timestamp_col=cols["timestamp_col"],
    )


def _load_or_build_vocab(data_cfg: dict, seq_cfg: dict, cols: dict, type_col: str | None):
    """(stoi, itos, stats) of data.vocab_path: reused when the file exists and is complete,
    else built from the training split."""
    vocab_path = Path(data_cfg["vocab_path"])
    if not vocab_path.is_file():
        return _build_training_vocab(data_cfg, seq_cfg, cols, type_col)
    # Rebuilt when it lacks the type mapping (types in use) or the front block
    stoi, itos, stats = load_vocab(str(vocab_path))
    if type_col is not None and not stats.get("type_to_id"):
        print("Vocabulary is missing stable type_to_id mapping; rebuilding vocabulary.")
        return _build_training_vocab(data_cfg, seq_cfg, cols, type_col)
    if "preamble_slots" not in stats:
        print("Vocabulary predates MEDS input (it records no front block); rebuilding vocabulary.")
        return _build_training_vocab(data_cfg, seq_cfg, cols, type_col)
    return stoi, itos, stats


def _set_t2e_exclude_ids(model_cfg: dict, stoi: dict, stats: dict, t2e_weight: float) -> None:
    """Write model.t2e_exclude_ids. The competing-risk sum contains only tokens that can occur
    as real future events. Sex and static tokens are reinserted as front-block attributes and
    are never prediction targets, so allowing them into the rate would create an
    unsupervised auxiliary hazard."""
    model_cfg["t2e_exclude_ids"] = resolve_t2e_exclude_ids(
        stoi, extra_ids=[*stats.get("sex_token_ids", ()), *stats.get("static_token_ids", ())])
    print(f"[t2e] weight={t2e_weight:g}: coupled time_deltas from time_age, "
          f"excluded token ids: {model_cfg['t2e_exclude_ids']}")


def _checked_preamble(model_cfg: dict, seq_cfg: dict, stats: dict):
    """The vocabulary's front block, after checking that a window still has room for
    sequences.min_seq_len events: every window holds the start token, the front block's S
    slots and the events."""
    preamble = preamble_spec(stats)
    capacity = window_capacity(model_cfg["max_seq_len"], preamble.slots)
    if capacity < seq_cfg.get("min_seq_len", 1):
        raise ValueError(
            f"model.max_seq_len {model_cfg['max_seq_len']} leaves {capacity} event position(s) "
            f"after the start token and the front block's {preamble.slots} slot(s), fewer than "
            f"sequences.min_seq_len {seq_cfg.get('min_seq_len', 1)}. Raise model.max_seq_len.")
    return preamble


def _load_event_times(data_cfg: dict, cols: dict, t2e_pred: bool):
    """The run's checked event times (load_event_times), with the time resolution and each
    split's patient and row counts reported.

    Every model resolves time to the minute (TIME_RESOLUTION_DAYS, ntp_transformer.py): the
    T2E grid, the fastest Age-RoPE rotation and Time2Vec's daily cycle are fixed. Every
    patient's rows are checked once here; each reader leaves out the rows and patients that
    fail and computes time_age and time_abs from the timestamps.
    """
    print("[time] times are resolved to the minute: T2E grid, fastest Age-RoPE rotation "
          "(1 rad per minute), Time2Vec daily cycle")
    event_times = load_event_times(data_cfg["parquet_path"], cols, data_cfg["vocab_path"])
    for split, ids_key in (("train", "train_ids"), ("val", "val_ids")):
        n_patients, n_rows = count_rows(data_cfg["parquet_path"], event_times, cols["id_col"],
                                        cols["tok_col"], load_patient_id_filter(data_cfg[ids_key]))
        print(f"  {split} split ({data_cfg[ids_key]}): {n_patients:,} patients, {n_rows:,} rows")
    if t2e_pred:
        print("[t2e] waiting times scored on the 1-minute grid as P(gap in its cell), events "
              "in the same minute as k=0; finer recordings (seconds) are coarsened into the cells")
    return event_times


def _window_store_datasets(config: dict, cols: dict, time_abs_col, type_col, preamble,
                           event_times, t2e_pred: bool, num_types: int):
    """(train, val, num_types) from the window stores data.train_npz / data.val_npz, each
    reused when it matches the training config and (re)built otherwise."""
    data_cfg, model_cfg, seq_cfg = config["data"], config["model"], config.get("sequences", {})
    # The window stores are built straight from the input (tokens encoded in SQL).
    window_kwargs = dict(
        parquet_path=data_cfg["parquet_path"],
        vocab_path=data_cfg["vocab_path"],
        max_seq_len=model_cfg["max_seq_len"],
        overlap_percent=model_cfg["window_overlap"],
        id_col=cols["id_col"],
        tok_col=cols["tok_col"],
        time_age_col=cols["age_col"],
        time_abs_col=time_abs_col,
        type_col=type_col,
        sos_token="<SOS>",
        con_token="<CON>",
        pad_token="<PAD>",
        min_seq_len=seq_cfg.get("min_seq_len", 1),
        add_final_window=True,
        event_times=event_times,
    )
    # Validation windows partition each history - no overlap and no extra tail window -
    # so every held-out token is scored exactly once, the same windows as the streaming
    # val split. Training keeps its configured overlap. A val store built with the
    # training geometry fails the geometry check below and is regenerated.
    split_geometry = {
        "train_npz": {},
        "val_npz": {"overlap_percent": 0, "add_final_window": False},
    }
    for split, ids_key in (("train_npz", "train_ids"), ("val_npz", "val_ids")):
        npz_path = Path(data_cfg[split])
        geometry = {**window_kwargs, **split_geometry[split]}
        if npz_path.is_file():
            # Pre-existing windows are only reused when their geometry and vocabulary match
            # the training config; stale files (e.g. a different max_seq_len) are regenerated
            mismatch = window_npz_geometry_mismatch(
                str(npz_path),
                max_seq_len=geometry["max_seq_len"],
                overlap_percent=geometry["overlap_percent"],
                min_seq_len=geometry["min_seq_len"],
                add_final_window=geometry["add_final_window"],
                vocab_path=data_cfg["vocab_path"],
                preamble_slots=preamble.slots,
                uses_types=type_col is not None,
            )
            if mismatch is None:
                print(f"[windows] {npz_path} matches the training config, reusing it.")
                continue
            print(f"[windows] {npz_path} is stale ({mismatch}); regenerating.")
        create_sequence_windows(output_path=str(npz_path), ids_path=data_cfg[ids_key], **geometry)
    train_dataset = NTPDataset(
        data_cfg["train_npz"], compute_time_deltas=t2e_pred,
        shuffle_simultaneous=data_cfg.get("shuffle_simultaneous", False))
    val_dataset = NTPDataset(data_cfg["val_npz"], compute_time_deltas=t2e_pred)
    if type_col is not None and num_types == 0:
        num_types = infer_num_types_from_npz(data_cfg["train_npz"])
    return train_dataset, val_dataset, num_types


def _streaming_datasets(config: dict, cols: dict, time_abs_col, type_col, event_times,
                        t2e_pred: bool, num_types: int):
    """(train, val, num_types) streamed from the MEDS input on the fly."""
    data_cfg, model_cfg, seq_cfg = config["data"], config["model"], config.get("sequences", {})
    stream_kwargs = dict(
        parquet_path=data_cfg["parquet_path"],
        vocab_path=data_cfg["vocab_path"],
        max_seq_len=model_cfg["max_seq_len"],
        overlap_percent=model_cfg["window_overlap"],
        id_col=cols["id_col"],
        tok_col=cols["tok_col"],
        time_age_col=cols["age_col"],
        time_abs_col=time_abs_col,
        type_col=type_col,
        sos_token="<SOS>",
        con_token="<CON>",
        pad_token="<PAD>",
        min_seq_len=seq_cfg.get("min_seq_len", 1),
        compute_time_deltas=t2e_pred,
        seed=config["training"]["seed"],
        event_times=event_times,
    )
    print("Streaming the windows from the MEDS input on the fly (no window store).")
    train_dataset = RawStreamingNTPDataset(
        ids_path=data_cfg["train_ids"],
        shuffle_buffer_windows=data_cfg.get("shuffle_buffer_windows", 8192),
        shuffle_simultaneous=data_cfg.get("shuffle_simultaneous", False),
        **stream_kwargs,
    )
    # Validation metrics must not double-count tokens: use non-overlapping partition
    # windows so each held-out token is scored exactly once (overlap_percent=0 alone
    # is NOT enough — the add_final_window tail window also re-covers the tail). The
    # train dataset above keeps its configured overlap (useful coverage/augmentation);
    # only the metric split needs a clean partition.
    val_dataset = RawStreamingNTPDataset(
        ids_path=data_cfg["val_ids"],
        shuffle_buffer_windows=0,
        **{**stream_kwargs, "overlap_percent": 0, "add_final_window": False},
    )
    if type_col is not None:
        num_types = train_dataset.num_types
    return train_dataset, val_dataset, num_types


def _make_loader(dataset, config: dict, shuffle: bool) -> DataLoader:
    """A DataLoader over dataset with the run's batch size and hardware settings."""
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        collate_fn=collate_fn,
    )


# ============================================================================ #
# Model, optimizer and learning-rate schedule
# ============================================================================ #
def _build_model(model_cfg: dict, vocab_size: int, num_types: int, stoi: dict, t2e_weight: float,
                 loss_mode: str, train_loader: DataLoader, device) -> NTPTransformer:
    """The NTPTransformer of model_cfg on device, with the T2E rate offset calibrated on one
    probe batch (and that batch's ages checked against the Age-RoPE span); its summaries
    are printed."""
    # Resolved ONCE here and reused for both the span check below and the constructor, so
    # the check and the model it is checking can never disagree about the ladder.
    rope_mode = model_cfg.get("rope_mode", "index")
    rope_time_max_days = float(model_cfg.get("rope_time_max_days", ROPE_TIME_MAX_DAYS))
    rope_hybrid_split = float(model_cfg.get("rope_hybrid_split", 0.5))
    rope_needs_age = rope_mode in ("age", "hybrid")
    # One probe batch, for the Age-RoPE span check below and the T2E rate offset. Every
    # DataLoader iterator draws a base seed from the global torch RNG, so the state is
    # restored - otherwise the model's init weights would depend on whether a probe ran at
    # a fixed seed.
    rng_state = torch.get_rng_state()
    probe = next(iter(train_loader))
    torch.set_rng_state(rng_state)
    attn_mask = probe["attention_mask"].bool()
    # The Age-RoPE ladder spans rope_time_max_days; data wider than half of that wraps
    # its slowest pair. Uses the SAME local the model is built from, not the module constant.
    if rope_needs_age and "time_age" in probe:
        vmax = float(probe["time_age"][attn_mask].max())
        if vmax > rope_time_max_days / 2:
            print("\n" + "!" * 78)
            print(f"!! WARNING: max time_age in the first batch is {vmax:.0f} days, beyond the "
                  f"{rope_time_max_days / 2:.0f}-day span")
            print("!! the Age-RoPE ladder is pinned for. The slowest rotation wraps, so long "
                  "gaps stop being decodable.")
            print("!" * 78 + "\n")

    model = NTPTransformer(
        vocab_size=vocab_size,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        max_seq_len=model_cfg["max_seq_len"],
        use_time_age=model_cfg["use_time_age"],
        use_time_abs=model_cfg["use_time_abs"],
        time_fourier_dims=model_cfg["time_fourier_dims"],
        use_type=model_cfg["use_type"],
        num_types=num_types,  # from the vocabulary or the training data
        dropout=model_cfg["dropout"],
        rope_base=model_cfg["rope_base"],
        # All three resolved above with .get defaults, so a yaml without them gets index
        # positions. Passing every one of them matters: a config field the constructor never
        # receives is silently ignored while the checkpoint records the default as if it applied.
        rope_mode=rope_mode,
        rope_hybrid_split=rope_hybrid_split,
        rope_time_max_days=rope_time_max_days,
        pad_id=int(stoi["<PAD>"]),  # single source of truth: the vocab's pad id (always 0)
        t2e_weight=t2e_weight,
        loss_mode=loss_mode,
        t2e_exclude_ids=model_cfg.get("t2e_exclude_ids"),
    ).to(device)
    # T2E rate level: fit the coupled scalar offset to the probe batch's exact grid likelihood
    # before any training step, so neither the token logits nor the trunk has to move to set it.
    # The probe is deterministic here: the global RNG was restored after it.
    if model.t2e_pred:
        print(model.calibrate_t2e_rate_offset(
            {k: v.to(device) for k, v in probe.items() if isinstance(v, torch.Tensor)}))

    print(f"Model parameters: {count_params(model):,}")
    print(model.rope_summary())
    print(model.time2vec_summary())
    print(model.loss_summary())
    return model


def _build_optimizer(model: NTPTransformer, config: dict, device) -> torch.optim.AdamW:
    """AdamW with no weight decay on biases, norms, and 1-d params (standard practice), plus
    whatever the objective declares off-limits. An objective's parameters can be geometry
    rather than a weight matrix (a centroid table, a temperature), where AdamW decay pulls the
    whole table toward the origin and quietly decalibrates the objective as the run gets
    longer. Plain cross-entropy declares nothing."""
    loss_no_decay = {id(p) for p in model.token_loss.no_decay_params()}
    decay_params = [p for p in model.parameters()
                    if p.requires_grad and p.ndim >= 2 and id(p) not in loss_no_decay]
    no_decay_params = [p for p in model.parameters()
                       if p.requires_grad and (p.ndim < 2 or id(p) in loss_no_decay)]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config["training"]["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config["training"]["learning_rate"],
        # Fused kernel on CUDA (one launch for all parameters, as in Delphi); elsewhere
        # PyTorch's default
        **({"fused": True} if device.type == "cuda" else {}),
    )


def _steps_per_epoch(train_loader: DataLoader, train_dataset, config: dict, accum_steps: int) -> int:
    """Optimizer steps in one epoch: batches per epoch over accum_steps, rounded up.

    len(train_loader) is exact for map-style data. For a worker-sharded IterableDataset it is
    a global estimate: each worker batches its own shard and flushes its own partial tail, so
    an epoch runs up to (num_workers - 1) extra batches. Round UP by that bound — over-running
    T_max makes CosineAnnealingLR anneal back upward, while over-estimating only leaves the
    last few steps a hair short of the final LR.
    """
    extra = (config["hardware"]["num_workers"] - 1) if isinstance(train_dataset, RawStreamingNTPDataset) else 0
    return -(-(len(train_loader) + max(0, extra)) // accum_steps)  # ceil division


def _check_warmup_fits(config: dict, steps_per_epoch: int, resume_path) -> None:
    """A fresh run whose warmup does not fit would never decay, so it is refused. A resume
    keeps its warmup_steps, so a short new phase shortens the warmup instead."""
    lr_kind = config["training"]["lr_scheduler"]
    warmup_cfg = int(config["training"].get("warmup_steps", 0))
    total_cfg = steps_per_epoch * config["training"]["num_epochs"]
    if resume_path is None and lr_kind in ("linear", "cosine") and warmup_cfg >= total_cfg:
        raise ValueError(f"training.warmup_steps ({warmup_cfg}) must be smaller than the total "
                         f"optimizer steps ({total_cfg}); lower warmup_steps or train for more epochs.")


def _build_scheduler(optimizer, total_steps: int, training_cfg: dict):
    """(scheduler, warmup steps) over total_steps optimizer steps; (None, 0) for a constant
    LR. Built once the checkpoint is read: a fresh run's schedule covers every epoch, a
    resumed one the epochs from where its schedule started (see resume_lr_schedule)."""
    lr_kind = training_cfg["lr_scheduler"]
    warmup_cfg = int(training_cfg.get("warmup_steps", 0))
    warmup_steps = min(warmup_cfg, max(total_steps - 1, 0))
    if lr_kind in ("linear", "cosine") and warmup_steps < warmup_cfg:
        print(f"[lr] WARNING: warmup_steps shortened to {warmup_steps} to fit the "
              f"{total_steps} optimizer steps of this phase")
    if lr_kind == "linear":
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps - warmup_steps)
    elif lr_kind == "cosine":
        # Same 0.1x floor as "linear" (and Delphi's min_lr = lr/10), so the two options
        # differ only in the shape of the decay
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps,
            eta_min=0.1 * training_cfg["learning_rate"])
    else:
        return None, 0
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]), warmup_steps
    return decay, warmup_steps


def _build_run_scheduler(optimizer, state: _RunState, steps_per_epoch: int, num_epochs: int,
                         training_cfg: dict):
    """The scheduler of this run, reported: it covers the run's epochs from where its schedule
    started - every epoch for a fresh run, the checkpoint's schedule when continuing it, the
    remaining epochs for a new phase."""
    scheduler, warmup_steps = _build_scheduler(
        optimizer, steps_per_epoch * (num_epochs - state.schedule_start), training_cfg)
    lr_plan = describe_lr_schedule(training_cfg["lr_scheduler"], training_cfg["learning_rate"],
                                   warmup_steps, steps_per_epoch * (num_epochs - state.schedule_start),
                                   state.schedule_start, num_epochs)
    if state.new_phase:
        print(f"[lr] {state.new_phase}: new phase from epoch {state.start_epoch}, warming up "
              f"again (Adam state kept from the checkpoint). {lr_plan}")
    else:
        unused = (" (warmup_steps unused)" if scheduler is None
                  and int(training_cfg.get("warmup_steps", 0)) > 0 else "")
        print(f"[lr] {lr_plan}{unused}")
    return scheduler


# ============================================================================ #
# Best-model selection
# ============================================================================ #
def _selection_metric(model: NTPTransformer, t2e_pred: bool) -> str:
    """The metric best_model.pt is selected on, reported: val_loss for a cross-entropy
    objective. Any other objective (HALO) is not a CE and can fall while prediction gets
    worse, so select on perplexity (the CE of its predictive distribution) instead - or, with
    t2e on, on log(perplexity) + weighted t2e loss: the same token-CE + t2e sum a CE
    objective's val_loss is."""
    if model.token_loss.is_cross_entropy:
        selection_metric = "val_loss"
    elif t2e_pred:
        selection_metric = "val_ce+t2e"
    else:
        selection_metric = "perplexity"
    print(f"[checkpoint] best_model.pt = epoch with the lowest {selection_metric}"
          + ("" if selection_metric == "val_loss" else
             f" ({model.config.loss_mode} is not a cross-entropy; its loss is not used for selection)"))
    return selection_metric


def _selection_value(selection_metric: str, val_metrics: dict) -> float:
    """The value best_model.pt selection minimizes, from one epoch's validation metrics."""
    if selection_metric == "val_loss":
        return val_metrics['loss']
    if selection_metric == "perplexity":
        return val_metrics['perplexity']
    return math.log(val_metrics['perplexity']) + val_metrics['weighted_t2e_loss']


# ============================================================================ #
# Resuming from a checkpoint
# ============================================================================ #
def _start_or_resume(resume_path, *, config: dict, cols: dict, stoi: dict, event_times,
                     model: NTPTransformer, selection_metric: str, vocab_fp: str,
                     steps_per_epoch: int, save_dir: Path, device) -> _RunState | None:
    """The state this run starts from: a fresh one, or the checkpoint at resume_path after
    check_training_resume (training/resume.py decides what a resume may change, and how the
    run adapts to a changed dataset). None when the checkpoint already completed the run.

    A changed dataset is adapted to, never refused: a changed epoch length (new training
    data) starts a new LR phase (resume_lr_schedule); the best model is re-scored on the
    current validation data in _restore_from_checkpoint, once the model is built.
    """
    if resume_path is None:
        print("\n" + "=" * 80 + "\nNO CHECKPOINT TO RESUME FROM - TRAINING FROM SCRATCH\n" + "=" * 80)
        return _RunState(history_path=save_dir / "training_history.json")

    data_cfg = config["data"]
    num_epochs = config["training"]["num_epochs"]
    print(f"Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    check_training_resume(ckpt, resume_path, model_config=model.config,
                          selection_metric=selection_metric, config=config, vocab_fp=vocab_fp,
                          vocab_path=data_cfg["vocab_path"])
    state = _RunState(history_path=resume_path.parent / "training_history.json",
                      start_epoch=ckpt['epoch'], best_val_select=ckpt['best_val_select'],
                      best_epoch=ckpt['best_epoch'], data_changes=list(ckpt['data_changes']),
                      ckpt=ckpt)
    print(f"Resuming from epoch {state.start_epoch}, best {selection_metric}: "
          f"{state.best_val_select:.4f} (epoch {state.best_epoch})")
    print(f"Checkpoint config: {ckpt.get('config', {})}")
    if state.history_path.is_file():
        print(f"Checkpoint training history path found: {state.history_path}")
    # Exit if training already complete
    if state.start_epoch >= num_epochs:
        print(f"Training already complete ({state.start_epoch}/{num_epochs} epochs; num_epochs is "
              f"the total - raise it to train further). Exiting.")
        return None

    state.steps_changed = steps_per_epoch != ckpt['steps_per_epoch']
    if state.steps_changed:
        print(f"[data] an epoch is now {steps_per_epoch} optimizer steps, not "
              f"{ckpt['steps_per_epoch']} (the training data or hardware.num_workers changed)")
        n_unk, n_tokens = unk_token_share(data_cfg["parquet_path"], data_cfg["train_ids"],
                                          cols["id_col"], cols["tok_col"], stoi, event_times)
        print(f"[vocab] {n_unk:,} of the {n_tokens:,} training tokens "
              f"({n_unk / max(n_tokens, 1):.2%}) are not in the vocabulary and train as <UNK>")
    state.schedule_start, state.new_phase = resume_lr_schedule(
        lr_kind=config["training"]["lr_scheduler"], start_epoch=state.start_epoch,
        num_epochs=num_epochs, planned_epochs=ckpt["config"]["training"]["num_epochs"],
        saved_start=ckpt['lr_schedule_start_epoch'],
        saved_steps=ckpt['steps_per_epoch'], steps_per_epoch=steps_per_epoch)
    return state


def _restore_from_checkpoint(state: _RunState, resume_path, *, model: NTPTransformer, optimizer,
                             scheduler, val_loader: DataLoader, selection_metric: str,
                             save_dir: Path, device) -> None:
    """Continue the resumed run: re-score its best model, record a data change, and load the
    model, optimizer and scheduler states.

    The re-score runs this run's best_model.pt on the current validation data: every epoch
    from here is compared with it on the same data, so best_model.pt stays the best of the
    whole run even when the validation data changed. Only when the checkpoint resumed lies
    in output_dir (its best_model.pt is this run's); a resume into another folder starts
    selection afresh there. Iterating a DataLoader draws from the global torch RNG, so its
    state is restored and an unchanged resume continues exactly.
    """
    ckpt = state.ckpt
    best_path = save_dir / "best_model.pt"
    val_changed = False
    if resume_path.parent.resolve() == save_dir.resolve() and best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(best['model_state_dict'])
        rng_state = torch.get_rng_state()
        rescored = _selection_value(selection_metric, validate(model, val_loader, device))
        torch.set_rng_state(rng_state)
        if not math.isclose(rescored, best['val_select'], rel_tol=1e-4):
            val_changed = True
            print(f"[data] best_model.pt (epoch {best['epoch']}) scores {rescored:.4f} on the "
                  f"current validation data, not the {best['val_select']:.4f} it was saved "
                  f"with: the validation data changed, so new epochs are compared with it "
                  f"on the current data")
        state.best_val_select = rescored if math.isfinite(rescored) else float('inf')
        state.best_epoch = best['epoch']
    else:
        print(f"[resume] {save_dir} holds no best_model.pt of the run resumed: best-model "
              f"selection starts afresh")
        state.best_val_select, state.best_epoch = float('inf'), 0
    if state.steps_changed or val_changed:
        state.data_changes.append({"after_epoch": state.start_epoch, "train": state.steps_changed,
                                   "val": val_changed})
    model.load_state_dict(ckpt['model_state_dict'])
    # Loading the optimizer restores its param groups as saved, including lr/initial_lr.
    # A new phase keeps Adam's moments and step counts but starts the new schedule, so the
    # lr/initial_lr the new scheduler just set are put back. Only those two: the groups'
    # fused/capturable flags must stay as saved, because loading placed Adam's step
    # counters by them. weight_decay already matches (a changed yaml value is refused).
    fresh_lrs = [(g["lr"], g["initial_lr"]) for g in optimizer.param_groups] if state.new_phase else None
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if state.new_phase:
        for group, (lr_now, initial_lr) in zip(optimizer.param_groups, fresh_lrs):
            group["lr"], group["initial_lr"] = lr_now, initial_lr
    elif scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])


# ============================================================================ #
# Training history
# ============================================================================ #
def _load_history(history_path: Path, start_epoch: int) -> dict:
    """The per-epoch series of training_history.json (_HISTORY_SERIES), empty for a fresh run;
    when resuming, the saved series cut to the epochs already trained."""
    history = {key: [] for key in _HISTORY_SERIES}
    if start_epoch > 0 and history_path.is_file():
        with open(history_path, 'r') as f:
            prev_history = json.load(f)
        for key in _HISTORY_SERIES[:-1]:
            # Padded with 0.0 ({} for the diagnostics) when the saved series is shorter
            default = {} if key.endswith("_loss_diags") else 0.0
            history[key] = (prev_history.get(key, []) + [default] * start_epoch)[:start_epoch]
        history["all_batch_losses"] = prev_history.get('all_batch_losses', [])[:start_epoch]
        print(f"Loaded training history: {len(history['train_losses'])} previous epochs")
    return history


def _record_epoch(history: dict, train_loss: float, batch_losses: list, train_metrics: dict,
                  val_metrics: dict, val_select: float) -> None:
    """Append one epoch's training and validation results to the history series."""
    history["train_losses"].append(train_loss)
    for key, metric in _TRAIN_SERIES.items():
        history[key].append(train_metrics.get(metric, 0.0))
    history["all_batch_losses"].append(batch_losses)
    history["train_loss_diags"].append(train_metrics.get('loss_diag', {}))
    history["val_losses"].append(val_metrics['loss'])
    for key, metric in _VAL_SERIES.items():
        history[key].append(val_metrics.get(metric, 0.0))
    history["val_perplexities"].append(val_metrics['perplexity'])
    history["val_accuracies"].append(val_metrics['accuracy'])
    history["val_selections"].append(val_select)
    history["val_loss_diags"].append(val_metrics.get('loss_diag', {}))


def _print_epoch(train_loss: float, train_metrics: dict, val_metrics: dict, val_select: float,
                 selection_metric: str, model_cfg: dict) -> None:
    """The epoch's one-line summary, then the objective's diagnostics when it reports any."""
    print(f"Train loss: {train_loss:.4f} "
          f"(token: {train_metrics.get('token_loss', 0):.4f}, "
          f"t2e: {train_metrics.get('t2e_loss', 0):.4f} * {model_cfg.get('t2e_weight', 0):.6f} "
          f"= {train_metrics.get('weighted_t2e_loss', 0):.4f}, "
          f"ratio: {train_metrics.get('t2e_to_token_ratio', 0):.4f}), "
          f"Val loss: {val_metrics['loss']:.4f} "
          f"(token: {val_metrics.get('token_loss', 0):.4f}, "
          f"t2e: {val_metrics.get('t2e_loss', 0):.4f} * {model_cfg.get('t2e_weight', 0):.6f} "
          f"= {val_metrics.get('weighted_t2e_loss', 0):.4f}, "
          f"ratio: {val_metrics.get('t2e_to_token_ratio', 0):.4f}), "
          f"Val PPL: {val_metrics['perplexity']:.2f}, "
          f"Val Acc: {val_metrics['accuracy']:.4f}"
          + (f", Selection ({selection_metric}): {val_select:.4f}"
             if selection_metric == "val_ce+t2e" else ""))
    # Whatever the objective reported this epoch, named by the objective itself.
    # Absent for plain cross-entropy, so nothing is printed.
    if val_metrics.get("loss_diag"):
        print("  [loss] " + ", ".join(
            f"{k} {v:.4g}" for k, v in val_metrics["loss_diag"].items()))


def _save_epoch(checkpoint: dict, history: dict, state: _RunState, *, improved: bool,
                selection_metric: str, val_select: float, epoch: int, save_dir: Path,
                config: dict) -> None:
    """Write one epoch's outputs: best_model.pt when it improved, latest.pt (for a robust
    resume), training_history.json and the plots (every epoch, so an interrupted run can still
    be analyzed), and every save_every-th epoch an epoch_N.pt, keeping the keep_last_n most
    recent."""
    if improved:
        torch.save(checkpoint, save_dir / "best_model.pt")
        print(f"Saved best model ({selection_metric}: {val_select:.4f})")
    torch.save(checkpoint, save_dir / "latest.pt")

    with open(save_dir / "training_history.json", 'w') as f:
        json.dump({
            **history,
            'selection_metric': selection_metric,
            'best_val_select': state.best_val_select,
            'best_epoch': state.best_epoch,
            # Epochs after which a resume changed the training/validation data: val curves are
            # not comparable across a validation change
            'data_changes': state.data_changes,
            'config': config,
        }, f, indent=2)

    plot_losses(history["train_losses"], history["val_losses"], str(save_dir / "loss_plot.png"))
    plot_perplexity(history["val_perplexities"], str(save_dir / "perplexity_plot.png"))
    plot_accuracy(history["val_accuracies"], str(save_dir / "accuracy_plot.png"))
    plot_all_batch_losses(history["all_batch_losses"], str(save_dir / "all_batch_losses.png"))

    if (epoch + 1) % config["checkpoint"]["save_every"] == 0:
        torch.save(checkpoint, save_dir / f"epoch_{epoch+1}.pt")
        prune_epoch_checkpoints(save_dir, config["checkpoint"]["keep_last_n"])


# ============================================================================ #
# The pipeline
# ============================================================================ #
def main(config_path: str):
    """Train, or resume training, the model described by the yaml at config_path."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    seq_cfg = config.get("sequences", {})

    _refuse_removed_settings(model_cfg)
    # Resume target, resolved before any data work: data.model_checkpoint_path when that file
    # exists, else <output_dir>/latest.pt when it exists, else None (train from scratch).
    resume_path = resolve_resume_checkpoint(data_cfg.get("model_checkpoint_path"),
                                            data_cfg["output_dir"], "data.model_checkpoint_path")
    t2e_weight = _resolve_t2e_weight(model_cfg)
    t2e_pred = t2e_weight > 0.0
    loss_mode = model_cfg.get("loss_mode", "cross_entropy")

    cols = resolve_columns(config, parquet_path=data_cfg["parquet_path"])
    time_abs_col = enabled_col(cols, "abstime_col", model_cfg["use_time_abs"], "model.use_time_abs")
    type_col = enabled_col(cols, "type_col", model_cfg["use_type"], "model.use_type")

    mode = data_cfg.get("mode", "streaming")
    valid_modes = {"streaming", "save_intermediate"}
    if mode not in valid_modes:
        raise ValueError(f"data.mode must be one of {sorted(valid_modes)}, got {mode!r}.")

    device = torch.device(resolve_device(config["hardware"]["device"]))
    print(f"Using device: {device}")
    set_seed(config["training"]["seed"])
    save_dir = Path(data_cfg["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Vocabulary and data ───────────────────────────────────────────────────
    stoi, itos, stats = _load_or_build_vocab(data_cfg, seq_cfg, cols, type_col)
    vocab_size = len(stoi)
    num_types = stats.get("num_types", 0)   # may be set from the training data below (types in use)
    print(f"Vocabulary size: {vocab_size}")
    print(stats)
    # The resolved objective is reported by model.loss_summary() once the model exists,
    # rather than re-deriving its numbers from model_cfg here, so the report always shows
    # the settings the loss actually uses.
    if t2e_pred:
        _set_t2e_exclude_ids(model_cfg, stoi, stats, t2e_weight)
    preamble = _checked_preamble(model_cfg, seq_cfg, stats)
    event_times = _load_event_times(data_cfg, cols, t2e_pred)

    if mode == "save_intermediate":
        train_dataset, val_dataset, num_types = _window_store_datasets(
            config, cols, time_abs_col, type_col, preamble, event_times, t2e_pred, num_types)
    else:
        train_dataset, val_dataset, num_types = _streaming_datasets(
            config, cols, time_abs_col, type_col, event_times, t2e_pred, num_types)
    print(f"Number of types: {num_types}")
    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")
    # Only map-style window stores are shuffled by the loader; streaming shuffles in its buffer
    train_loader = _make_loader(train_dataset, config, shuffle=mode == "save_intermediate")
    val_loader = _make_loader(val_dataset, config, shuffle=False)

    # ── Model, optimizer and schedule ─────────────────────────────────────────
    # DAYS CONTRACT: time_age, time_abs and the T2E waiting times are in DAYS (computed from
    # the timestamps), and predicted waiting times are returned in DAYS. Only the T2E rate the
    # network outputs is per year (DAYS_PER_YEAR, ntp_transformer.py); the loss does not depend on it.
    print("[time] time_age, time_abs and the t2e waiting times are in DAYS, computed from the "
          "timestamps")
    model = _build_model(model_cfg, vocab_size, num_types, stoi, t2e_weight, loss_mode,
                         train_loader, device)
    optimizer = _build_optimizer(model, config, device)
    # GradScaler for mixed precision training (no-op on CPU)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    # The learning rate scheduler is stepped once per optimizer step (inside train_one_epoch)
    accum_steps = max(1, int(config["training"].get("accum_steps", 1)))
    steps_per_epoch = _steps_per_epoch(train_loader, train_dataset, config, accum_steps)
    _check_warmup_fits(config, steps_per_epoch, resume_path)
    selection_metric = _selection_metric(model, t2e_pred)

    # ── Resuming from a checkpoint ────────────────────────────────────────────
    # Every checkpoint records the vocabulary fingerprint and the epoch length.
    num_epochs = config["training"]["num_epochs"]
    vocab_fp = vocabulary_fingerprint(stoi, stats, use_type=type_col is not None)
    state = _start_or_resume(resume_path, config=config, cols=cols, stoi=stoi,
                             event_times=event_times, model=model,
                             selection_metric=selection_metric, vocab_fp=vocab_fp,
                             steps_per_epoch=steps_per_epoch, save_dir=save_dir, device=device)
    if state is None:
        return
    scheduler = _build_run_scheduler(optimizer, state, steps_per_epoch, num_epochs,
                                     config["training"])
    if state.ckpt is not None:
        _restore_from_checkpoint(state, resume_path, model=model, optimizer=optimizer,
                                 scheduler=scheduler, val_loader=val_loader,
                                 selection_metric=selection_metric, save_dir=save_dir,
                                 device=device)

    # T2E weight is applied AS-IS (no auto-calibration): loss = token + weight * t2e.
    # Tune it using the t2e/token ratio printed every epoch.
    if t2e_pred:
        print(f"[t2e] weight={model.t2e_weight} applied as a fixed multiplier "
              f"(watch t2e_to_token_ratio in the epoch logs to rebalance)")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = _load_history(state.history_path, state.start_epoch)
    for epoch in range(state.start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # Train one epoch (scheduler is stepped per optimizer step inside)
        train_loss, batch_losses, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=use_amp,
            grad_clip=config["training"]["grad_clip"],
            show_progress=True,
            epoch=epoch,
            num_epochs=num_epochs,
            scheduler=scheduler,
            accum_steps=accum_steps,
        )
        val_metrics = validate(model, val_loader, device)
        val_select = _selection_value(selection_metric, val_metrics)
        _record_epoch(history, train_loss, batch_losses, train_metrics, val_metrics, val_select)
        _print_epoch(train_loss, train_metrics, val_metrics, val_select, selection_metric, model_cfg)

        improved = val_select < state.best_val_select
        if improved:
            state.best_val_select = val_select
            state.best_epoch = epoch + 1
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_metrics['loss'],
            'selection_metric': selection_metric,
            'val_select': val_select,
            'best_val_select': state.best_val_select,
            'best_epoch': state.best_epoch,
            'train_loss': train_loss,
            'config': config,
            'model_config': model.config.to_dict(),   # canonical architecture spec for load_checkpoint
            'num_types': num_types,
            'vocab_size': vocab_size,
            # What a resume checks this run against and adapts to (training/resume.py)
            'vocab_fingerprint': vocab_fp,
            'steps_per_epoch': steps_per_epoch,
            'lr_schedule_start_epoch': state.schedule_start,
            'data_changes': state.data_changes,
        }
        _save_epoch(checkpoint, history, state, improved=improved,
                    selection_metric=selection_metric, val_select=val_select, epoch=epoch,
                    save_dir=save_dir, config=config)

    print("Training completed!")


def cli():
    """Console entry point (tracehr-train): run main on --config."""
    parser = argparse.ArgumentParser(description="Pretrain an NTPTransformer on next-token prediction from MEDS input.")
    parser.add_argument("--config", type=str, required=True, help="Path to training configuration file")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
