#!/usr/bin/env python3
# src/tracehr/pipelines/finetune_model.py
"""Pipeline 06: task-generic outcome finetuning from the MEDS input.

Inputs: the MEDS input (encoded on the fly), ONE outcomes file (patient id,
anchor timestamp, target), and the ORIGINAL pretraining split id files. The outcomes
file is auto-partitioned by intersecting its patients with train/val/test ids,
so pretraining split membership is preserved exactly. A patient may have several
anchors (one sample each, all in the patient's split); rows missing the anchor or
the outcome are dropped and reported (load_anchors). The pretrained LM head is
replaced by a task head (regression | binary | multiclass | survival) and finetuned
with a selectable strategy (probe | full | two_phase). Optional IG + cohort + attention
explainability. Outputs: checkpoints, test_metrics.json, predictions.parquet,
and an explainability/ directory.

SURVIVAL: set finetuning.task: survival and name anchor_columns.censor_timestamp_col.
The outcomes file then carries the 0/1 EVENT indicator in target_col and the timestamp of
the event or of the censoring in censor_timestamp_col; the duration is the days from the
anchor to it (load_anchors). The head trains on the Cox partial
likelihood and is monitored by Harrell's C-index, computed by the same survival_core
helper stage 05 reports through - so the two stages' C-indices are directly comparable.

DAYS CONTRACT: Age and anchors are all in DAYS.
Cutoff contract (same flag as 03/04/05): remove_cutoff_token: true (the code default)
keeps Age < (anchor - time_before_event), so the event exactly AT the anchor is
EXCLUDED; false keeps Age <= (anchor - time_before_event), so the anchor event is KEPT.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from tracehr.dataset.cohort import load_anchors, load_patient_id_filter
from tracehr.dataset.column_mapping import enabled_col, resolve_anchor_columns, resolve_columns
from tracehr.dataset.meds_reader import load_event_times
from tracehr.dataset.ntp_batching import collate_fn
from tracehr.dataset.streaming_datasets import StreamingAnchorsDataset
from tracehr.dataset.vocabulary import load_vocab, vocabulary_fingerprint
from tracehr.finetuning.finetune import finetune
from tracehr.finetuning.outcome_evaluation import evaluate_outcome, plot_outcome_eval
from tracehr.finetuning.outcome_explain import explain_outcome_predictions
from tracehr.finetuning.outcome_model import NTPOutcomeModel
from tracehr.finetuning.survival_core import detect_survival_task
from tracehr.generation.model_io import check_vocab_fingerprint, json_safe, load_checkpoint
from tracehr.training.resume import config_differences, resolve_resume_checkpoint
from tracehr.training.runtime import batch_to_device, resolve_device

# strategy -> (freeze_backbone at construction, unfreeze_after_epochs)
_STRATEGY = {
    "probe": (True, None),      # backbone frozen throughout
    "full": (False, None),      # all trainable from epoch 0
    "two_phase": (True, "freeze_epochs"),  # head-only, then unfreeze (LP-FT)
}

# Yaml keys a resume may change (a key also covers everything under it). Every other key
# shapes the run being continued, so changing it refuses the resume.
_RESUME_MAY_CHANGE = (
    "hardware",                  # where the run executes, not what it learns
    "checkpoint.finetuned_checkpoint_path",  # the checkpoint resumed from
    "finetuning.output_dir",
    "finetuning.num_epochs",     # raising it extends the run
    "data.vocab_path",           # the file may move; its content is checked (vocab fingerprint)
    "explainability",            # runs after training
)
# What a resume reads from a finetune checkpoint. best_model.pt has no optimizer or scheduler
# state, so only latest.pt / epoch_N.pt can be continued.
_RESUME_FIELDS = ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch",
                  "vocab_fingerprint", "config")


def _read_ids(path: str) -> set:
    """The patient ids of the id file at path, as a set."""
    return set(load_patient_id_filter(path).tolist())


def _validate_targets(task: str, targets: np.ndarray, durations: np.ndarray | None = None):
    """Loudly validate + summarize the target column for the chosen task.

    `durations` is survival-only and required there: the label is the (duration, event)
    pair, and detect_survival_task owns every check on it - the same validator stage 05
    runs, so a swapped pair is caught identically in both.
    """
    if task == "survival":
        detect_survival_task(pd.DataFrame({"event": targets, "duration": durations}),
                             event_col="event", duration_col="duration")
    elif task == "binary":
        uniq = np.unique(targets)
        if not set(uniq.tolist()).issubset({0, 1}):
            raise ValueError(f"binary task requires targets in {{0,1}}, found {uniq.tolist()}")
        prev = float(targets.mean())
        print(f"  [binary] prevalence={prev:.4f}  (pos={int(targets.sum())}, neg={int(len(targets)-targets.sum())})")
    elif task == "multiclass":
        if np.any(targets < 0) or np.any(targets != targets.astype(int)):
            raise ValueError("multiclass task requires non-negative integer targets.")
        classes, counts = np.unique(targets.astype(int), return_counts=True)
        print(f"  [multiclass] K={len(classes)} classes, balance={dict(zip(classes.tolist(), counts.tolist()))}")
    else:  # regression
        n_bad = int((~np.isfinite(targets)).sum())  # NaN rows were already dropped by load_anchors
        if n_bad:
            raise ValueError(f"regression task requires finite targets, found {n_bad} infinite value(s)")
        print(f"  [regression] range=[{float(targets.min()):.3f}, {float(targets.max()):.3f}]  mean={float(targets.mean()):.3f}")


def _check_resume_checkpoint(path: Path, config: dict, vocab_fp: str) -> None:
    """Refuse a checkpoint that cannot continue THIS finetuning run.

    It must be a finetune checkpoint saved with its config (latest.pt or epoch_N.pt), trained
    with the same settings - only _RESUME_MAY_CHANGE may differ, all differences are listed -
    and on the same vocabulary. Read memory-mapped, so its weights are not loaded here.
    """
    fresh = ("finetune from scratch: set checkpoint.finetuned_checkpoint_path to null and use "
             "a finetuning.output_dir without a latest.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    missing = [k for k in _RESUME_FIELDS if ckpt.get(k) is None]
    if not missing and "finetuning" not in ckpt["config"]:
        missing = ["config.finetuning"]
    if missing:
        raise ValueError(
            f"{path} is not a finetune checkpoint this run can continue: it has no {missing} "
            f"(best_model.pt has no optimizer state, a pretraining checkpoint no finetuning "
            f"settings, a checkpoint saved before resumes were checked no config). Point "
            f"checkpoint.finetuned_checkpoint_path at a latest.pt or epoch_N.pt saved by "
            f"pipeline 06, or {fresh}.")
    forbidden, allowed = config_differences(ckpt["config"], config, may_change=_RESUME_MAY_CHANGE)
    if forbidden:
        raise ValueError(
            f"Cannot resume from {path}: these settings differ from the run that produced "
            f"it:\n  " + "\n  ".join(forbidden)
            + f"\nResume with the settings it was trained with, or {fresh}.")
    check_vocab_fingerprint(ckpt, vocab_fp, path, config["data"]["vocab_path"])
    if allowed:
        print("[resume] allowed setting changes: " + "; ".join(allowed))


def _check_finetune_settings(ft: dict, config: dict) -> tuple[str, str]:
    """(task, strategy) of the finetuning block, after checking them, two_phase's
    freeze_epochs and explainability.n_steps - all before any data is read."""
    task = ft["task"]
    if task not in ("regression", "binary", "multiclass", "survival"):
        raise ValueError(
            f"finetuning.task must be regression|binary|multiclass|survival, got {task!r}")
    strategy = ft.get("strategy", "two_phase")
    if strategy not in _STRATEGY:
        raise ValueError(f"finetuning.strategy must be one of {tuple(_STRATEGY)}, got {strategy!r}")
    # two_phase unfreezes the backbone when the epoch number reaches freeze_epochs. A value
    # that is missing, not a whole number, or not below num_epochs is never reached, and the
    # run would silently stay a probe. (bool is an int subclass, so true is refused explicitly.)
    if strategy == "two_phase":
        fe = ft.get("freeze_epochs")
        if isinstance(fe, bool) or not isinstance(fe, int) or not 1 <= fe < ft["num_epochs"]:
            raise ValueError(
                f"finetuning.freeze_epochs must be a whole number from 1 to num_epochs - 1 for "
                f"strategy two_phase (num_epochs is {ft['num_epochs']}), got {fe!r}. Use strategy "
                f"probe to keep the backbone frozen, or full to train it from the start.")
    # Checked here, before training, because explainability runs only after it:
    # n_steps < 1 divides by zero in Integrated Gradients and every attribution is NaN.
    expl_cfg = config.get("explainability", {})
    n_steps = expl_cfg.get("n_steps", 50)
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"explainability.n_steps must be a whole number >= 1, got {n_steps!r}.")
    return task, strategy


def _reconcile_max_seq_len(ft: dict, backbone) -> int:
    """The finetune window width. The finetune windows must not exceed the backbone's context
    (the forward would raise mid-training): default to it when unset, respect a smaller
    value, and fail fast on a larger one."""
    backbone_msl = backbone.max_seq_len
    requested_msl = ft.get("max_seq_len")
    if requested_msl is None:
        return backbone_msl
    if int(requested_msl) > backbone_msl:
        raise ValueError(
            f"finetuning.max_seq_len={requested_msl} exceeds the pretrained backbone's "
            f"max_seq_len={backbone_msl}. Set it <= {backbone_msl} (or omit it to use the "
            f"backbone's context).")
    max_seq_len = int(requested_msl)
    if max_seq_len < backbone_msl:
        print(f"[finetune] max_seq_len={max_seq_len} (< backbone {backbone_msl}): "
              f"finetuning on a reduced context window.")
    return max_seq_len


def _resolve_splits(anchors: pd.DataFrame, ft: dict, task: str, id_col: str, target_col: str,
                    duration_col: str | None) -> tuple[dict, int | None]:
    """({"train"|"val"|"test": patient ids}, num_classes): the outcomes file partitioned by the
    ORIGINAL pretraining split id files, every anchor of a patient in that patient's split.
    Each split's patient and anchor counts are printed and its targets checked; an empty train
    or val split is refused.

    num_classes (multiclass only, else None) is counted over the patients in a split: a class
    id only unassigned patients carry never reaches the model and must not widen the head.
    """
    anchor_pids = set(anchors[id_col])
    split_ids = {name: _read_ids(ft[f"{name}_ids"]) for name in ("train", "val", "test")}
    resolved = {name: anchor_pids & ids for name, ids in split_ids.items()}
    assigned = resolved["train"] | resolved["val"] | resolved["test"]
    dropped = anchor_pids - assigned

    for name in ("train", "val", "test"):
        pids = resolved[name]
        rows = anchors[anchors[id_col].isin(pids)]
        print(f"  {name:<5}: {len(pids)} patients, {len(rows)} anchors")
        if len(rows):
            _validate_targets(task, rows[target_col].to_numpy(),
                              rows[duration_col].to_numpy() if duration_col is not None else None)
    print(f"  dropped (in outcomes file but no split): {len(dropped)}")
    if not resolved["train"] or not resolved["val"]:
        raise ValueError("Empty train or val split after resolution — check the id files and outcomes file overlap.")

    num_classes = (int(anchors.loc[anchors[id_col].isin(assigned), target_col].astype(int).max()) + 1
                   if task == "multiclass" else None)
    return resolved, num_classes


def _class_weights(task: str, train_labels: np.ndarray, num_classes: int | None,
                   target_col: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """(pos_weight, class_weights), set automatically from the TRAIN labels: binary ->
    negatives / positives, multiclass -> inverse class frequencies; (None, None) otherwise."""
    pos_weight = class_weights = None
    if task == "binary":
        n_pos = float((train_labels == 1).sum())
        n_neg = float((train_labels == 0).sum())
        # A one-class train split has nothing to learn: all positive gives pos_weight=0, hence zero
        # loss and zero gradients; all negative trains a constant "no event" predictor.
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"binary: the train split has only one class (pos={int(n_pos)}, neg={int(n_neg)} windows); "
                f"finetuning needs both 0 and 1 targets in train. Check '{target_col}' and the split id files.")
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
        print(f"pos_weight (auto): {pos_weight.item():.4f}")
    elif task == "multiclass":
        counts = np.bincount(train_labels.astype(int), minlength=num_classes).astype(np.float32)
        # Class ids index the head's outputs: they must be 0..K-1 with every class in train. A gap
        # (e.g. 1-based ids) or a class missing from train leaves an output that is never a target
        # and makes ovr_auroc NaN on every split.
        missing = np.flatnonzero(counts == 0).tolist()
        if missing:
            raise ValueError(
                f"multiclass: class id(s) {missing} have no train windows, but the head has "
                f"num_classes={num_classes} (largest class id + 1). '{target_col}' must hold ids "
                f"0..{num_classes - 1}, each present in the train split; re-code it "
                f"(e.g. 1-based ids -> subtract 1).")
        class_weights = torch.tensor(len(train_labels) / (num_classes * np.clip(counts, 1.0, None)), dtype=torch.float32)
        print(f"class_weights (auto): {class_weights.tolist()}")
    return pos_weight, class_weights


def _write_test_metrics(model: NTPOutcomeModel, test_loader: DataLoader, device, task: str,
                        pretrain_ckpt: dict, output_dir: Path) -> None:
    """Evaluate on the test split, print the metrics and write test_metrics.json, with the
    pretraining arm (from the backbone's checkpoint: the architecture and its single T2E
    setting live in the persisted model_config)."""
    test_metrics = evaluate_outcome(model, test_loader, device, task, show_progress=True)
    pretrain_mc = pretrain_ckpt.get("model_config", {})
    test_metrics["pretrain_arm"] = {
        "loss_mode": pretrain_mc.get("loss_mode"),
        "t2e_weight": pretrain_mc.get("t2e_weight"),
    }
    print("Test metrics:", json.dumps(json_safe({k: v for k, v in test_metrics.items() if not isinstance(v, dict)}), indent=2))
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(json_safe(test_metrics), f, indent=2)


def _write_predictions(model: NTPOutcomeModel, datasets: dict, task: str, batch_size: int,
                       loader_kwargs: dict, device, output_dir: Path) -> None:
    """predictions.parquet: one row per window of every split in datasets (train, val, test)
    with patient_id, target (survival: the event indicator, and the duration), prediction,
    split and the window's own anchor_age."""
    preds_rows = []
    for split_name, ds in datasets.items():
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
        model.eval()
        pids = ds.patient_ids
        anchor_ages = ds.anchor_ages  # each window's own anchor
        cursor = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch_to_device(batch, device)
                out = model(batch)
                bs = batch["input_ids"].shape[0]
                if task in ("binary", "multiclass"):
                    pred = out["probs"].cpu().numpy().tolist()  # prob / class-prob vector
                else:
                    pred = out["predictions"].cpu().numpy().tolist()  # value / log risk
                labels = batch["labels"].cpu().numpy()
                for b in range(bs):
                    # Survival splits its two-column label back into scalars here, giving
                    # the same (target, duration) pair stage 05 writes; every other task
                    # keeps the single value. Key order is the saved parquet's column order.
                    row = {"patient_id": str(pids[cursor + b])}
                    if task == "survival":
                        row["target"] = float(labels[b][1])     # the 0/1 event indicator
                        row["duration"] = float(labels[b][0])   # time to event/censoring
                    else:
                        row["target"] = labels[b].tolist()
                    row["prediction"] = pred[b]
                    row["split"] = split_name
                    row["anchor_age"] = float(anchor_ages[cursor + b])
                    preds_rows.append(row)
                cursor += bs
    pd.DataFrame(preds_rows).to_parquet(output_dir / "predictions.parquet", index=False)
    print(f"Saved predictions.parquet ({len(preds_rows)} rows)")


def _plot_test_eval(model: NTPOutcomeModel, test_loader: DataLoader, task: str, device,
                    output_dir: Path) -> None:
    """test_eval.png: the task's evaluation plot of the test split (plot_outcome_eval)."""
    stats = {"labels": [], "preds": []}
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch_to_device(batch, device)
            out = model(batch)
            stats["preds"].append(
                (out["probs"] if task in ("binary", "multiclass")
                 else out["predictions"]).cpu().numpy())
            stats["labels"].append(batch["labels"].cpu().numpy())
    plot_outcome_eval(task, np.concatenate(stats["labels"]), np.concatenate(stats["preds"]),
                      str(output_dir / "test_eval.png"))


def main(config_path: str):
    """Finetune, evaluate and explain the outcome model of the yaml at config_path."""
    print("Loading the config file...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(resolve_device(config["hardware"]["device"]))
    print(f"Using device: {device}")

    torch.manual_seed(2000)
    np.random.seed(2000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2000)

    ft = config["finetuning"]
    task, strategy = _check_finetune_settings(ft, config)
    expl_cfg = config.get("explainability", {})

    output_dir = Path(ft["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # The finetune checkpoint to resume from (None = finetune from scratch), checked below
    # once the backbone's vocabulary is known
    resume_path = resolve_resume_checkpoint(config["checkpoint"].get("finetuned_checkpoint_path"),
                                            output_dir, "checkpoint.finetuned_checkpoint_path")

    cols = resolve_columns(config, parquet_path=ft["raw_parquet_path"])
    acols = resolve_anchor_columns(config, parquet_path=ft["anchors_path"], require_target=True,
                                   require_duration=(task == "survival"))
    id_col, age_col, target_col = acols["id_col"], acols["age_col"], acols["target_col"]
    # Survival only; None everywhere else, which leaves every label path below scalar.
    duration_col = acols["duration_col"] if task == "survival" else None
    print(f"[time] time_age and the anchors are in DAYS, computed from the timestamps. "
          f"task={task}, strategy={strategy}")

    # ── Vocabulary and pretrained backbone ────────────────────────────────────
    print("\nLoading vocabulary + pretrained model...")
    stoi, itos, vocab_stats = load_vocab(config["data"]["vocab_path"])
    vocab_itos = {i: tok for i, tok in enumerate(itos)}
    print(f"Vocabulary size: {len(itos)}")

    # ── Load the pretrained backbone ONCE and reconcile window geometry ───────
    # This load is reused for the model and the provenance in test_metrics.json.
    backbone, pretrain_ckpt = load_checkpoint(
        config["checkpoint"]["pretrained_checkpoint_path"], config["data"]["vocab_path"], str(device))
    # The vocabulary the backbone was verified against, recorded in every finetune checkpoint
    # so a resume and the best-model reload below refuse one trained on another vocabulary.
    vocab_fp = vocabulary_fingerprint(stoi, vocab_stats, use_type=bool(backbone.config.use_type))
    # A resume must continue THIS run: refuse another one before any dataset is built, and
    # only then record this run's settings (a refused run leaves the folder's record intact)
    if resume_path is not None:
        _check_resume_checkpoint(resume_path, config, vocab_fp)
    max_seq_len = _reconcile_max_seq_len(ft, backbone)

    # Each patient's rows are checked once; the readers leave out the rows and patients
    # that fail and compute time_age and time_abs from the timestamps (the anchors too).
    time_abs_col = enabled_col(cols, "abstime_col", backbone.use_time_abs, "model.use_time_abs")
    event_times = load_event_times(ft["raw_parquet_path"], cols, config["data"]["vocab_path"])

    # ── Automatic split resolution ────────────────────────────────────────────
    # Rows missing the anchor age or the outcome are dropped and reported by load_anchors.
    print("\nResolving splits from the original pretraining id files...")
    # For survival, the duration (days from the anchor to the censor timestamp) is computed here.
    anchors = load_anchors(
        ft["anchors_path"], id_col, age_col, [target_col],
        event_times=event_times, timestamp_col=acols["timestamp_col"],
        censor_timestamp_col=acols["censor_timestamp_col"] if task == "survival" else None)
    resolved, num_classes = _resolve_splits(anchors, ft, task, id_col, target_col, duration_col)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # ── Per-split streaming datasets (shared cutoff contract) ─────────────────
    def make_dataset(pids: set) -> StreamingAnchorsDataset:
        """The anchored windows of the patients pids, cut by the shared cutoff rule."""
        return StreamingAnchorsDataset(
            parquet_path=ft["raw_parquet_path"],   # streamed + encoded on the fly (no intermediate)
            vocab_path=config["data"]["vocab_path"],
            anchors=anchors,
            patient_ids=pids,
            time_before_event=ft.get("time_before_event", 0),
            remove_cutoff_token=ft.get("remove_cutoff_token", True),
            max_seq_len=max_seq_len,               # reconciled to the pretrained backbone
            id_col=cols["id_col"],
            tok_col=cols["tok_col"],
            time_age_col=cols["age_col"],
            time_abs_col=time_abs_col,
            type_col=enabled_col(cols, "type_col", backbone.use_type, "model.use_type"),
            outcome_patient_col=id_col,
            outcome_age_col=age_col,
            outcome_col=target_col,
            outcome_duration_col=duration_col,
            event_times=event_times,
        )

    train_dataset = make_dataset(resolved["train"])
    val_dataset = make_dataset(resolved["val"])
    print(f"\nTrain windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")
    pos_weight, class_weights = _class_weights(task, train_dataset.labels, num_classes, target_col)

    # ── Build the outcome model ───────────────────────────────────────────────
    freeze_backbone, unfreeze_field = _STRATEGY[strategy]
    unfreeze_after_epochs = ft.get(unfreeze_field) if isinstance(unfreeze_field, str) else unfreeze_field

    # Built on the backbone loaded above, so its single load and the reconciled max_seq_len
    # are reused.
    model = NTPOutcomeModel(
        backbone=backbone,
        task=task,
        num_classes=num_classes,
        pooling=ft.get("pooling", "mean"),
        dropout=ft.get("dropout", 0.1),
        freeze_backbone=freeze_backbone,
        pos_weight=pos_weight,
        class_weights=class_weights,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,} | freeze_backbone={freeze_backbone} | unfreeze_after_epochs={unfreeze_after_epochs}")

    loader_kwargs = dict(num_workers=config["hardware"]["num_workers"],
                         pin_memory=config["hardware"]["pin_memory"],
                         collate_fn=collate_fn)
    batch_size = ft["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\n  FINE-TUNING\n" + "=" * 60)
    finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task=task,
        num_epochs=ft["num_epochs"],
        lr=ft["lr"],
        backbone_lr=ft.get("backbone_lr"),
        weight_decay=ft.get("weight_decay", 0.01),
        warmup_epochs=ft.get("warmup_epochs", 1),
        device=str(device),
        output_dir=output_dir,
        early_stopping_patience=ft.get("early_stopping_patience", 5),
        grad_clip=ft.get("grad_clip", 1.0),
        checkpoint_path=resume_path,
        unfreeze_after_epochs=unfreeze_after_epochs,
        vocab_fingerprint=vocab_fp,
        config=config,
    )

    # ── Load best checkpoint ──────────────────────────────────────────────────
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        check_vocab_fingerprint(ckpt, vocab_fp, best_ckpt, config["data"]["vocab_path"])
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint (epoch {ckpt.get('epoch', '?')})")

    # ── Test evaluation and predictions ───────────────────────────────────────
    if not resolved["test"]:
        print(f"\nNo test split resolved. Done! Outputs saved to: {output_dir}")
        return
    print("\n" + "=" * 60 + "\n  TEST EVALUATION\n" + "=" * 60)
    test_dataset = make_dataset(resolved["test"])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    print(f"Test windows: {len(test_dataset)}")
    _write_test_metrics(model, test_loader, device, task, pretrain_ckpt, output_dir)
    _write_predictions(model, {"train": train_dataset, "val": val_dataset, "test": test_dataset},
                       task, batch_size, loader_kwargs, device, output_dir)
    _plot_test_eval(model, test_loader, task, device, output_dir)

    # ── Explainability ────────────────────────────────────────────────────────
    if expl_cfg.get("enabled", True):
        print("\n" + "=" * 60 + "\n  EXPLAINABILITY\n" + "=" * 60)
        explain_outcome_predictions(
            model=model,
            dataset=test_dataset,
            vocab_itos=vocab_itos,
            task=task,
            output_dir=str(output_dir),
            collate_fn=collate_fn,
            num_samples=expl_cfg.get("num_samples", 20),
            top_k_tokens=expl_cfg.get("top_k_tokens", 25),
            n_steps=expl_cfg.get("n_steps", 50),
            batch_size=batch_size,
            device=str(device),
            plot_attention=expl_cfg.get("plot_attention", True),
        )

    print(f"\nDone! Outputs saved to: {output_dir}")


def cli():
    """Console entry point (tracehr-finetune): run main on --config."""
    parser = argparse.ArgumentParser(description="Task-generic outcome finetuning from MEDS input")
    parser.add_argument("--config", type=str, required=True, help="Path to the stage's YAML config")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
