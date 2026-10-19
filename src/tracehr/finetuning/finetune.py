# src/tracehr/finetuning/finetune.py
"""The outcome finetuning loop of pipeline 06: training, validation on the task's monitored
metric, checkpoints, resume, the two-phase LP-FT switch, early stopping and the loss plots."""

import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ..generation.model_io import check_vocab_fingerprint, json_safe
from ..training.plots import plot_all_batch_losses, plot_epoch_metric
from ..training.runtime import (
    ProgressLog,
    batch_to_device,
    prune_epoch_checkpoints,
    resolve_device,
)
from .finetune_plots import plot_finetune_batch_losses, plot_finetune_losses
from .outcome_evaluation import evaluate_outcome

# Per-task validation metric to monitor for best-checkpoint / early stopping.
# (metric key in evaluate_outcome output, direction) — higher/lower is better.
_MONITOR = {
    "binary": ("auroc", "max"),
    "multiclass": ("macro_f1", "max"),
    "regression": ("mae", "min"),
    "survival": ("cindex", "max"),
}


@dataclass
class _RunState:
    """What the finetuning loop carries from epoch to epoch, and over a resume. The monitor is
    the task's metric (_MONITOR) until that metric proves undefined on the split, then the
    val loss."""
    optimizer: torch.optim.Optimizer
    scheduler: object
    monitor_key: str
    monitor_mode: str           # "max" or "min": the direction in which monitor_key improves
    best_score: float
    history: dict
    start_epoch: int = 0
    patience_counter: int = 0

    def is_better(self, new, ref) -> bool:
        """True when the score new beats ref in the current monitor direction."""
        return new > ref if self.monitor_mode == "max" else new < ref

    def worst_score(self) -> float:
        """The score any measured one beats in the current monitor direction."""
        return -math.inf if self.monitor_mode == "max" else math.inf

    def monitor_loss_instead(self) -> None:
        """Monitor the val loss for the rest of the run (the task's metric is undefined on
        this split), and say so."""
        print(f"  WARNING: val {self.monitor_key} is undefined on this split — monitoring "
              f"val loss instead for the rest of the run.")
        self.monitor_key, self.monitor_mode = "loss", "min"


def _decay_groups(module, group_lr: float, weight_decay: float) -> list[dict]:
    """AdamW groups of module's trainable parameters at group_lr: weight decay on the
    matrices, none on biases, norms and other 1-d parameters (as in pretraining)."""
    params = [p for p in module.parameters() if p.requires_grad]
    return [
        {"params": [p for p in params if p.ndim >= 2], "lr": group_lr, "weight_decay": weight_decay},
        {"params": [p for p in params if p.ndim < 2], "lr": group_lr, "weight_decay": 0.0},
    ]


def _build_optim_sched(model, train_loader: DataLoader, epochs: int, do_warmup: bool, *,
                       lr: float, backbone_lr: float, weight_decay: float, warmup_epochs: int):
    """AdamW over the head (and the backbone unless frozen) with a scheduler over `epochs`
    epochs: a linear warmup over warmup_epochs (do_warmup only), then a cosine decay to
    1% of the lr. Rebuilt at the two-phase boundary so the freshly unfrozen backbone
    gets its own schedule."""
    groups = _decay_groups(model.head, lr, weight_decay)
    if not model.freeze_backbone:
        groups += _decay_groups(model.backbone, backbone_lr, weight_decay)
    opt = AdamW(groups)
    total = max(epochs * len(train_loader), 1)
    warm = (warmup_epochs * len(train_loader)) if do_warmup else 0
    cos_steps = max(total - warm, 1)

    def cosine_with_min(step):
        """LR factor at step: a cosine from 1 down to 0.01 over cos_steps, then 0.01."""
        progress = min(step / cos_steps, 1.0)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    cosine = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=cosine_with_min)
    if warm > 0:
        warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warm)
        sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warm])
    else:
        sched = cosine
    return opt, sched


def _resume(state: _RunState, resume_path: Path, build, *, model, val_loader: DataLoader,
            device, task: str, output_dir: Path | None, num_epochs: int,
            unfreeze_after_epochs: int | None, vocab_fingerprint: str | None) -> bool:
    """Continue the run saved at resume_path: its weights, optimizer and scheduler states,
    counters and history, then its best model re-scored (_rescore_best). False when the
    checkpoint already completed num_epochs: the history is then final."""
    print(f"Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    if vocab_fingerprint is not None:
        check_vocab_fingerprint(ckpt, vocab_fingerprint, resume_path, "this run's vocabulary")
    model.load_state_dict(ckpt["model_state_dict"])
    state.start_epoch = ckpt.get("epoch", 0)
    state.best_score = ckpt.get("best_score", state.best_score)
    state.patience_counter = ckpt.get("patience_counter", 0)
    # Rebuild phase-2 optimizer BEFORE loading its state if we resume past the boundary.
    # Strictly past: a checkpoint saved AT it holds the phase-1 (head-only) optimizer, and
    # the loop's own boundary switch unfreezes and rebuilds after loading it.
    if unfreeze_after_epochs is not None and state.start_epoch > unfreeze_after_epochs and model.freeze_backbone:
        model.unfreeze_backbone()
        state.optimizer, state.scheduler = build(num_epochs - unfreeze_after_epochs, do_warmup=False)
    state.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict") is not None:
        state.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Resuming from epoch {state.start_epoch}, best {state.monitor_key}: {state.best_score:.4f}")

    history_path = resume_path.parent / "training_history.json"
    if history_path.is_file():
        with open(history_path) as f:
            prev = json.load(f)
        for key in state.history:
            state.history[key] = prev.get(key, [])[:state.start_epoch]

    if state.start_epoch >= num_epochs:
        print(f"Training already complete ({state.start_epoch}/{num_epochs} epochs).")
        return False
    if output_dir:
        _rescore_best(state, ckpt, resume_path, model=model, val_loader=val_loader,
                      device=device, task=task, output_dir=output_dir)
    return True


def _rescore_best(state: _RunState, ckpt: dict, resume_path: Path, *, model,
                  val_loader: DataLoader, device, task: str, output_dir: Path) -> None:
    """Re-score this run's best_model.pt on the current validation data: every epoch from
    here is compared with it on the same data, so best_model.pt stays the best of the whole
    run even when the validation data changed. Only when the checkpoint resumed lies in
    output_dir (its best_model.pt is this run's); a resume into another folder starts
    selection afresh there. Iterating a DataLoader draws from the global torch RNG, so its
    state is restored and an unchanged resume continues exactly."""
    best_path = output_dir / "best_model.pt"
    if resume_path.parent.resolve() == output_dir.resolve() and best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(best["model_state_dict"])
        rng_state = torch.get_rng_state()
        rescored = evaluate_outcome(model, val_loader, device, task)
        torch.set_rng_state(rng_state)
        model.load_state_dict(ckpt["model_state_dict"])
        # The loop's switch to the loss for a monitor undefined on this split, made
        # here so the re-scored best is in the units every new epoch is scored in
        if not math.isfinite(rescored[state.monitor_key]) and state.monitor_key != "loss":
            state.monitor_loss_instead()
        score, saved = rescored[state.monitor_key], best["val_metrics"][state.monitor_key]
        if not math.isclose(score, saved, rel_tol=1e-4):
            print(f"[resume] best_model.pt (epoch {best['epoch']}) scores {state.monitor_key} "
                  f"{score:.4f} on the current validation data, not the {saved:.4f} it "
                  f"was saved with: the validation data changed, so new epochs are "
                  f"compared with it on the current data")
        state.best_score = score if math.isfinite(score) else state.worst_score()
    else:
        print(f"[resume] {output_dir} holds no best_model.pt of the run resumed: "
              f"best-model selection starts afresh")
        state.best_score = state.worst_score()


def _train_epoch(model, train_loader: DataLoader, state: _RunState, *, device, grad_clip: float,
                 show_progress: bool, epoch: int, num_epochs: int) -> tuple[float, list[float]]:
    """One pass over train_loader -> (mean batch loss, the batch losses)."""
    model.train()
    if model.freeze_backbone:
        # Frozen features must be deterministic: keep backbone dropout off.
        model.backbone.eval()

    train_loss = 0.0
    num_batches = 0
    batch_losses = []

    pbar = ProgressLog(train_loader, f"[finetune] epoch {epoch+1}/{num_epochs}", disable=not show_progress)
    for batch in pbar:
        batch = batch_to_device(batch, device)

        state.optimizer.zero_grad(set_to_none=True)
        loss = model(batch)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        state.optimizer.step()
        state.scheduler.step()

        batch_loss = loss.item()
        train_loss += batch_loss
        num_batches += 1
        batch_losses.append(batch_loss)
        pbar.set_postfix({"loss": f"{batch_loss:.4f}", "avg_loss": f"{train_loss/num_batches:.4f}"})

    return train_loss / max(num_batches, 1), batch_losses


def _validate_epoch(model, val_loader: DataLoader, state: _RunState, *, device,
                    task: str) -> tuple[dict, float]:
    """(validation metrics, the score on the current monitor) of one epoch.

    An undefined monitor (AUROC on a one-class val split, a C-index with no admissible
    pairs) is NaN, and NaN loses every comparison: best_model.pt would never be written
    and patience would run out on a metric nobody measured. Measurability is a property
    of the val labels, so it cannot change between epochs — switch once to the loss.
    """
    val_metrics = evaluate_outcome(model, val_loader, device, task)
    val_score = val_metrics[state.monitor_key]
    if not math.isfinite(val_score) and state.monitor_key != "loss":
        state.monitor_loss_instead()
        state.best_score, state.patience_counter = math.inf, 0
        val_score = val_metrics["loss"]
    return val_metrics, val_score


def _save_epoch(state: _RunState, model, val_metrics: dict, val_score: float,
                batch_losses: list[float], *, epoch: int, output_dir: Path, save_every: int,
                keep_last_n: int, vocab_fingerprint: str | None, config: dict | None) -> None:
    """One epoch's checkpoints, history and plots in output_dir: best_model.pt and a reset of
    the patience counter on an improvement (else one more epoch of patience used), latest.pt,
    every save_every-th epoch an epoch_N.pt (keeping the last keep_last_n),
    training_history.json and the loss, val-score and batch-loss plots."""
    if state.is_better(val_score, state.best_score):
        state.best_score = val_score
        state.patience_counter = 0
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "best_score": state.best_score,
            "monitor": state.monitor_key,
            "val_metrics": val_metrics,
            "vocab_fingerprint": vocab_fingerprint,
            "config": config,
        }, output_dir / "best_model.pt")
        print(f"  Saved best model ({state.monitor_key}: {state.best_score:.4f})")
    else:
        state.patience_counter += 1

    latest = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
        "scheduler_state_dict": state.scheduler.state_dict(),
        "best_score": state.best_score,
        "patience_counter": state.patience_counter,
        "val_metrics": val_metrics,
        "vocab_fingerprint": vocab_fingerprint,
        "config": config,
    }
    torch.save(latest, output_dir / "latest.pt")

    if (epoch + 1) % save_every == 0:
        torch.save(latest, output_dir / f"epoch_{epoch+1}.pt")
        prune_epoch_checkpoints(output_dir, keep_last_n)

    history = state.history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(json_safe(history), f, indent=2)

    plot_finetune_losses(history["train_loss"], history["val_loss"], str(output_dir / "loss_plot.png"))
    plot_epoch_metric(history["val_score"], str(output_dir / "val_score_plot.png"),
                      label=f"Val {state.monitor_key}", ylabel=state.monitor_key,
                      title=f"Validation {state.monitor_key} Across Epochs", color="royalblue")
    plot_finetune_batch_losses(batch_losses, epoch + 1, str(output_dir / f"batch_losses_epoch_{epoch+1}.png"))
    plot_all_batch_losses(history["all_batch_losses"], str(output_dir / "all_batch_losses.png"),
                          title="Fine-tuning Loss per Batch (All Epochs)",
                          epoch_labels="legend", announce=False)


def finetune(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task: str = "binary",
    num_epochs: int = 10,
    lr: float = 1e-4,
    backbone_lr: float | None = None,
    weight_decay: float = 0.01,
    warmup_epochs: int = 1,
    device: str = "auto",
    output_dir: str | None = None,
    early_stopping_patience: int = 5,
    show_progress: bool = True,
    save_every: int = 1,
    keep_last_n: int = 2,
    grad_clip: float = 1.0,
    checkpoint_path: str | None = None,
    unfreeze_after_epochs: int | None = None,
    vocab_fingerprint: str | None = None,
    config: dict | None = None,
) -> dict[str, list[float]]:
    """Fine-tune an NTPOutcomeModel (regression | binary | multiclass | survival); returns the
    per-epoch history (train_loss, val_loss, val_score, all_batch_losses).

    Every epoch is validated with evaluate_outcome and scored on the task's monitored metric
    (_MONITOR), or on the val loss once that metric proves undefined on the split. With
    output_dir, an improvement saves best_model.pt, every epoch saves latest.pt (and every
    save_every-th epoch an epoch_N.pt, keeping the last keep_last_n), the history and the
    plots are rewritten, and early_stopping_patience epochs without improvement end the run.

    Args:
        model: NTPOutcomeModel to fine-tune (computes its own task loss).
        task: "regression" | "binary" | "multiclass" | "survival" (drives the monitored metric).
        lr: Learning rate for the task head.
        backbone_lr: Learning rate for the backbone (None = same as lr).
        unfreeze_after_epochs: Two-phase LP-FT boundary. If set, the backbone is
            unfrozen at this epoch and the optimizer/scheduler are rebuilt to
            include it (backbone_lr). None = single phase (probe or full).
        checkpoint_path: Checkpoint file to resume from (latest.pt or epoch_N.pt, in any
            folder); None = start fresh. Pipeline 06 resolves and checks it first.
        vocab_fingerprint: dataset.vocabulary.vocabulary_fingerprint of this run's vocabulary.
            Saved in every checkpoint; a resume from a checkpoint recording another one is
            refused (None = not recorded, not checked).
        config: The run's yaml, saved in every checkpoint so a resume can be checked
            against it (None = not recorded).
    """
    if task not in _MONITOR:
        raise ValueError(f"task must be one of {tuple(_MONITOR)}, got {task!r}.")
    monitor_key, monitor_mode = _MONITOR[task]

    device = torch.device(resolve_device(device))
    model = model.to(device)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    backbone_lr = backbone_lr or lr
    build = partial(_build_optim_sched, model, train_loader, lr=lr, backbone_lr=backbone_lr,
                    weight_decay=weight_decay, warmup_epochs=warmup_epochs)
    optimizer, scheduler = build(num_epochs, do_warmup=True)
    state = _RunState(optimizer=optimizer, scheduler=scheduler, monitor_key=monitor_key,
                      monitor_mode=monitor_mode,
                      best_score=-math.inf if monitor_mode == "max" else math.inf,
                      history={"train_loss": [], "val_loss": [], "val_score": [], "all_batch_losses": []})

    if checkpoint_path:
        resumed = _resume(state, Path(checkpoint_path), build, model=model, val_loader=val_loader,
                          device=device, task=task, output_dir=output_dir, num_epochs=num_epochs,
                          unfreeze_after_epochs=unfreeze_after_epochs,
                          vocab_fingerprint=vocab_fingerprint)
        if not resumed:
            return state.history

    # ── Training loop ─────────────────────────────────────────────────────────
    history = state.history
    for epoch in range(state.start_epoch, num_epochs):
        # Two-phase LP-FT: unfreeze the backbone at the boundary and rebuild optim.
        if unfreeze_after_epochs is not None and epoch == unfreeze_after_epochs and model.freeze_backbone:
            model.unfreeze_backbone()
            state.optimizer, state.scheduler = build(num_epochs - unfreeze_after_epochs, do_warmup=False)
            print(f"[finetune] phase 2: backbone unfrozen (backbone_lr={backbone_lr:g})")

        avg_train_loss, batch_losses = _train_epoch(
            model, train_loader, state, device=device, grad_clip=grad_clip,
            show_progress=show_progress, epoch=epoch, num_epochs=num_epochs)
        history["train_loss"].append(avg_train_loss)
        history["all_batch_losses"].append(batch_losses)

        val_metrics, val_score = _validate_epoch(model, val_loader, state, device=device, task=task)
        history["val_loss"].append(val_metrics["loss"])
        history["val_score"].append(val_score)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val {state.monitor_key}: {val_score:.4f}")

        if output_dir:
            _save_epoch(state, model, val_metrics, val_score, batch_losses, epoch=epoch,
                        output_dir=output_dir, save_every=save_every, keep_last_n=keep_last_n,
                        vocab_fingerprint=vocab_fingerprint, config=config)

        if state.patience_counter >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs (no {state.monitor_key} improvement for {early_stopping_patience} epochs)")
            break

    print("Fine-tuning completed!")
    return history
