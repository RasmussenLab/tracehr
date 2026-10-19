# src/tracehr/training/one_epoch.py
"""One training epoch of a next-token model (pipeline 01)."""

import torch
from torch import nn

from .runtime import ProgressLog, batch_to_device


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = True,
    grad_clip: float | None = None,
    show_progress: bool = True,
    epoch: int = 0,
    num_epochs: int = 1,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,  # stepped once per optimizer step
    accum_steps: int = 1,                                           # gradient accumulation steps
) -> tuple[float, list[float], dict[str, float]]:
    """Train the model for one pass over dataloader.

    Mixed precision runs on CUDA only (amp); gradients are accumulated over accum_steps
    batches and clipped to grad_clip before each optimizer step, and the scheduler steps
    once per optimizer step. Returns (combined loss, every batch's loss, metrics): the token
    loss is averaged over the scored tokens and the T2E loss over the observed waiting
    times, and the objective's diagnostics, if any, are averaged under "loss_diag".
    """
    model.train()
    use_amp = amp and device.type == "cuda"
    # GradScaler is a no-op when disabled, so one code path covers CPU and AMP
    if scaler is None:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    total_token_loss_sum = 0.0
    total_t2e_loss_sum = 0.0
    total_tokens = 0
    total_t2e_count = 0
    batch_losses = []
    t2e_enabled = getattr(model, "t2e_pred", False)
    t2e_weight = float(getattr(model, "t2e_weight", 0.0))
    # Objective-specific diagnostics (whatever the token loss reports), summed ON DEVICE
    # and reduced once at the end: .item() per key per step would add a GPU sync to every
    # training step to log numbers only read once an epoch. Empty for plain cross-entropy.
    loss_diag_sums: dict[str, torch.Tensor] | None = None

    iterator = ProgressLog(dataloader, f"[train] epoch {epoch+1}/{num_epochs}", disable=not show_progress)

    optimizer.zero_grad(set_to_none=True)
    last_step = len(dataloader) - 1

    for step, batch in enumerate(iterator):
        batch = batch_to_device(batch, device, non_blocking=True)
        # Count tokens in this batch (excluding padding)
        num_tokens = (batch["labels"] != -100).sum().item()
        total_tokens += num_tokens

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch)
            loss = outputs["loss"]

        scaler.scale(loss / accum_steps).backward()

        # Step the optimizer every accum_steps batches (and on the final batch)
        if (step + 1) % accum_steps == 0 or step == last_step:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        # Track metrics
        batch_loss = loss.item()
        batch_losses.append(batch_loss)
        if "token_loss" in outputs:
            total_token_loss_sum += outputs["token_loss"].item() * num_tokens
        if "t2e_loss" in outputs:
            # NTPTransformer exposes the exact mask count used by its T2E mean; a model that
            # does not report t2e_count falls back to the scored-token count.
            batch_t2e_count = int(outputs.get("t2e_count", num_tokens))
            total_t2e_loss_sum += outputs["t2e_loss"].item() * batch_t2e_count
            total_t2e_count += batch_t2e_count
        if "loss_diag" in outputs:
            if loss_diag_sums is None:
                loss_diag_sums = {k: torch.zeros((), device=v.device)
                                  for k, v in outputs["loss_diag"].items()}
            for key, value in outputs["loss_diag"].items():
                loss_diag_sums[key] += value.detach().float() * num_tokens

        # Update progress line
        if show_progress:
            current_avg_loss = total_token_loss_sum / max(total_tokens, 1)
            postfix = {
                "loss": f"{batch_loss:.4f}",
                "token": f"{current_avg_loss:.4f}"
            }
            if t2e_enabled and "t2e_loss" in outputs:
                postfix["t2e"] = f"{outputs['t2e_loss'].item():.4f}"
                postfix["wt2e"] = f"{t2e_weight * outputs['t2e_loss'].item():.4f}"
            iterator.set_postfix(postfix)

    # Each objective uses its own population: token targets for token loss, observed waiting
    # times for T2E. Preamble and boundary positions can belong to only the first population.
    avg_token_loss = total_token_loss_sum / max(total_tokens, 1)
    avg_t2e_loss = total_t2e_loss_sum / max(total_t2e_count, 1) if t2e_enabled else 0.0
    weighted_t2e_loss = t2e_weight * avg_t2e_loss
    combined_loss = avg_token_loss + weighted_t2e_loss

    metrics = {
        "loss": combined_loss,
        "token_loss": avg_token_loss,
        "total_tokens": total_tokens,
    }
    if t2e_enabled:
        metrics["t2e_loss"] = avg_t2e_loss
        metrics["total_t2e_count"] = total_t2e_count
        metrics["weighted_t2e_loss"] = weighted_t2e_loss
        metrics["t2e_to_token_ratio"] = weighted_t2e_loss / avg_token_loss if avg_token_loss > 0 else 0.0
        metrics["t2e_loss_fraction"] = weighted_t2e_loss / combined_loss if combined_loss > 0 else 0.0
    # Nested rather than merged into metrics: self-delimiting, so a caller can report an
    # objective's diagnostics without knowing which keys are standard and which are its.
    if loss_diag_sums is not None:
        metrics["loss_diag"] = {k: (v / max(total_tokens, 1)).item()
                                for k, v in loss_diag_sums.items()}

    return combined_loss, batch_losses, metrics
