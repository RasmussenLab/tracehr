# src/tracehr/training/validation.py
"""One validation pass of a next-token model (pipeline 01)."""

import numpy as np
import torch
import torch.nn.functional as F

from .runtime import ProgressLog, batch_to_device


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    show_progress: bool = False,
) -> dict[str, float]:
    """Validation metrics of model over loader: the combined and token losses, perplexity,
    accuracy, and with T2E the waiting-time loss and MAE, all token-weighted. Perplexity is
    exp(token loss) for a cross-entropy objective, else exp of the cross-entropy of the
    predictive distribution, so it stays comparable across objectives."""
    model.eval()

    total_token_loss_sum = 0.0
    total_t2e_loss_sum = 0.0
    total_correct = 0
    total_tokens = 0
    total_batches = 0
    t2e_enabled = getattr(model, 't2e_pred', False)
    t2e_weight = float(getattr(model, "t2e_weight", 0.0))
    # exp(mean token_loss) is a perplexity for a cross-entropy objective. An objective for
    # which it is not (a distance/margin loss, say) sets token_loss.is_cross_entropy False,
    # and perplexity is then measured from the predictive distribution (softmax of
    # out["logits"] — the same scores used for accuracy) so it stays comparable across
    # objectives. For a cross-entropy objective total_ce_sum stays unused.
    is_ce = model.token_loss.is_cross_entropy
    total_ce_sum = 0.0
    # Objective-specific diagnostics, summed on device and reduced once (see train_one_epoch).
    loss_diag_sums = None

    # T2E metrics
    t2e_mae_sum = 0.0
    t2e_mae_count = 0
    total_t2e_count = 0

    iterator = ProgressLog(loader, "[validation]", disable=not show_progress)

    for batch in iterator:
        batch = batch_to_device(batch, device, non_blocking=True)

        # Forward pass
        outputs = model(batch)
        logits = outputs["logits"]  # [B, T, V]
        targets = batch["labels"]   # [B, T], -100 where nothing is scored

        # Flatten for loss calculation, Token accuracy
        logits_flat = logits.reshape(-1, logits.size(-1))  # [B*T, V]
        targets_flat = targets.reshape(-1)                 # [B*T]
        predictions = logits_flat.argmax(dim=-1)
        non_pad_mask = (targets_flat != -100)
        correct = (predictions[non_pad_mask] == targets_flat[non_pad_mask]).sum().item()

        total_correct += correct
        num_tokens = non_pad_mask.sum().item()
        total_tokens += num_tokens
        total_batches += 1

        # Accumulate token-weighted losses
        if "token_loss" in outputs:
            # Model returns mean loss per token, multiply by num_tokens to get sum
            total_token_loss_sum += outputs["token_loss"].item() * num_tokens
        else:
            # Compute token loss manually
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100, reduction='sum')
            total_token_loss_sum += loss.item()

        # Non-CE objectives only: genuine token cross-entropy from the predictive
        # distribution, for a comparable perplexity (token_loss is not a CE there).
        if not is_ce:
            total_ce_sum += model.token_loss.token_cross_entropy(logits_flat, targets_flat).item()

        if t2e_enabled and "t2e_loss" in outputs:
            batch_t2e_count = int(outputs.get("t2e_count", num_tokens))
            total_t2e_loss_sum += outputs["t2e_loss"].item() * batch_t2e_count
            total_t2e_count += batch_t2e_count

        if "loss_diag" in outputs:
            if loss_diag_sums is None:
                loss_diag_sums = {k: torch.zeros((), device=v.device)
                                  for k, v in outputs["loss_diag"].items()}
            for key, value in outputs["loss_diag"].items():
                loss_diag_sums[key] += value.detach().float() * num_tokens

        # T2E MAE (if enabled)
        if t2e_enabled and "predicted_time_delta" in outputs and "time_deltas" in batch:
            pred_times = outputs["predicted_time_delta"]  # [B, T]
            time_deltas = batch["time_deltas"]  # [B, T]
            valid_mask = (time_deltas >= 0) & (targets != -100)

            if valid_mask.any():
                mae = torch.abs(pred_times[valid_mask] - time_deltas[valid_mask]).sum().item()
                t2e_mae_sum += mae
                t2e_mae_count += valid_mask.sum().item()

        # Update progress bar
        if show_progress:
            current_loss = total_token_loss_sum / max(total_tokens, 1)
            current_acc = total_correct / max(total_tokens, 1)
            iterator.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}"
            })

    # Calculate final metrics using token-weighted averages
    avg_token_loss = total_token_loss_sum / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)
    # CE objectives: exp(avg token loss). Others: exp(avg CE from the predictive
    # distribution), so perplexity stays comparable across objectives.
    perplexity = np.exp(avg_token_loss if is_ce else (total_ce_sum / max(total_tokens, 1)))

    # T2E is averaged over observed waiting times, not all token targets.
    avg_t2e_loss = total_t2e_loss_sum / max(total_t2e_count, 1) if t2e_enabled else 0.0
    weighted_t2e_loss = t2e_weight * avg_t2e_loss
    combined_loss = avg_token_loss + weighted_t2e_loss

    results = {
        "loss": combined_loss,          # token mean + weighted waiting-time mean
        "token_loss": avg_token_loss,   # token-weighted
        "perplexity": perplexity,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "total_batches": total_batches,
        "total_sequences": len(loader.dataset),
    }

    if t2e_enabled:
        results["t2e_loss"] = avg_t2e_loss
        results["total_t2e_count"] = total_t2e_count
        results["weighted_t2e_loss"] = weighted_t2e_loss
        results["t2e_to_token_ratio"] = weighted_t2e_loss / avg_token_loss if avg_token_loss > 0 else 0.0
        results["t2e_loss_fraction"] = weighted_t2e_loss / combined_loss if combined_loss > 0 else 0.0
        if t2e_mae_count > 0:
            results["t2e_mae"] = t2e_mae_sum / t2e_mae_count

    # Nested rather than merged into results: self-delimiting, so a caller can report an
    # objective's diagnostics without knowing which keys are standard and which are its.
    if loss_diag_sums is not None:
        results["loss_diag"] = {k: (v / max(total_tokens, 1)).item()
                                for k, v in loss_diag_sums.items()}

    return results
