# src/tracehr/generation/evaluation.py
"""Next-token test metrics of a trained model on held-out windows (pipeline 02): the
metrics, their figure and their console summary."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..training.runtime import ProgressLog, batch_to_device, resolve_device


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "auto",
    show_progress: bool = True,
) -> dict[str, float]:
    """Next-token metrics of model over dataloader, token-weighted over the scored positions
    (padding and the front block excluded): the loss (token loss, plus the weighted T2E
    loss with T2E), perplexity, accuracy, top-5/top-10 accuracy, mean reciprocal rank and,
    with T2E, the waiting-time loss and MAE. Perplexity is exp(token loss) for a
    cross-entropy objective, else exp of the cross-entropy of the predictive distribution,
    so it stays comparable across objectives."""

    device = torch.device(resolve_device(device))

    model.eval()
    model.to(device)

    total_loss_batch_sum = 0.0
    total_token_loss_sum = 0.0
    total_t2e_loss_sum = 0.0
    total_correct = 0
    total_tokens = 0
    total_batches = 0
    top5_sum = 0.0
    top10_sum = 0.0
    mrr_sum = 0.0

    t2e_enabled = getattr(model, 't2e_pred', False)
    t2e_weight = float(getattr(model, "t2e_weight", 0.0))
    t2e_available = False
    t2e_mae_sum = 0.0
    t2e_mae_count = 0
    total_t2e_count = 0
    # exp(mean token_loss) is a perplexity for a cross-entropy objective. An objective for
    # which it is not sets token_loss.is_cross_entropy False, and perplexity is then measured
    # from the predictive distribution (softmax of out["logits"] — the same scores used for
    # accuracy/ranks) so it stays comparable across objectives.
    is_ce = model.token_loss.is_cross_entropy
    total_ce_sum = 0.0

    iterator = ProgressLog(dataloader, "[test]", disable=not show_progress)

    for batch in iterator:
        batch = batch_to_device(batch, device, non_blocking=True)
        # Forward pass
        outputs = model(batch)
        logits = outputs["logits"]  # [B, T, V]
        targets = batch["labels"]     # [B, T], -100 where nothing is scored

        # Flatten for loss calculation
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        # Cross-entropy with ignore_index automatically masks padding
        if "token_loss" in outputs:
            token_loss = outputs["token_loss"]
        else:
            token_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100, reduction='mean')

        # Calculate accuracy (only on non-padded tokens)
        predictions = logits_flat.argmax(dim=-1)
        non_pad_mask = (targets_flat != -100)
        correct = (predictions[non_pad_mask] == targets_flat[non_pad_mask]).sum().item()
        total_correct += correct
        num_tokens = non_pad_mask.sum().item()
        total_tokens += num_tokens
        total_batches += 1
        total_token_loss_sum += token_loss.item() * num_tokens
        if not is_ce:  # genuine token cross-entropy for a comparable perplexity
            total_ce_sum += model.token_loss.token_cross_entropy(logits_flat, targets_flat).item()

        # Rank-based metrics (comparable across loss_mode arms: only the ordering matters)
        lv, tv = logits_flat[non_pad_mask], targets_flat[non_pad_mask]
        ranks = 1 + (lv > lv.gather(1, tv.unsqueeze(1))).sum(dim=-1)   # rank of the true next token
        top5_sum += (ranks <= 5).sum().item()
        top10_sum += (ranks <= 10).sum().item()
        mrr_sum += (1.0 / ranks.float()).sum().item()

        # t2e loss (if enabled and available)
        t2e_loss = None
        if t2e_enabled and "t2e_loss" in outputs:
            t2e_loss = outputs["t2e_loss"]
            t2e_available = True
            batch_t2e_count = int(outputs.get("t2e_count", num_tokens))
            total_t2e_loss_sum += t2e_loss.item() * batch_t2e_count
            total_t2e_count += batch_t2e_count

        # Combined loss (token + optional t2e)
        combined_loss = token_loss if t2e_loss is None else token_loss + t2e_weight * t2e_loss
        total_loss_batch_sum += combined_loss.item()

        # T2E MAE (if enabled)
        if t2e_enabled and "predicted_time_delta" in outputs and "time_deltas" in batch:
            pred_times = outputs["predicted_time_delta"]  # [B, T]
            time_deltas = batch["time_deltas"]  # [B, T]
            valid_mask = (time_deltas >= 0) & (targets != -100)
            if valid_mask.any():
                t2e_mae_sum += torch.abs(pred_times[valid_mask] - time_deltas[valid_mask]).sum().item()
                t2e_mae_count += valid_mask.sum().item()

        # Update progress bar with current metrics
        if show_progress:
            current_token_loss = total_token_loss_sum / max(total_tokens, 1)
            current_t2e_loss = total_t2e_loss_sum / max(total_t2e_count, 1)
            current_loss = current_token_loss + (t2e_weight * current_t2e_loss
                                                  if t2e_available else 0.0)
            current_acc = total_correct / max(total_tokens, 1)
            iterator.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}",
                "ppl": f"{np.exp((total_token_loss_sum if is_ce else total_ce_sum) / max(total_tokens, 1)):.2f}"
            })

    # Final metrics
    avg_token_loss = total_token_loss_sum / max(total_tokens, 1)
    avg_t2e_loss = total_t2e_loss_sum / max(total_t2e_count, 1) if t2e_available else 0.0
    avg_loss = avg_token_loss + t2e_weight * avg_t2e_loss if t2e_available else avg_token_loss
    avg_loss_batch = total_loss_batch_sum / max(total_batches, 1)
    accuracy = total_correct / max(total_tokens, 1)
    # CE objectives: exp(avg token loss). Others: exp(avg CE from the predictive
    # distribution), so perplexity stays comparable across objectives.
    perplexity = np.exp(avg_token_loss if is_ce else (total_ce_sum / max(total_tokens, 1)))

    results = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "top5_accuracy": top5_sum / max(total_tokens, 1),
        "top10_accuracy": top10_sum / max(total_tokens, 1),
        "mrr": mrr_sum / max(total_tokens, 1),
        "loss_batch_avg": avg_loss_batch,
        "token_loss": avg_token_loss,
        "total_tokens": total_tokens,
        "total_batches": total_batches,
        "total_sequences": len(dataloader.dataset)
    }

    if t2e_available:
        results["t2e_loss"] = avg_t2e_loss
        results["total_t2e_count"] = total_t2e_count
        # Record the weight the reported loss was computed with, so test_metrics.json is
        # self-documenting and comparable across checkpoints trained with different weights.
        results["t2e_weight"] = t2e_weight
        if t2e_mae_count > 0:
            results["t2e_mae"] = t2e_mae_sum / t2e_mae_count

    return results


def plot_evaluation_metrics(metrics: dict[str, float], out_png: str) -> None:
    """Plot evaluation metrics as bar chart."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Loss
    loss = metrics['loss']
    bars = ax[0].bar(['Loss'], [loss], color='steelblue')
    bar = bars[0]
    ax[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{loss:.4f}",
        ha="center",
        va="bottom"
    )
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss')
    ax[0].grid(True, alpha=0.3)

    # Perplexity
    perplexity = metrics['perplexity']
    bars = ax[1].bar(['Perplexity'], [perplexity], color='coral')
    bar = bars[0]
    ax[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{perplexity:.2f}",
        ha="center",
        va="bottom"
    )
    ax[1].set_ylabel('Perplexity')
    ax[1].set_title('Perplexity')
    ax[1].grid(True, alpha=0.3)

    # Accuracy
    accuracy = metrics['accuracy']
    bars = ax[2].bar(['Accuracy'], [accuracy], color='seagreen')
    bar = bars[0]
    ax[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{accuracy:.4f}",
        ha="center",
        va="bottom"
    )
    ax[2].set_ylabel('Accuracy')
    ax[2].set_title('Token Prediction Accuracy')
    ax[2].set_ylim([0, 1])
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evaluation plot saved to {out_png}")


def print_metrics(metrics: dict[str, float]) -> None:
    """Print the headline next-token test metrics (pipeline 02)."""
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"Loss:         {metrics['loss']:.4f}")
    print(f"Perplexity:   {metrics['perplexity']:.2f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Total Tokens: {metrics['total_tokens']:,}")
    print(f"Total Batches: {metrics['total_batches']:,}")
    print("="*60 + "\n")
