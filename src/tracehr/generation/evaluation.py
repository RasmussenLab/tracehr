# src/tracehr/generation/evaluation.py
"""Next-token test metrics of a trained model on held-out windows (pipeline 02): the
metrics, their figure and their console summary, and their breakdown by target
(TargetSplitMetrics)."""

from pathlib import Path

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
    target_splits: "TargetSplitMetrics | None" = None,
) -> dict[str, float]:
    """Next-token metrics of model over dataloader, token-weighted over the scored positions
    (padding and the front block excluded): the loss (token loss, plus the weighted T2E
    loss with T2E), perplexity, accuracy, top-5/top-10 accuracy, mean reciprocal rank and,
    with T2E, the waiting-time loss and MAE. Perplexity is exp(token loss) for a
    cross-entropy objective, else exp of the cross-entropy of the predictive distribution,
    so it stays comparable across objectives. With target_splits, every batch also feeds
    that breakdown by target (read it with target_splits.result()); the metrics returned
    here are the same either way."""

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
        if target_splits is not None:   # the breakdown by target (TargetSplitMetrics below)
            target_splits.update(batch, lv, tv, ranks, outputs)
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


# ============================================================================ #
# Breakdown by target: new or repeat, same time or later, with reference predictors
# ============================================================================ #
# The groups a scored target falls into. new + repeat + unk and same_time + later + unk each
# add up to all, and the four combinations add up to all - unk. <UNK> lumps different rare
# codes together, so its "repeats" mean nothing and it has a group of its own.
TARGET_GROUPS = ("all", "new", "repeat", "same_time", "later", "new_same_time", "new_later",
                 "repeat_same_time", "repeat_later", "unk")
# The per-target values summed for every group, in this order (TargetSplitMetrics.sums)
_SUMMED = ("tokens", "model_ce", "model_top1", "model_top5", "model_top10", "model_rr",
           "copy_top1", "copy_top5", "copy_top10", "copy_rr",
           "t2e_abs_error", "t2e_count")
# The unigram's per-target values, summed for every group once the unigram is fitted (_unigram_sums)
_UNIGRAM = ("unigram_ce", "unigram_top1", "unigram_top5", "unigram_top10", "unigram_rr")
PNG_DPI = 400   # resolution of the publication figures, as in token_analysis.py


def never_target_ids(stats: dict) -> list[int]:
    """Token ids that are never a next-token target: the special tokens except <UNK>, and the
    sex and static tokens (the readers move every one of them into the front block)."""
    specials = {int(i) for name, i in stats["special_tokens"].items() if name != "<UNK>"}
    front = set(stats.get("sex_token_ids", [])) | set(stats.get("static_token_ids", []))
    return sorted(specials | {int(i) for i in front})


def _unigram_sums(target_counts: np.ndarray) -> np.ndarray:
    """[groups, len(_UNIGRAM)] sums of the unigram's per-target values for every group, from
    target_counts [groups, vocabulary], every group's count of each target id (row 0: all).

    The unigram is fitted on all scored test targets: it ranks the codes by their count, the
    lower id first among equal counts (so its top guess breaks ties like the model's argmax),
    and gives each code its share of the targets. No fixed ranking has a higher top-k accuracy,
    and no fixed distribution a lower cross-entropy, over all test targets: it is the best a
    predictor that ignores the history can do there. Every scored target has a count, so every
    loss is finite."""
    total = target_counts[0]
    ids = np.arange(len(total))
    rank = np.empty(len(total))
    rank[np.lexsort((ids, -total))] = ids + 1        # the most frequent first, then the lower id
    seen = total > 0
    log_prob = np.zeros(len(total))                  # an id never scored adds nothing to any group
    log_prob[seen] = np.log(total[seen] / total.sum())
    per_target = np.stack([-log_prob, rank == 1, rank <= 5, rank <= 10, 1.0 / rank], axis=1)
    return target_counts @ per_target


def _context_copy(input_ids: torch.Tensor, labels: torch.Tensor,
                  events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(in_context, copy_rank), both [B,T], for the target labels[t] of every position t:
    whether its code is held by one of the window's events at or before t, and its rank under
    the context-copy predictor. That predictor ranks the codes of those events by their latest
    position, a strict order since a position holds one code, so a repeated target ranks 1 +
    the number of distinct codes seen after its own latest occurrence. A target missing from
    the context has rank 0, a miss. Pairwise over positions: [B,T,T] booleans, the size of
    one attention map."""
    T = labels.size(1)
    pos = torch.arange(T, device=labels.device)
    row, col = pos.view(1, T, 1), pos.view(1, 1, T)
    seen = events.unsqueeze(1) & (col <= row)                         # [b,t,s]: event s is in t's context
    match = seen & (input_ids.unsqueeze(1) == labels.unsqueeze(2))    # ... and holds t's target
    in_context = match.any(dim=2)
    latest = torch.where(match, col, -1).amax(dim=2)                  # the target's latest position
    # nxt[b,s]: the next event after s with the same code (T when none). s is its code's
    # latest occurrence up to t exactly when nxt[b,s] > t.
    same_later = events.unsqueeze(1) & (input_ids.unsqueeze(2) == input_ids.unsqueeze(1)) & (col > row)
    nxt = torch.where(same_later, col, T).amin(dim=2)
    newer = seen & (col > latest.unsqueeze(2)) & (nxt.unsqueeze(1) > row)
    return in_context, torch.where(in_context, 1 + newer.sum(dim=2), 0)


def _same_time(time_age: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """[B,T] whether the target labels[t], the event at t+1, has the time of the latest event
    at or before t. False when the window holds no earlier event: a window's first event
    counts as later, since its predecessor sits in the previous window."""
    pos = torch.arange(time_age.size(1), device=time_age.device).expand_as(events)
    prev = torch.where(events, pos, -1).cummax(dim=1).values
    target_time = time_age.roll(-1, dims=1)   # the time at t+1 (the wrapped last column is never scored)
    return (prev >= 0) & (target_time == time_age.gather(1, prev.clamp(min=0)))


def _target_groups(labels: torch.Tensor, in_context: torch.Tensor, same_time: torch.Tensor,
                   unk_id: int) -> dict[str, torch.Tensor]:
    """[B,T] mask of every TARGET_GROUPS group (False where nothing is scored)."""
    scored = labels != -100
    known = scored & (labels != unk_id)
    new, repeat = known & ~in_context, known & in_context
    same, later = known & same_time, known & ~same_time
    return {"all": scored, "new": new, "repeat": repeat, "same_time": same, "later": later,
            "new_same_time": new & same, "new_later": new & later,
            "repeat_same_time": repeat & same, "repeat_later": repeat & later,
            "unk": scored & ~known}


def _group_result(s: dict, n_all: float, has_t2e: bool) -> dict:
    """One group's block of by_target from its sums s (keyed by _SUMMED and _UNIGRAM)."""
    n = s["tokens"]

    def mean(key):
        return s[key] / n if n else float("nan")

    def ranked(prefix):
        return {"accuracy": mean(f"{prefix}_top1"), "top5_accuracy": mean(f"{prefix}_top5"),
                "top10_accuracy": mean(f"{prefix}_top10"), "mrr": mean(f"{prefix}_rr")}

    group = {"tokens": int(n), "share": n / n_all if n_all else float("nan"),
             "model": {"cross_entropy": mean("model_ce"),
                       "perplexity": float(np.exp(mean("model_ce"))), **ranked("model")},
             "context_copy": ranked("copy"),
             "unigram": {"cross_entropy": mean("unigram_ce"),
                         "perplexity": float(np.exp(mean("unigram_ce"))), **ranked("unigram")}}
    if has_t2e:
        group["model"]["t2e_mae"] = (s["t2e_abs_error"] / s["t2e_count"]
                                     if s["t2e_count"] else float("nan"))
    return group


class TargetSplitMetrics:
    """Pipeline 02's test metrics broken down by target (TARGET_GROUPS), next to two reference
    predictors scored on the same targets: the context copy (the window's earlier codes,
    latest first) and the unigram (the codes most frequent among all scored test targets,
    _unigram_sums). Both use the test windows only. evaluate_model feeds it every batch;
    result() returns the by_target block of test_metrics.json.

    A target is a repeat when one of the window's earlier events holds its code, and
    same_time when it has the time of the latest earlier event. The events are the attended
    positions whose code is not one of never_target_ids (the ids the evaluation never scores),
    so no context holds a start, sex or padding token.
    """

    def __init__(self, unk_id: int, never_target_ids, vocab_size: int):
        self.unk_id = int(unk_id)
        self.never_target_ids = torch.as_tensor(sorted(never_target_ids), dtype=torch.int64)
        self.sums = np.zeros((len(TARGET_GROUPS), len(_SUMMED)))
        self.target_counts = torch.zeros(len(TARGET_GROUPS), vocab_size, dtype=torch.float64)
        self.has_t2e = False

    @torch.no_grad()
    def update(self, batch: dict, lv: torch.Tensor, tv: torch.Tensor, ranks: torch.Tensor,
               outputs: dict) -> None:
        """Add one batch. lv, tv and ranks are evaluate_model's logits, targets and model ranks
        of the scored positions, in the row-major order of batch["labels"]."""
        if "time_age" not in batch:
            raise KeyError("The breakdown by target needs 'time_age' in every batch, to tell "
                           "same-time targets from later ones; this batch has none.")
        labels, input_ids = batch["labels"], batch["input_ids"]
        never = self.never_target_ids.to(input_ids.device)
        # Fail fast: a scored never-target is kept out of every context, so the context copy
        # could never predict it
        if torch.isin(tv, never).any():
            raise ValueError("A scored target is one of never_target_ids; they must be ids the "
                             "evaluation never scores.")
        events = batch["attention_mask"].bool() & ~torch.isin(input_ids, never)
        in_context, copy_rank = _context_copy(input_ids, labels, events)
        groups = _target_groups(labels, in_context, _same_time(batch["time_age"], events), self.unk_id)
        scored = labels != -100

        copy_rank = copy_rank[scored]
        copy_hit = copy_rank > 0
        columns = {
            "tokens": torch.ones_like(tv),
            "model_ce": F.cross_entropy(lv.float(), tv, reduction="none"),
            "model_top1": lv.argmax(dim=-1) == tv,
            "model_top5": ranks <= 5,
            "model_top10": ranks <= 10,
            "model_rr": 1.0 / ranks,
            "copy_top1": copy_rank == 1,
            "copy_top5": copy_hit & (copy_rank <= 5),
            "copy_top10": copy_hit & (copy_rank <= 10),
            "copy_rr": torch.where(copy_hit, 1.0 / copy_rank.clamp(min=1), 0.0),
        }
        if "predicted_time_delta" in outputs and "time_deltas" in batch:   # same rule as the MAE above
            self.has_t2e = True
            deltas = batch["time_deltas"][scored]
            valid = deltas >= 0
            error = (outputs["predicted_time_delta"][scored] - deltas).abs()
            columns.update(t2e_abs_error=torch.where(valid, error, 0.0), t2e_count=valid)

        values = torch.zeros(len(tv), len(_SUMMED), dtype=torch.float64, device=tv.device)
        for j, name in enumerate(_SUMMED):
            if name in columns:
                values[:, j] = columns[name].double()
        masks = torch.stack([groups[g][scored] for g in TARGET_GROUPS]).double()   # [groups, targets]
        self.sums += (masks @ values).cpu().numpy()
        self.target_counts.index_add_(1, tv.cpu(), masks.cpu())   # the unigram is fitted in result()

    def result(self) -> dict:
        """The by_target block: for every group its tokens and share of all scored targets, and
        the cross-entropy, perplexity, accuracy, top-5/top-10 accuracy and MRR of the model, the
        context copy (no probabilities, so only the ranked metrics) and the unigram; plus the
        model's T2E MAE with T2E. An empty group's metrics are NaN (written as null)."""
        n_all = self.sums[0, 0]
        sums = np.hstack([self.sums, _unigram_sums(self.target_counts.numpy())])
        return {g: _group_result(dict(zip(_SUMMED + _UNIGRAM, row)), n_all, self.has_t2e)
                for g, row in zip(TARGET_GROUPS, sums)}


def print_target_splits(by_target: dict) -> None:
    """Print the by_target block: every group's share of the targets, then accuracy, top-5
    accuracy and MRR (and perplexity) of the model and of each reference predictor."""
    predictors = ["model", "context_copy", "unigram"]
    print("=" * 60)
    print("TEST METRICS BY TARGET (accuracy / top-5 / MRR, perplexity in brackets)")
    print("=" * 60)
    print(f"{'Group':<18}{'Share':>7}" + "".join(f"  {p:<28}" for p in predictors))
    for g, block in by_target.items():
        cells = []
        for p in predictors:
            m = block[p]
            cell = f"{m['accuracy']:.3f} {m['top5_accuracy']:.3f} {m['mrr']:.3f}"
            if "perplexity" in m:
                cell += f" ({m['perplexity']:.2f})"
            cells.append(f"  {cell:<28}")
        print(f"{g:<18}{block['share']:>7.1%}" + "".join(cells))
    print("=" * 60 + "\n")


def plot_target_splits(by_target: dict, out_stem) -> None:
    """The figure of the four target cells, saved as out_stem.png (PNG_DPI) and out_stem.svg:
    one panel per cell (new or repeat, same time or later) with the accuracy, top-5 and top-10
    accuracy and MRR of the model, the context copy and the unigram, and the cell's share of
    the targets in its title."""
    cells = [("new_same_time", "New, same time"), ("new_later", "New, later"),
             ("repeat_same_time", "Repeat, same time"), ("repeat_later", "Repeat, later")]
    predictors = [("model", "Model"), ("context_copy", "Context copy"), ("unigram", "Unigram")]
    metrics =[("accuracy", "Accuracy"), ("top5_accuracy", "Top-5"),
               ("top10_accuracy", "Top-10"), ("mrr", "MRR")]
    x = np.arange(len(metrics))
    width = 0.8 / len(predictors)
    out_stem = Path(out_stem)
    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharey=True)
        for ax, (key, title) in zip(axes.flat, cells):
            cell = by_target[key]
            for i, (predictor, label) in enumerate(predictors):
                offset = (i - (len(predictors) - 1) / 2) * width
                ax.bar(x + offset, [cell[predictor][m] for m, _ in metrics], width,
                       color=f"C{i}", label=label)
            ax.set_xticks(x, [name for _, name in metrics])
            ax.set_ylim(0, 1)
            ax.set_title(f"{title} ({cell['share']:.1%} of targets)", loc="left")
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        for suffix, kwargs in ((".png", {"dpi": PNG_DPI}), (".svg", {})):
            fig.savefig(out_stem.parent / f"{out_stem.name}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"Breakdown by target saved to {out_stem}.png and {out_stem}.svg")
