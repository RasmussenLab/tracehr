# src/tracehr/finetuning/outcome_explain.py
"""Explainability for NTPOutcomeModel (regression | binary | multiclass | survival).
Three complementary views, all attributing the head's raw scalar (binary: event logit;
multiclass: the predicted class's logit; regression: the predicted value; survival: the log
relative risk) against an all-<PAD> token baseline with the event times and types kept, so
only token identity is attributed:
  1. Integrated Gradients on the token input embeddings (stage 03's integrated_gradients
     applied to the head's scalar) -> per-token, per-sample signed attributions
     (token_attributions.parquet, joined to predictions.parquet on patient_id + anchor_age;
     completeness_residual = a sample's attributions minus its output change, the n_steps
     discretisation error) and a signed bar plot per sample.
  2. Cohort summary: signed attributions pooled by token across the explained
     samples (per explained class for multiclass) -> SHAP-style beeswarm
     (global_token_importance.png/.json; multiclass: global_token_importance_class{k}.png/.json).
     Figure and json hold the same tokens: those seen in at least min_patients distinct patients.
  3. Final-layer attention heatmaps over each explained sample's most recent events
     (complementary diagnostic).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..generation.attribution import NEGATIVE_COLOR, POSITIVE_COLOR, integrated_gradients
from ..training.runtime import batch_to_device, resolve_device

# ============================================================================ #
# Plots
# ============================================================================ #
# What a positive / negative signed attribution pushes the explained output toward.
# Multiclass is per explained class (_direction).
_DIRECTION = {
    "binary": ("event", "no event"),
    "regression": ("higher prediction", "lower prediction"),
    "survival": ("higher risk", "lower risk"),
}


def _direction(task: str, target: int | None) -> tuple[str, str]:
    """(positive, negative) labels of one explanation; target is the explained class
    (multiclass), else None."""
    return _DIRECTION[task] if target is None else (f"class {target}", f"away from class {target}")


def plot_token_importance(
    token_ids: np.ndarray,
    attributions: np.ndarray,
    vocab: dict[int, str],
    title: str,
    top_k: int = 20,
    save_path: str | None = None,
    direction: tuple[str, str] = ("event", "no event"),
):
    """Bar plot of one sample's top_k events by |attribution|, drawn signed (direction names
    what the positive and negative sides push toward), saved to save_path when given. Each
    label carries the event's position (as in token_attributions.parquet), so a code
    recorded several times stays apart."""
    import matplotlib.pyplot as plt

    order = np.argsort(-np.abs(attributions), kind="stable")[:top_k]
    names = [f"{vocab.get(int(token_ids[i]), f'tok_{token_ids[i]}')} [{i}]" for i in order]
    scores = attributions[order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(order) * 0.4)))
    ax.barh(range(len(order)), scores, color=[POSITIVE_COLOR if s >= 0 else NEGATIVE_COLOR for s in scores])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
    pos, neg = direction
    ax.set_xlabel(f"Signed Importance  (positive → {pos}, negative → {neg})", fontsize=11)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_token_importance(
    token_importances: dict[str, list[float]],
    horizon_name: str,
    top_k: int = 20,
    save_path: str | None = None,
    max_points_per_token: int = 300,
    direction: tuple[str, str] = ("event", "no event"),
):
    """SHAP-style beeswarm plot showing global signed token importance.

    Each dot represents one occurrence of a token across the explained samples.
    Position on the x-axis shows signed importance (positive = pushes the
    explained output toward direction[0], negative = toward direction[1]).
    Tokens are ranked on the y-axis by mean |importance|.

    Args:
        token_importances: {token_name: [signed_importance_values]}, only the tokens that may
            be shown (the caller applies the patient threshold)
        horizon_name: Label for the prediction horizon (used in title)
        top_k: Number of top tokens to display
        save_path: Where to save the plot
        max_points_per_token: Sub-sample per row to avoid over-plotting
        direction: (positive, negative) labels for the x-axis and legend
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    filtered = {k: np.array(v) for k, v in token_importances.items()}
    if not filtered:
        print(f"  No tokens to plot for {horizon_name}")
        return

    # Rank by mean |importance|
    mean_abs = {k: np.mean(np.abs(v)) for k, v in filtered.items()}
    sorted_tokens = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    # Reverse so most important is at the top of the plot
    token_names = [t[0] for t in sorted_tokens][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(token_names) * 0.4)))

    rng = np.random.RandomState(42)

    for row_idx, name in enumerate(token_names):
        values = filtered[name]
        if len(values) > max_points_per_token:
            indices = rng.choice(len(values), max_points_per_token, replace=False)
            values = values[indices]

        y_jitter = rng.uniform(-0.3, 0.3, size=len(values))
        y_positions = row_idx + y_jitter

        colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]
        ax.scatter(values, y_positions, c=colors, alpha=0.35, s=8, edgecolors='none')

    ax.set_yticks(range(len(token_names)))
    ax.set_yticklabels(token_names, fontsize=9)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    pos, neg = direction
    ax.set_xlabel(f"Signed Importance  (positive → {pos}, negative → {neg})", fontsize=11)
    ax.set_title(f"Global Token Importance — {horizon_name}", fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=POSITIVE_COLOR,
               markersize=8, label=f'Positive (→ {pos})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NEGATIVE_COLOR,
               markersize=8, label=f'Negative (→ {neg})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_heatmap(
    attention: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int = -1,
    head: int | None = None,
    vocab: dict[int, str] | None = None,
    token_ids: torch.Tensor | None = None,
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 10),
    max_tokens: int = 50,
):
    """Heatmap of one layer's attention ([H, T, T]: head `head`, or the mean over the heads;
    [T, T] as is) over the last max_tokens real positions, labeled with the token names when
    vocab and token_ids are given (else the positions), saved to save_path when given.
    layer only labels the title."""
    import matplotlib.pyplot as plt

    valid = attention_mask.bool().cpu()
    n_valid = valid.sum().item()
    n_show = min(n_valid, max_tokens)
    # The last n_show real positions (windows are right-padded): the events nearest the
    # prediction point, not the oldest ones at the start of the window
    shown = slice(n_valid - n_show, n_valid)

    if attention.dim() == 3:
        # [H, T, T] – pick a head or average
        if head is not None:
            attn = attention[head, shown, shown].cpu().numpy()
            title = title or f"Layer {layer}, Head {head}"
        else:
            attn = attention.mean(dim=0)[shown, shown].cpu().numpy()
            title = title or f"Layer {layer}, Average over heads"
    else:
        attn = attention[shown, shown].cpu().numpy()
        title = title or "Attention map"

    # Token labels
    if token_ids is not None and vocab is not None:
        ids = token_ids[shown].cpu().numpy()
        labels = [vocab.get(int(i), f"{i}") for i in ids]
    else:
        labels = [str(i) for i in range(shown.start, shown.stop)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n_show))
    ax.set_yticks(range(n_show))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================ #
# Attribution, attention and the patients to explain
# ============================================================================ #
def integrated_gradients_signed(model, batch: dict[str, torch.Tensor], n_steps: int = 50,
                                target_index: int | None = None, batch_size: int = 64) -> np.ndarray:
    """Signed Integrated Gradients per input token for a single-sample batch.

    Stage 03's integrated_gradients (all-<PAD> token baseline, midpoint rule, path points
    run batch_size at a time, gradients w.r.t. the path embeddings only) attributing the
    head's scalar: interpolated embeddings go through the backbone (its forward with
    input_embeds), pooling, and the task head. Multiclass attributes the logit of
    `target_index`, the predicted class when not given. The window is first trimmed to its
    real tokens, exact as in stage 03: it is right-padded, attention is causal and pooling
    reads only real positions. Returns a length-`seq_len` signed attribution.
    """
    model.eval()
    device = next(model.parameters()).device
    seq_len = int(batch["attention_mask"][0].sum().item())
    sample = {k: v[:1, :seq_len].to(device) for k, v in batch.items() if k != "labels"}

    if model.task == "multiclass" and target_index is None:
        with torch.no_grad():
            target_index = int(model(sample)["logits"].argmax(dim=-1)[0].item())

    def head_output(backbone, rows, embeds, k):
        """Head output k of every row of path embeddings -> [N]; the backbone runs to its
        hidden states only."""
        hidden = backbone(rows, hidden_only=True, input_embeds=embeds)["hidden_states"]
        return model.head(model._pool_hidden_states(hidden, rows["attention_mask"]))[:, k]

    k = 0 if target_index is None else target_index        # one-output heads explain output 0
    return integrated_gradients(model.backbone, sample, k, n_steps, batch_size, score=head_output)


@torch.no_grad()
def _capture_attentions(model, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Per-layer causal attention weights [B, H, T, T] of one forward, from the backbone's own
    attention capture; the backbone's previous capture setting is restored afterwards."""
    backbone = model.backbone
    was_capturing = backbone.store_attn
    backbone.enable_attention_capture(True)
    try:
        model.eval()
        model(batch)
        return backbone.get_attention_maps()
    finally:
        backbone.enable_attention_capture(was_capturing)


@torch.no_grad()
def _collect_predictions(model, loader, device, task) -> dict[str, np.ndarray]:
    """One batched pass: per-sample prediction, label, and a confidence score
    used to pick the most decisive samples to explain."""
    model.eval()
    preds, labels, conf = [], [], []
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(batch)
        if task in ("survival", "regression"):
            # A risk score or a predicted value has no decision threshold, so "decisive"
            # means far from the middle of the cohort's predictions: clearly high or clearly
            # low. Centred below on the median of the whole cohort (robust to a long tail),
            # once every batch is in; a batch's median would rank samples against whatever
            # shares their batch.
            preds.append(out["predictions"].cpu().numpy())
        elif task == "binary":
            p = out["probs"].cpu().numpy()
            preds.append(p)
            conf.append(np.abs(p - 0.5))
        else:  # multiclass
            p = out["probs"].cpu().numpy()
            preds.append(p.argmax(axis=-1))
            conf.append(p.max(axis=-1))
        labels.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(preds)
    if task in ("survival", "regression"):
        confidence = np.abs(preds - np.median(preds))
    else:
        confidence = np.concatenate(conf)
    return {"preds": preds, "labels": np.concatenate(labels), "confidence": confidence}


def _explain_sample(model, batch: dict[str, torch.Tensor], idx: int, rank: int, *,
                    target: int | None, pid: str, anchor: float, task: str,
                    vocab_itos: dict[int, str], n_steps: int, batch_size: int, top_k_tokens: int,
                    plot_attention: bool, expl_dir: Path) -> list[dict]:
    """The token-attribution rows of one explained sample (one per position), with its signed
    bar plot and, with plot_attention, its final-layer attention heatmap. target is the class
    IG explains (multiclass), else None."""
    signed = integrated_gradients_signed(model, batch, n_steps=n_steps, target_index=target,
                                         batch_size=batch_size)
    # Completeness: the attributions sum to the explained output's change from the all-<PAD>
    # baseline (times and types kept), up to the n_steps discretisation error recorded here
    pad_batch = {**batch, "input_ids": torch.full_like(batch["input_ids"], model.backbone.pad_id)}
    k = target if target is not None else 0               # the head output IG explains
    with torch.no_grad():
        change = model.head(model(batch)["pooled"])[0, k] - model.head(model(pad_batch)["pooled"])[0, k]
    residual = float(signed.sum()) - change.item()
    seq_len = len(signed)
    ids = batch["input_ids"][0, :seq_len].cpu().numpy()
    ages = batch["time_age"][0, :seq_len].cpu().numpy() if "time_age" in batch else np.full(seq_len, np.nan)

    rows = []
    for pos in range(seq_len):
        row = {
            "patient_id": pid,
            "sample_index": idx,
            "position": pos,
            "token": vocab_itos.get(int(ids[pos]), f"tok_{ids[pos]}"),
            "age": float(ages[pos]),
            "attribution": float(signed[pos]),
            "abs_attribution": float(abs(signed[pos])),
            "completeness_residual": residual,
        }
        if target is not None:
            row["target_class"] = target
        row["anchor_age"] = anchor  # last, so every earlier column keeps its position
        rows.append(row)

    plot_token_importance(
        ids, signed, vocab_itos, title=f"Patient {pid}, anchor {anchor:g} | top token attributions",
        top_k=top_k_tokens, direction=_direction(task, target),
        save_path=str(expl_dir / f"sample_{rank+1}_{pid}_importance.png"),
    )
    if plot_attention:
        attns = _capture_attentions(model, batch)
        if attns:
            plot_attention_heatmap(
                attns[-1][0].cpu(), batch["attention_mask"][0].cpu(), layer=len(attns) - 1,
                vocab=vocab_itos, token_ids=batch["input_ids"][0].cpu(),
                title=f"Patient {pid}, anchor {anchor:g} | final-layer attention, most recent events",
                save_path=str(expl_dir / f"sample_{rank+1}_{pid}_attn.png"),
            )
    return rows


def _write_cohort_summaries(cohorts: dict, cohort_patients: dict, *, task: str, n_explain: int,
                            explained_preds: np.ndarray, min_patients: int, top_k_tokens: int,
                            expl_dir: Path) -> None:
    """The cohort summary: a signed beeswarm and a json ranked by mean |attribution|.
    Multiclass gets one summary per explained class; other tasks the single group None, which
    keeps the unsuffixed file names (and still writes them when no sample was explained).
    explained_preds holds the explained samples' predictions, in explanation order."""
    groups = sorted(cohorts) if task == "multiclass" else [None]
    for target in groups:
        # Only tokens seen in at least min_patients distinct patients, in the figure and the json
        # alike, so both rank the same tokens and none seen in fewer patients is reported
        patients = cohort_patients[target]
        cohort = {name: v for name, v in cohorts[target].items() if len(patients[name]) >= min_patients}
        suffix = "" if target is None else f"_class{target}"
        # The summary pools only the explained samples (multiclass: those predicted this class)
        n = n_explain if target is None else int((explained_preds == target).sum())
        plot_global_token_importance(
            cohort, horizon_name=(task if target is None else f"{task}, class {target}")
                                 + f" ({n} most decisive samples)",
            top_k=top_k_tokens,
            save_path=str(expl_dir / f"global_token_importance{suffix}.png"),
            direction=_direction(task, target),
        )
        summary = {
            name: {
                "mean_abs": float(np.mean(np.abs(v))),
                "mean_signed": float(np.mean(v)),
                "n_occurrences": len(v),
                "n_patients": len(patients[name]),
            }
            for name, v in sorted(cohort.items(), key=lambda x: np.mean(np.abs(x[1])), reverse=True)[:top_k_tokens * 2]
        }
        with open(expl_dir / f"global_token_importance{suffix}.json", "w") as f:
            json.dump(summary, f, indent=2)


# ============================================================================ #
# The explainability run (pipeline 06)
# ============================================================================ #
def explain_outcome_predictions(
    model,
    dataset,
    vocab_itos: dict[int, str],
    task: str,
    output_dir: str,
    collate_fn,
    num_samples: int = 20,
    top_k_tokens: int = 25,
    n_steps: int = 50,
    batch_size: int = 64,
    device: str = "auto",
    plot_attention: bool = True,
    min_patients: int = 5,
) -> pd.DataFrame:
    """Integrated-Gradients + cohort-summary + attention explainability.

    Explains the `num_samples` most confident/extreme samples, writes
    per-token signed attributions, a SHAP-style global token-importance plot
    (one per explained class for multiclass), and per-sample signed attribution
    bars and attention heatmaps into `<output_dir>/explainability/`. The cohort
    summary (plot and json) keeps only tokens seen in at least `min_patients`
    distinct patients. Returns the token-attributions dataframe.
    """
    device = torch.device(resolve_device(device))
    model = model.to(device).eval()

    expl_dir = Path(output_dir) / "explainability"
    expl_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = getattr(dataset, "patient_ids", None)
    # Each sample's own anchor (a patient may have several): with patient_id, the key of its
    # row in predictions.parquet
    anchor_ages = getattr(dataset, "anchor_ages", None)

    # Rank the samples by prediction confidence, then explain the top ones.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    stats = _collect_predictions(model, loader, device, task)
    n_explain = min(num_samples, len(dataset))
    order = np.argsort(-stats["confidence"])[:n_explain]

    rows = []
    # token_name -> [signed attributions]; multiclass keeps one cohort per explained class
    # (attributions toward different classes share no sign), other tasks the single key None
    cohorts = defaultdict(lambda: defaultdict(list))
    cohort_patients = defaultdict(lambda: defaultdict(set))  # same keys -> {distinct patient ids}
    print(f"[explain] Integrated Gradients on {n_explain} samples (n_steps={n_steps})...")
    for rank, idx in enumerate(order):
        batch = batch_to_device(collate_fn([dataset[int(idx)]]), device)
        target = int(stats["preds"][idx]) if task == "multiclass" else None  # class IG explains
        pid = str(patient_ids[int(idx)]) if patient_ids is not None else str(int(idx))
        anchor = float(anchor_ages[int(idx)]) if anchor_ages is not None else float("nan")
        sample_rows = _explain_sample(model, batch, int(idx), rank, target=target, pid=pid,
                                      anchor=anchor, task=task, vocab_itos=vocab_itos,
                                      n_steps=n_steps, batch_size=batch_size,
                                      top_k_tokens=top_k_tokens, plot_attention=plot_attention,
                                      expl_dir=expl_dir)
        for row in sample_rows:
            cohorts[target][row["token"]].append(row["attribution"])
            cohort_patients[target][row["token"]].add(pid)
        rows.extend(sample_rows)

    attributions = pd.DataFrame(rows)
    attributions.to_parquet(expl_dir / "token_attributions.parquet", index=False)
    _write_cohort_summaries(cohorts, cohort_patients, task=task, n_explain=n_explain,
                            explained_preds=stats["preds"][order], min_patients=min_patients,
                            top_k_tokens=top_k_tokens, expl_dir=expl_dir)

    worst = max((abs(r["completeness_residual"]) for r in rows), default=0.0)
    print(f"[explain] IG completeness: largest |residual| {worst:.2e} (raise n_steps if not small)")
    print(f"[explain] Wrote token attributions + global importance to {expl_dir}")
    return attributions
