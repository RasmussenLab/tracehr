# src/tracehr/generation/explainability.py
"""Explainability for pipeline 03's generation: every generated token is attributed to the
events of the exact window that produced it (explain_generated, called from
generate_tokens) by the two signed methods of attribution.py. Also the explainability
config block and the figure of each explained step.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .attribution import NEGATIVE_COLOR, POSITIVE_COLOR, integrated_gradients, occlusion

EXPLAIN_METHODS = ("integrated_gradients", "occlusion")
# The keys of the config block and the value a missing key takes (off unless enabled)
EXPLAIN_DEFAULTS = {"enabled": False, "methods": list(EXPLAIN_METHODS), "n_steps": 50,
                    "top_k_tokens": 10, "plot_first_k": 0}


def resolve_explainability(block) -> dict | None:
    """config['explainability'] -> None (off) or the validated settings.

    Off when the block is absent or `enabled` is not true. The block is validated either
    way, so a typo fails now rather than after the switch is turned on for a long run; a
    missing key takes its EXPLAIN_DEFAULTS value.
    """
    if block is None:
        return None
    if not isinstance(block, dict):
        raise TypeError(f"explainability must be a block of settings, got {block!r}.")
    unknown = sorted(set(block) - set(EXPLAIN_DEFAULTS))
    if unknown:
        raise ValueError(f"explainability: unknown key(s) {unknown}; the keys are "
                         f"{list(EXPLAIN_DEFAULTS)}.")
    cfg = {**EXPLAIN_DEFAULTS, **block}

    if not isinstance(cfg["enabled"], bool):
        raise TypeError(f"explainability.enabled must be true or false, got {cfg['enabled']!r}.")
    methods = cfg["methods"]
    if not isinstance(methods, list) or not methods:
        raise TypeError(f"explainability.methods must be a non-empty list such as "
                        f"{list(EXPLAIN_METHODS)}, got {methods!r}.")
    unknown = [m for m in methods if m not in EXPLAIN_METHODS]
    if unknown:
        raise ValueError(f"explainability.methods: unknown method(s) {unknown}; choose from "
                         f"{list(EXPLAIN_METHODS)}.")
    if len(set(methods)) != len(methods):
        raise ValueError(f"explainability.methods lists a method more than once: {methods}.")
    # bool is an int subclass in Python, so true/false must be refused explicitly
    for key, low in (("n_steps", 1), ("top_k_tokens", 1), ("plot_first_k", 0)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"explainability.{key} must be a whole number >= {low}, got {value!r}.")
    if not cfg["enabled"]:
        return None
    return {"methods": list(methods), "n_steps": cfg["n_steps"],
            "top_k_tokens": cfg["top_k_tokens"], "plot_first_k": cfg["plot_first_k"]}


def explain_generated(model, batch: dict[str, torch.Tensor], last: torch.Tensor,
                      picks: np.ndarray, itos: list[str], explain: dict,
                      batch_size: int) -> list[dict]:
    """Attribute each generated token to the window that produced it; one dict per row.

    batch, last and picks are generate_tokens' own forward input, last real positions and
    selected tokens, so the explained target is by construction the generated token
    (after the structural / allowed-token mask). Each window is first trimmed to its real
    tokens: windows are right-padded and attention is causal, so no pad ever reaches a
    real position and the trim is exact while saving the padded compute. Per method the
    dict holds attr_<method> (one signed score per context_tokens entry) and top_<method>
    (the top_k_tokens highest-scoring events as {position, token, score}; ties go to the
    earlier position).
    """
    rows = []
    for j, (end, target) in enumerate(zip(last.tolist(), picks.tolist())):
        sample = {k: v[j:j + 1, :end + 1] for k, v in batch.items()}
        tokens = [itos[t] for t in sample["input_ids"][0].tolist()]
        row = {}
        for method in explain["methods"]:
            scores = (integrated_gradients(model, sample, target, explain["n_steps"], batch_size)
                      if method == "integrated_gradients"
                      else occlusion(model, sample, target, batch_size))
            top = np.argsort(-scores, kind="stable")[:explain["top_k_tokens"]]
            row[f"attr_{method}"] = scores.tolist()
            row[f"top_{method}"] = [{"position": int(i), "token": tokens[i], "score": float(scores[i])}
                                    for i in top]
        rows.append(row)
    return rows


def _explanation_figure(row: pd.Series, methods: list[str]):
    """One generated prediction: a signed bar panel per method over its context events."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tokens = list(row["context_tokens"])
    # The window ends with the step - 1 tokens generated at earlier steps (never the start token)
    n_generated = min(int(row["step"]) - 1, len(tokens) - 1)
    labels = [tok + "*" if i >= len(tokens) - n_generated else tok for i, tok in enumerate(tokens)]
    x = np.arange(len(tokens))

    width = min(max(8.0, 0.18 * len(tokens)), 40.0)
    fig, axes = plt.subplots(len(methods), 1, sharex=True, squeeze=False,
                             figsize=(width, 1.2 + 2.4 * len(methods)))
    for ax, method in zip(axes[:, 0], methods):
        scores = np.asarray(row[f"attr_{method}"], dtype=float)
        ax.bar(x, scores, width=0.8, color=[POSITIVE_COLOR if s >= 0 else NEGATIVE_COLOR for s in scores])
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_ylabel(f"{method.replace('_', ' ')}\n(change in log p)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[-1, 0].set_xlim(-0.6, len(tokens) - 0.4)

    title = (f"Patient {row['patient_id']}, step {row['step']}: generated {row['predicted_token']} "
             f"(p={row['probs'][row['predicted_token']]:.3f}")
    title += f", abstain={row['abstain_prob']:.3f})" if "abstain_prob" in row else ")"
    if n_generated:
        title += "\n* generated at an earlier step"
    axes[0, 0].set_title(title, fontsize=10, loc="left")
    axes[0, 0].legend(handles=[Patch(color=POSITIVE_COLOR, label="raises p(generated token)"),
                               Patch(color=NEGATIVE_COLOR, label="lowers p(generated token)")],
                      loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_generated_explanations(gen_df: pd.DataFrame, methods: list[str], plot_first_k: int,
                                out_dir) -> int:
    """Save one PNG per generated step of the first plot_first_k prompts (generation order)
    as out_dir/patient_<id>_anchor_<age>_step_<n>.png; returns how many were written. The
    anchor is in the name because a patient with several anchors has one prompt each."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = gen_df.groupby(["patient_id", "anchor_age"], sort=False, dropna=False).ngroup()
    rows = gen_df[prompt < plot_first_k]
    for _, row in rows.iterrows():
        fig = _explanation_figure(row, methods)
        pid = re.sub(r"[^\w.-]", "_", str(row["patient_id"]))                 # file-name safe
        fig.savefig(out_dir / f"patient_{pid}_anchor_{row['anchor_age']}_step_{row['step']}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    return len(rows)
