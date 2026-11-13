# src/tracehr/generation/explainability.py
"""Explainability for pipeline 03's generation: the token generated at step 1, the prediction
at the anchor from the real history, is attributed to the events of the exact window that
produced it (explain_generated, called from generate_tokens) by the two signed methods of
attribution.py, and the explained predictions are summarised over the cohort under the
5-patient rule (summarise_generated_explanations): Integrated Gradients with pipeline 06's
tables and figures, occlusion by the time of the dropped event
(time_before_prediction_occlusion). Also the explainability config block.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..finetuning.explain_summary import (
    N_BOOTSTRAP,
    SEED,
    TIME_BIN_EDGES,
    TIME_BIN_LABELS,
    _window,
    _write,
    patient_bootstrap,
    summarise_explanations,
)
from ..finetuning.finetune_plots import FIG_WIDTH, HIDDEN, MIN_PATIENTS, STYLE, hidden_groups
from .attribution import integrated_gradients, occlusion
from .model_io import json_safe

EXPLAIN_METHODS = ("integrated_gradients", "occlusion")
# The keys of the config block and the value a missing key takes (off unless enabled)
EXPLAIN_DEFAULTS = {"enabled": False, "methods": list(EXPLAIN_METHODS), "n_steps": 50,
                    "top_k_tokens": 10}
# time_before_prediction_occlusion: the last event of the context on its own, then the earlier
# events by time before the prediction; each split by whether the event has the generated token's code
LAST_EVENT = "last event"
OCCLUSION_SERIES = ("same code as the generated token", "other codes")


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
    for key, low in (("n_steps", 1), ("top_k_tokens", 1)):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise ValueError(f"explainability.{key} must be a whole number >= {low}, got {value!r}.")
    if not cfg["enabled"]:
        return None
    return {"methods": list(methods), "n_steps": cfg["n_steps"], "top_k_tokens": cfg["top_k_tokens"]}


def explain_generated(model, batch: dict[str, torch.Tensor], ages: np.ndarray, last: torch.Tensor,
                      picks: np.ndarray, itos: list[str], explain: dict,
                      batch_size: int) -> list[dict]:
    """Attribute each generated token to the window that produced it; one dict per row.

    batch, last and picks are generate_tokens' own forward input, last real positions and
    selected tokens, so the explained target is by construction the generated token
    (after the structural / allowed-token mask); ages [B,T] are the windows' time_age, for
    any model. Each window is first trimmed to its real tokens: windows are right-padded and
    attention is causal, so no pad ever reaches a real position and the trim is exact while
    saving the padded compute. The dict holds context_ages (one per context_tokens entry) and,
    per method, attr_<method> (one signed score per context_tokens entry; NaN where a method
    gives none) and top_<method> (the top_k_tokens events with the largest |score|, each with
    its signed score, as {position, token, score}; ties go to the earlier position).
    """
    rows = []
    for j, (end, target) in enumerate(zip(last.tolist(), picks.tolist())):
        sample = {k: v[j:j + 1, :end + 1] for k, v in batch.items()}
        tokens = [itos[t] for t in sample["input_ids"][0].tolist()]
        row = {"context_ages": ages[j, :end + 1].tolist()}
        for method in explain["methods"]:
            scores = (integrated_gradients(model, sample, target, explain["n_steps"], batch_size)
                      if method == "integrated_gradients"
                      else occlusion(model, sample, target, batch_size))
            scored = np.flatnonzero(np.isfinite(scores))
            top = scored[np.argsort(-np.abs(scores[scored]), kind="stable")][:explain["top_k_tokens"]]
            row[f"attr_{method}"] = scores.tolist()
            row[f"top_{method}"] = [{"position": int(i), "token": tokens[i], "score": float(scores[i])}
                                    for i in top]
        rows.append(row)
    return rows


def _position_rows(explained: pd.DataFrame, method: str, front_tokens: set) -> pd.DataFrame:
    """The explained predictions' rows as pipeline 06's attribution rows, one per scored window
    position: patient_id, sample_index (the prediction), position, token, age, attribution,
    abs_attribution, anchor_age and front_block."""
    lengths = explained["context_tokens"].map(len).to_numpy()
    rows = pd.DataFrame({
        "patient_id": np.repeat(explained["patient_id"].astype(str).to_numpy(), lengths),
        "sample_index": np.repeat(np.arange(len(explained)), lengths),
        "position": np.concatenate([np.arange(n) for n in lengths]),
        "token": np.concatenate([np.asarray(t, dtype=object) for t in explained["context_tokens"]]),
        "age": np.concatenate([np.asarray(a, dtype=np.float64) for a in explained["context_ages"]]),
        "attribution": np.concatenate([np.asarray(a, dtype=np.float64) for a in explained[f"attr_{method}"]]),
        "anchor_age": np.repeat(explained["anchor_age"].to_numpy(np.float64), lengths),
    })
    rows = rows[np.isfinite(rows["attribution"])].reset_index(drop=True)
    return rows.assign(abs_attribution=rows["attribution"].abs(),
                       front_block=rows["token"].isin(front_tokens))


def time_before_prediction_occlusion(explained: pd.DataFrame, front_tokens: set, structural: set,
                                     rng: np.random.Generator, min_patients: int = MIN_PATIENTS) -> pd.DataFrame:
    """The mean occlusion score of the explained predictions' events (each prediction with an
    anchor), one row per cell shown: by window, the last event of the context on its own or the
    time before the prediction (TIME_BIN_LABELS), and by series, whether the event has the
    generated token's code (OCCLUSION_SERIES). A cell holds n_patients, n_events and the mean
    with its patient-bootstrap interval. The front block and structural tokens are left out:
    the start token is never dropped, and a window without its sex token never occurs in
    training. The two series of a window are hidden together under the 5-patient rule
    (hidden_groups)."""
    columns = ["window", "series", "n_patients", "n_events", "mean", "mean_low", "mean_high"]
    rows = _position_rows(explained, "occlusion", front_tokens)
    prediction = rows["sample_index"].to_numpy()
    last = explained["context_tokens"].map(len).to_numpy() - 1
    same = rows["token"].to_numpy() == explained["predicted_token"].to_numpy()[prediction]
    rows = rows.assign(is_last=rows["position"].to_numpy() == last[prediction],
                       series=np.where(same, *OCCLUSION_SERIES))
    rows = rows[~rows["front_block"] & ~rows["token"].isin(structural)]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows = rows.assign(window=np.where(rows["is_last"], LAST_EVENT, _window(rows)))
    cells = []
    for window in (LAST_EVENT, *TIME_BIN_LABELS):
        groups = [rows[(rows["window"] == window) & (rows["series"] == s)] for s in OCCLUSION_SERIES]
        hidden = hidden_groups([set(g["patient_id"]) for g in groups], min_patients)
        for series, g, h in zip(OCCLUSION_SERIES, groups, hidden):
            if h or g.empty:
                continue
            mean, low, high = patient_bootstrap(g["attribution"].to_numpy(), g["patient_id"].to_numpy(), rng)
            cells.append([window, series, g["patient_id"].nunique(), len(g), mean[0], low[0], high[0]])
    return pd.DataFrame(cells, columns=columns)


def plot_time_before_prediction_occlusion(table: pd.DataFrame):
    """A time_before_prediction_occlusion table: (a) the last event, (b) the earlier events by
    time before the prediction; each series a dot (mean) with its interval. Each panel has its
    own scale: dropping the last event also moves the prediction back one event. Returns the
    figure."""
    windows = [w for w in TIME_BIN_LABELS if w in set(table["window"])]
    with plt.rc_context(STYLE):
        fig, (ax_last, ax_earlier) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.0), constrained_layout=True,
                                                  gridspec_kw={"width_ratios": [1, 4]})
        for ax, labels in ((ax_last, [LAST_EVENT]), (ax_earlier, windows)):
            x = np.arange(len(labels))
            for i, series in enumerate(OCCLUSION_SERIES):
                cells = table[table["series"] == series].set_index("window").reindex(labels)
                xs = x + (i - 0.5) * 0.25
                ax.vlines(xs, cells["mean_low"], cells["mean_high"], color=f"C{i}", linewidth=0.8)
                ax.scatter(xs, cells["mean"], color=f"C{i}", s=12, zorder=3, label=series)
            ax.axhline(0, color="0.5", linewidth=0.6)
            ax.set_xlim(-0.6, len(labels) - 0.4)
        ax_last.set_xticks([])
        ax_last.set_ylabel("Mean decrease in log p\n(generated token) without the event\n← lowered p  |  raised p →")
        ax_last.set_title("a  Last event", loc="left", fontweight="bold")
        ax_earlier.set_xticks(np.arange(len(windows)), windows)
        ax_earlier.set_xlabel("Time before the prediction")
        ax_earlier.set_title("b  Earlier events", loc="left", fontweight="bold")
        ax_earlier.legend(frameon=False)
    return fig


def _summarise_occlusion(explained: pd.DataFrame, front_tokens: set, structural: set, meta: dict,
                         out_dir: Path) -> None:
    """time_before_prediction_occlusion of the explained predictions with an anchor in out_dir
    (CSV, and PNG + SVG when a cell is shown), and explainability_occlusion.json: meta plus the
    summary settings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timed = explained["anchor_age"].notna()
    if not timed.all():
        print(f"  [explain] {int((~timed).sum())} explained prediction(s) without an anchor are left "
              f"out of time_before_prediction_occlusion")
    summary = {**meta, "min_patients": MIN_PATIENTS,
               "time_windows_days": {"edges": list(TIME_BIN_EDGES[:-1]), "labels": list(TIME_BIN_LABELS)},
               "n_bootstrap": N_BOOTSTRAP, "seed": SEED}
    if explained["patient_id"].nunique() < MIN_PATIENTS:
        summary["skipped"] = f"fewer than {MIN_PATIENTS} explained patients: no summary written"
    else:
        table = time_before_prediction_occlusion(explained[timed].reset_index(drop=True), front_tokens,
                                                 structural, np.random.default_rng(SEED))
        _write(table, out_dir / "time_before_prediction_occlusion",
               lambda: plot_time_before_prediction_occlusion(table))
    with open(out_dir / "explainability_occlusion.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)


def summarise_generated_explanations(gen_df: pd.DataFrame, explain: dict, vocab_stats: dict,
                                     itos: list[str], output_dir) -> None:
    """The cohort summaries of the explained predictions (the rows of step 1) in
    <output_dir>/explainability/, every number pooling at least MIN_PATIENTS patients:
      Integrated Gradients  pipeline 06's tables (CSV) and figures (PNG + SVG) under 06's names:
                            code_contributions, time_before_prediction, explainability.json
                            (summarise_explanations);
      occlusion             time_before_prediction_occlusion (CSV, PNG + SVG) and
                            explainability_occlusion.json. Occlusion is not summed by code:
                            dropping one copy of a code leaves its other copies, so the sum
                            would understate a code that repeats.
    vocab_stats (load_vocab's) names the structural and front-block tokens."""
    explained = gen_df[gen_df["step"] == 1].reset_index(drop=True)
    special = vocab_stats["special_tokens"]
    structural = set(special)
    front_tokens = {t for t in ("<SOS>", "<CON>") if t in special}
    front_tokens |= {itos[i] for i in vocab_stats.get("sex_token_ids", []) + vocab_stats.get("static_token_ids", [])}
    n_patients = explained["patient_id"].nunique()
    explained_what = {
        "explained": "the token generated at step 1, the prediction at the anchor from the real history",
        "explained_output": "log p(generated token)",
    }
    counts = {
        # 5-patient rule: a count of predictions from fewer than MIN_PATIENTS patients is hidden
        "n_explained_predictions": len(explained) if n_patients >= MIN_PATIENTS else HIDDEN,
        "n_explained_patients": n_patients if n_patients >= MIN_PATIENTS else HIDDEN,
    }
    out_dir = Path(output_dir) / "explainability"
    # Integrated Gradients first: summarise_explanations deletes an earlier run's summary files
    # (time_before_prediction* among them) before it writes, so occlusion's must come after
    if "integrated_gradients" in explain["methods"]:
        meta = {
            **explained_what,
            "method": "integrated_gradients",
            "definition": ("Integrated Gradients from a zero-vector token baseline (times and types kept): "
                           "the scores of a prediction sum to log p(generated token) minus its value with "
                           "no code information"),
            "n_steps": explain["n_steps"],
            **counts,
        }
        summarise_explanations(_position_rows(explained, "integrated_gradients", front_tokens),
                               task="generation", out_dir=out_dir, structural=structural, meta=meta,
                               top_k=explain["top_k_tokens"])
    if "occlusion" in explain["methods"]:
        meta = {
            **explained_what,
            "method": "occlusion",
            "definition": ("log p(generated token) minus its value with the event dropped from the window "
                           "(token, time and type), averaged over the events of a cell; the start token "
                           "and the front block (sex, static) are left out"),
            "note": "dropping the last event also moves the prediction back one event; panel a includes that change",
            **counts,
        }
        _summarise_occlusion(explained, front_tokens, structural, meta, out_dir)
