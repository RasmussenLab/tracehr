# src/tracehr/finetuning/explain_summary.py
"""Cohort summaries of pipeline 06's explanations: the tables (CSV) and figures (PNG + SVG) of
<output_dir>/explainability/, computed from the rows of token_attributions.parquet (one row per
explained prediction x window position, see outcome_explain). Every number pools at least
MIN_PATIENTS distinct patients (the 5-patient rule, finetune_plots), and every figure is drawn
from its table alone, so it can be redrawn from the CSV without a model:

  code_contributions{_classK}       Integrated Gradients by code: importance (mean |contribution|)
                                    and direction (mean contribution), patient-bootstrap intervals
  time_before_prediction{_classK}   the share of each prediction's |IG| per window before it
  attention_by_code{_classK}        the readout attention on each code, per layer, against
                                    uniform attention
  attention_by_time{_classK}        the readout attention on the front block and per window
                                    before the prediction, per layer
  explainability.json               the settings and counts of the run

Multiclass gets one set per explained class: attributions toward different classes share no
sign, so they are never pooled.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..generation.attribution import NEGATIVE_COLOR, POSITIVE_COLOR
from ..generation.model_io import json_safe
from .finetune_plots import FIG_WIDTH, MIN_PATIENTS, STYLE, hidden_groups, save_figure

# Windows before the prediction, in DAYS
TIME_BIN_EDGES = (0, 30, 91, 365, 730, 1826, np.inf)
TIME_BIN_LABELS = ("< 1 month", "1–3 months", "3–12 months", "1–2 years", "2–5 years", "≥ 5 years")
FRONT_BLOCK = "front block"   # the SOS/CON, sex and static positions in the attention tables
N_BOOTSTRAP = 1000            # patient resamples per interval
SEED = 2000                   # the random sample of explained predictions and the bootstrap
_BOOTSTRAP_CHUNK = 100        # resamples drawn at once: memory stays ~100 x n_patients numbers

# The unit of the explained output: Integrated Gradients attributes the head's raw score
# (multiclass: each prediction's own class logit, see _unit); "generation" is pipeline 03's
# generated token (generation.explainability)
_UNIT = {"binary": "log-odds", "regression": "outcome units", "survival": "log relative risk",
         "generation": "log p(generated token)"}
# What a positive / negative contribution pushes the explained output toward
_DIRECTION = {
    "binary": ("event", "no event"),
    "regression": ("higher prediction", "lower prediction"),
    "survival": ("higher risk", "lower risk"),
    "generation": ("raises p(generated token)", "lowers p(generated token)"),
}
# The files summarise_explanations writes, deleted first so a folder never mixes two runs
_SUMMARY_FILES = ("code_contributions*", "time_before_prediction*", "attention_by_*",
                  "explainability.json")


def _unit(task: str, target: int | None) -> str:
    """The unit of the explained output; target is the explained class (multiclass), else None."""
    return _UNIT[task] if target is None else f"class {target} logit"


def _direction(task: str, target: int | None) -> tuple[str, str]:
    """(positive, negative) labels of a contribution; target is the explained class
    (multiclass), else None."""
    return _DIRECTION[task] if target is None else (f"class {target}", f"away from class {target}")


def _labels(codes, names: dict | None) -> list[str]:
    """The display name of each code: names[code] when given, else the code."""
    return [names.get(c, c) if names else c for c in codes]


# ============================================================================ #
# Tables
# ============================================================================ #
def patient_bootstrap(values: np.ndarray, patients: np.ndarray, rng: np.random.Generator,
                      n_bootstrap: int = N_BOOTSTRAP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, low, high) of every column of values ([n_predictions, k], one row per prediction):
    the mean over the predictions and its 95% percentile interval from resampling patients, a
    patient's predictions together."""
    values = np.asarray(values, dtype=np.float64).reshape(len(patients), -1)
    _, idx = np.unique(np.asarray(patients), return_inverse=True)
    n = int(idx.max()) + 1
    sums = np.stack([np.bincount(idx, weights=values[:, j], minlength=n)
                     for j in range(values.shape[1])], axis=1)                  # [n, k]
    counts = np.bincount(idx, minlength=n).astype(np.float64)
    means = []
    for start in range(0, n_bootstrap, _BOOTSTRAP_CHUNK):
        draws = rng.multinomial(n, np.full(n, 1.0 / n),
                                size=min(_BOOTSTRAP_CHUNK, n_bootstrap - start))   # [b, n]
        means.append((draws @ sums) / (draws @ counts)[:, None])
    low, high = np.percentile(np.concatenate(means), [2.5, 97.5], axis=0)
    return sums.sum(axis=0) / counts.sum(), low, high


def _window(rows: pd.DataFrame) -> pd.Series:
    """Each timed event's window before its prediction (TIME_BIN_LABELS)."""
    before = rows["anchor_age"] - rows["age"]
    if before.isna().any() or (before < 0).any():
        raise ValueError("every event must lie a known, non-negative number of days before its anchor")
    return pd.cut(before, TIME_BIN_EDGES, right=False, labels=TIME_BIN_LABELS).astype(str)


def _shares_by_window(rows: pd.DataFrame, value_col: str, windows: list[str],
                      min_patients: int) -> tuple[pd.DataFrame, list[str], dict]:
    """(shares, shown, n_patients) for the rows of the predictions that have a positive total of
    value_col: shares holds one row per prediction (index sample_index, patient_id) and one
    column per window, the share of the prediction's total value_col in it; shown the windows
    the 5-patient rule leaves visible (hidden_groups: the shares add up to 100%); n_patients
    the distinct patients with a position in each window."""
    share = rows[value_col] / rows.groupby("sample_index")[value_col].transform("sum")
    shares = (share.groupby([rows["sample_index"], rows["patient_id"], rows["window"]]).sum()
              .unstack("window", fill_value=0.0).reindex(columns=windows, fill_value=0.0))
    patients = [set(rows.loc[rows["window"] == w, "patient_id"]) for w in windows]
    hidden = hidden_groups(patients, min_patients)
    shown = [w for w, h in zip(windows, hidden) if not h]
    return shares, shown, {w: len(p) for w, p in zip(windows, patients)}


def code_contributions(attr: pd.DataFrame, structural: set, rng: np.random.Generator,
                       min_patients: int = MIN_PATIENTS) -> pd.DataFrame:
    """One row per code seen in at least min_patients patients, ranked by importance. A code's
    contribution to one prediction is the sum of its attributions over that prediction's
    history; mean_abs (importance) and mean (direction) average it over the predictions whose
    history holds the code, with patient-bootstrap intervals. prevalence is the share of
    explained predictions that hold the code. Structural tokens are left out; sex and static
    tokens are kept, as patient attributes."""
    columns = ["code", "n_patients", "n_predictions", "prevalence", "mean_abs", "mean_abs_low",
               "mean_abs_high", "mean", "mean_low", "mean_high"]
    n_explained = attr["sample_index"].nunique()
    per_prediction = (attr[~attr["token"].isin(structural)]
                      .groupby(["token", "sample_index", "patient_id"], as_index=False)["attribution"].sum())
    rows = []
    for code, g in per_prediction.groupby("token"):
        n_patients = g["patient_id"].nunique()
        if n_patients < min_patients:
            continue
        v = g["attribution"].to_numpy()
        mean, low, high = patient_bootstrap(np.column_stack([np.abs(v), v]), g["patient_id"].to_numpy(), rng)
        rows.append([code, n_patients, len(g), len(g) / n_explained,
                     mean[0], low[0], high[0], mean[1], low[1], high[1]])
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("mean_abs", ascending=False, ignore_index=True))


def time_before_prediction(attr: pd.DataFrame, structural: set, rng: np.random.Generator,
                           min_patients: int = MIN_PATIENTS) -> pd.DataFrame:
    """One row per window before the prediction: the mean share of each prediction's |IG| that
    falls on its events in the window, with its patient-bootstrap interval. Only timed events
    count (the front block and structural tokens are left out), so a prediction without any is
    left out too. Windows are hidden under the 5-patient rule."""
    columns = ["window", "n_patients_with_events", "share", "share_low", "share_high"]
    events = attr[~attr["front_block"] & ~attr["token"].isin(structural)]
    events = events[events.groupby("sample_index")["abs_attribution"].transform("sum") > 0]
    left_out = attr["sample_index"].nunique() - events["sample_index"].nunique()
    if left_out:
        print(f"  [explain] {left_out} explained prediction(s) without timed events are left out "
              f"of the time-before-prediction summary")
    if events.empty:
        return pd.DataFrame(columns=columns)
    events = events.assign(window=_window(events))
    shares, shown, n_patients = _shares_by_window(events, "abs_attribution", list(TIME_BIN_LABELS),
                                                  min_patients)
    mean, low, high = patient_bootstrap(shares.to_numpy(), shares.index.get_level_values("patient_id"), rng)
    return pd.DataFrame([[w, n_patients[w], mean[i], low[i], high[i]]
                         for i, w in enumerate(shares.columns) if w in shown], columns=columns)


def _attention_layers(attr: pd.DataFrame) -> list[int]:
    """The layers of the attention_layer{l} columns, in order."""
    return sorted(int(c.removeprefix("attention_layer")) for c in attr.columns
                  if c.startswith("attention_layer"))


def attention_by_time(attr: pd.DataFrame, rng: np.random.Generator,
                      min_patients: int = MIN_PATIENTS) -> pd.DataFrame:
    """One row per (layer, window): the mean share of the readout attention on the front block
    or on the events of the window, with its patient-bootstrap interval, and the share under
    uniform attention. The shares add up to 100% per layer, so the windows are hidden under the
    5-patient rule."""
    columns = ["layer", "window", "n_patients", "share", "share_low", "share_high", "uniform_share"]
    if attr.empty:
        return pd.DataFrame(columns=columns)
    rows = attr.assign(window=FRONT_BLOCK)
    timed = ~rows["front_block"]
    rows.loc[timed, "window"] = _window(rows[timed])
    windows = [FRONT_BLOCK, *TIME_BIN_LABELS]
    uniform, shown, n_patients = _shares_by_window(rows, "attention_uniform", windows, min_patients)
    uniform_mean = uniform.mean(axis=0)
    out = []
    for layer in _attention_layers(attr):
        shares, _, _ = _shares_by_window(rows, f"attention_layer{layer}", windows, min_patients)
        mean, low, high = patient_bootstrap(shares.to_numpy(), shares.index.get_level_values("patient_id"), rng)
        out += [[layer, w, n_patients[w], mean[i], low[i], high[i], uniform_mean[w]]
                for i, w in enumerate(shares.columns) if w in shown]
    return pd.DataFrame(out, columns=columns)


def attention_by_code(attr: pd.DataFrame, structural: set, rng: np.random.Generator,
                      min_patients: int = MIN_PATIENTS) -> pd.DataFrame:
    """One row per (layer, code seen in at least min_patients patients): the mean share of the
    readout attention on the code, over the predictions whose history holds it, with its
    patient-bootstrap interval, and the share under uniform attention. Structural tokens are
    left out. Rows are grouped by layer, each layer's codes ranked by share."""
    columns = ["layer", "code", "n_patients", "share", "share_low", "share_high", "uniform_share"]
    layers = _attention_layers(attr)
    value_cols = [f"attention_layer{layer}" for layer in layers]
    per_prediction = (attr[~attr["token"].isin(structural)]
                      .groupby(["token", "sample_index", "patient_id"], as_index=False)
                      [value_cols + ["attention_uniform"]].sum())
    out = []
    for code, g in per_prediction.groupby("token"):
        n_patients = g["patient_id"].nunique()
        if n_patients < min_patients:
            continue
        mean, low, high = patient_bootstrap(g[value_cols].to_numpy(), g["patient_id"].to_numpy(), rng)
        uniform = g["attention_uniform"].mean()
        out += [[layer, code, n_patients, mean[i], low[i], high[i], uniform]
                for i, layer in enumerate(layers)]
    return (pd.DataFrame(out, columns=columns)
            .sort_values(["layer", "share"], ascending=[True, False], ignore_index=True))


# ============================================================================ #
# Figures (each drawn from its table alone)
# ============================================================================ #
def plot_code_contributions(table: pd.DataFrame, task: str, target: int | None = None,
                            top_k: int = 20, names: dict | None = None):
    """The top_k codes of a code_contributions table: (a) importance, mean |contribution| with
    its interval; (b) direction, the mean contribution of each code with its interval. names
    maps codes to display names. Returns the figure."""
    top = table.head(top_k).iloc[::-1]                       # most important at the top
    y = np.arange(len(top))
    with plt.rc_context(STYLE):
        fig, (ax_abs, ax_dir) = plt.subplots(1, 2, sharey=True, figsize=(FIG_WIDTH, 0.8 + 0.2 * len(top)),
                                             gridspec_kw={"width_ratios": [1, 1.3]}, constrained_layout=True)
        ax_abs.barh(y, top["mean_abs"], color="0.65", height=0.7)
        ax_abs.hlines(y, top["mean_abs_low"], top["mean_abs_high"], color="0.2", linewidth=0.8)
        ax_abs.set_yticks(y, [f"{label} (n={n})" for label, n in zip(_labels(top["code"], names), top["n_patients"])])
        ax_abs.set_xlim(left=0)
        ax_abs.set_xlabel(f"Mean |contribution| ({_unit(task, target)})")
        ax_abs.set_title("a  Importance", loc="left", fontweight="bold")

        colors = [POSITIVE_COLOR if m >= 0 else NEGATIVE_COLOR for m in top["mean"]]
        ax_dir.axvline(0, color="0.5", linewidth=0.6)
        ax_dir.hlines(y, top["mean_low"], top["mean_high"], colors=colors, linewidth=0.8)
        ax_dir.scatter(top["mean"], y, c=colors, s=12, zorder=3)
        pos, neg = _direction(task, target)
        ax_dir.set_xlabel(f"Mean contribution ({_unit(task, target)})\n← {neg}  |  {pos} →")
        ax_dir.set_title("b  Direction", loc="left", fontweight="bold")
        ax_dir.tick_params(axis="y", length=0)
    return fig


def plot_time_before_prediction(table: pd.DataFrame):
    """A time_before_prediction table: the share of |IG| per window, as bars with intervals.
    Returns the figure."""
    x = np.arange(len(table))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(0.6 * FIG_WIDTH, 2.4), constrained_layout=True)
        ax.bar(x, 100 * table["share"], color="0.65", width=0.7)
        ax.vlines(x, 100 * table["share_low"], 100 * table["share_high"], color="0.2", linewidth=0.8)
        ax.set_xticks(x, table["window"], rotation=30, ha="right")
        ax.set_xlabel("Time before the prediction")
        ax.set_ylabel("Share of |attribution| (%)")
        ax.set_ylim(bottom=0)
    return fig


def plot_attention_by_time(table: pd.DataFrame):
    """An attention_by_time table as a heatmap: one row per layer, then the uniform reference;
    one column for the front block and each shown window; each cell its mean share (%).
    Returns the figure."""
    windows = list(dict.fromkeys(table["window"]))
    layers = sorted(table["layer"].unique())
    shares = table.pivot(index="layer", columns="window", values="share").reindex(index=layers, columns=windows)
    uniform = table[table["layer"] == layers[0]].set_index("window")["uniform_share"].reindex(windows)
    values = 100 * np.vstack([shares.to_numpy(), uniform.to_numpy()[None]])
    row_labels = [f"Layer {layer}" for layer in layers] + ["Uniform"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 0.9 + 0.28 * len(row_labels)), constrained_layout=True)
        vmax = max(float(values.max()), 1e-9)
        mesh = ax.pcolormesh(values, cmap="Blues", vmin=0, vmax=vmax, edgecolors="white", linewidth=0.5)
        for (r, c), v in np.ndenumerate(values):
            ax.text(c + 0.5, r + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if v > 0.6 * vmax else "black")
        ax.set_xticks(np.arange(len(windows)) + 0.5, windows, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(row_labels)) + 0.5, row_labels)
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Position in the history")
        fig.colorbar(mesh, ax=ax, label="Share of readout attention (%)")
    return fig


def plot_attention_by_code(table: pd.DataFrame, top_k: int = 20, names: dict | None = None):
    """The final layer's top_k codes of an attention_by_code table: the mean share of readout
    attention with its interval (filled), and the share under uniform attention (open). names
    maps codes to display names. Returns the figure."""
    final = table[table["layer"] == table["layer"].max()]
    top = final.sort_values("share", ascending=False).head(top_k).iloc[::-1]
    y = np.arange(len(top))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(0.6 * FIG_WIDTH, 0.8 + 0.2 * len(top)), constrained_layout=True)
        ax.hlines(y, 100 * top["share_low"], 100 * top["share_high"], color="0.2", linewidth=0.8)
        ax.scatter(100 * top["share"], y, color="C0", s=14, zorder=3, label="Observed")
        ax.scatter(100 * top["uniform_share"], y, facecolors="none", edgecolors="0.4", s=14, zorder=3,
                   label="Uniform attention")
        ax.set_yticks(y, [f"{label} (n={n})" for label, n in zip(_labels(top["code"], names), top["n_patients"])])
        ax.set_xlim(left=0)
        ax.set_xlabel("Share of final-layer readout attention (%)")
        ax.tick_params(axis="y", length=0)
        ax.legend(frameon=False, loc="lower right")
    return fig


# ============================================================================ #
# The summary of one explanation run (pipeline 06)
# ============================================================================ #
def summarise_explanations(attr: pd.DataFrame, *, task: str, out_dir, structural: set,
                           meta: dict, min_patients: int = MIN_PATIENTS, top_k: int = 20) -> None:
    """Every table (CSV) and figure (PNG + SVG) of the explanation rows attr in out_dir, and
    explainability.json: meta plus the summary settings. With the attention columns, the
    attention summaries too. Multiclass gets one set per explained class (suffix _class{k}); a
    group of fewer than min_patients patients is skipped, and the json says so. An earlier run's
    summary files in out_dir are deleted first, so a folder never mixes two runs. structural:
    the names of the structural tokens (left out of the code tables)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in _SUMMARY_FILES:
        for path in out_dir.glob(pattern):
            path.unlink()

    rng = np.random.default_rng(SEED)
    with_attention = "attention_uniform" in attr.columns
    # The time tables need each prediction's anchor; one without (a dataset with no anchor ages)
    # has no time before the prediction
    timed = attr["anchor_age"].notna() if "anchor_age" in attr else pd.Series(dtype=bool)
    if not timed.all():
        print(f"  [explain] {attr.loc[~timed, 'sample_index'].nunique()} explained prediction(s) "
              f"without an anchor are left out of the time summaries")
    if attr.empty:                         # no explained prediction: only the json is written
        groups = []
    elif task != "multiclass":
        groups = [(None, attr)]
    else:
        groups = [(int(k), g) for k, g in attr.groupby("target_class")]
    written, skipped = [], []
    for target, group in groups:
        if group["patient_id"].nunique() < min_patients:
            skipped.append(target)
            continue
        written.append(target)
        suffix = "" if target is None else f"_class{target}"
        codes = code_contributions(group, structural, rng, min_patients)
        times = time_before_prediction(group[timed[group.index]], structural, rng, min_patients)
        _write(codes, out_dir / f"code_contributions{suffix}",
               lambda: plot_code_contributions(codes, task, target, top_k))
        _write(times, out_dir / f"time_before_prediction{suffix}",
               lambda: plot_time_before_prediction(times))
        if with_attention:
            by_code = attention_by_code(group, structural, rng, min_patients)
            by_time = attention_by_time(group[timed[group.index]], rng, min_patients)
            _write(by_code, out_dir / f"attention_by_code{suffix}",
                   lambda: plot_attention_by_code(by_code, top_k))
            _write(by_time, out_dir / f"attention_by_time{suffix}",
                   lambda: plot_attention_by_time(by_time))

    summary = {**meta, "min_patients": min_patients,
               "time_windows_days": {"edges": list(TIME_BIN_EDGES[:-1]), "labels": list(TIME_BIN_LABELS)},
               "n_bootstrap": N_BOOTSTRAP, "seed": SEED}
    if task == "multiclass":
        summary["classes_summarised"] = written
        summary["classes_skipped"] = skipped      # fewer than min_patients explained patients
    elif not written:
        summary["skipped"] = f"fewer than {min_patients} explained patients: no summary written"
    with open(out_dir / "explainability.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)


def _write(table: pd.DataFrame, stem: Path, draw) -> None:
    """stem.csv, and the figure draw() returns as stem.png/.svg when the table has rows."""
    table.to_csv(stem.parent / f"{stem.name}.csv", index=False, float_format="%.6g")
    if not table.empty:
        save_figure(draw(), stem)
