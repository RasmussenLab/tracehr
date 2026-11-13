# src/tracehr/finetuning/outcome_evaluation.py
"""Evaluation of an NTPOutcomeModel (pipeline 06): the task's metrics over a loader, the
metrics as they may be reported (reportable_metrics), and the task's evaluation figure of the
test split with its tables (ROC/PR and calibration, calibration, confusion matrix, or
Kaplan-Meier curves by predicted-risk group). What is reported follows the 5-patient rule
(finetune_plots)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from ..training.runtime import ProgressLog, batch_to_device
from .finetune_plots import FIG_WIDTH, HIDDEN, MIN_PATIENTS, STYLE, hidden_groups, save_figure
from .survival_core import kaplan_meier, risk_groups, split_survival_labels, survival_metrics


@torch.no_grad()
def evaluate_outcome(
    model,
    loader: DataLoader,
    device: torch.device,
    task: str,
    show_progress: bool = False,
) -> dict[str, float]:
    """Task-generic evaluation for NTPOutcomeModel.

    Returns a dict that always contains "loss" plus the task's monitored metric
    (binary->auroc, multiclass->macro_f1, regression->mae, survival->cindex) and
    companion metrics. Collected predictions are: probabilities (binary), class-prob
    matrix (multiclass), raw values (regression), or log relative risk (survival).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_weight = 0.0, 0.0

    for batch in ProgressLog(loader, "[evaluate]", disable=not show_progress):
        batch = batch_to_device(batch, device)
        out = model(batch)
        # Only the two classification tasks produce a probability; regression and survival
        # both report the head's raw output.
        key = "probs" if task in ("binary", "multiclass") else "predictions"
        all_preds.append(out[key].cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
        if "loss" in out:
            # Scale each batch's mean loss back by what it was divided by, so the sum over
            # batches is the mean over the whole split, whatever the batch size: the batch
            # size, except for class-weighted cross-entropy, which PyTorch divides by the
            # summed class weights of the batch's labels.
            weight = (model.class_weights[batch["labels"].long()].sum().item()
                      if task == "multiclass" and model.class_weights is not None
                      else batch["labels"].shape[0])
            total_loss += out["loss"].item() * weight
            total_weight += weight

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    # A sum of class weights can be below 1, so only an empty split is guarded
    metrics = {"loss": total_loss / total_weight if total_weight else 0.0, "n_total": int(len(labels))}

    if task == "survival":
        # Straight from survival_core, the same function the stage-05 arms report through,
        # so "cindex" means exactly the same thing in both stages' metric files. It also
        # supplies n_total, so drop the scalar-label version set above.
        metrics.pop("n_total", None)
        metrics.update(survival_metrics(labels, preds))
    elif task == "regression":
        metrics["mse"] = float(mean_squared_error(labels, preds))
        metrics["mae"] = float(mean_absolute_error(labels, preds))
        metrics["r2"] = float(r2_score(labels, preds)) if len(labels) > 1 else 0.0
    elif task == "binary":
        y = labels.astype(int)
        pred = (preds >= 0.5).astype(int)
        # A one-class split leaves both ranking metrics undefined. NaN, not 0.0, for the
        # reason stage 05's classification_metrics gives: 0.0 reads as a measured failure. json_safe
        # writes it as null.
        both = len(np.unique(y)) > 1
        metrics.update({
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, preds)) if both else float("nan"),
            "auprc": float(average_precision_score(y, preds)) if both else float("nan"),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        })
    else:  # multiclass
        y = labels.astype(int)
        pred_class = preds.argmax(axis=-1)
        present = np.unique(y)
        metrics["accuracy"] = float(accuracy_score(y, pred_class))
        metrics["macro_f1"] = float(f1_score(y, pred_class, average="macro", zero_division=0))
        # One-vs-rest AUROC needs every class represented in the labels. With K=2 sklearn reads the
        # labels as binary and wants one score column: the class-1 probability, whose AUROC equals
        # the two-class one-vs-rest macro average.
        scores = preds[:, 1] if preds.shape[1] == 2 else preds
        metrics["ovr_auroc"] = (
            float(roc_auc_score(y, scores, multi_class="ovr", average="macro"))
            if len(present) == preds.shape[1] else float("nan")
        )

    return metrics


# ============================================================================ #
# What may be reported (the 5-patient rule, finetune_plots)
# ============================================================================ #
# The two outcome groups of a binary or survival split: their counts, and the metrics computed
# on them, are hidden together when either group is small
_GROUP_COUNTS = {"binary": ("n_positive", "n_negative"),
                 "survival": ("n_events", "n_censored", "event_rate")}
_GROUP_METRICS = {"binary": ("precision", "recall", "f1", "auroc", "auprc"), "survival": ("cindex",)}


def _small_outcome_group(task: str, labels: np.ndarray, patient_ids, min_patients: int) -> bool:
    """True when one outcome group of a binary or survival split (positive / negative, event /
    censored) is hidden under the 5-patient rule (hidden_groups: the two add up to the split)."""
    if task not in _GROUP_COUNTS:
        return False
    labels = np.asarray(labels)
    in_group = (labels.astype(int) == 1) if task == "binary" else (split_survival_labels(labels)[1] == 1)
    pids = np.asarray(patient_ids)
    return any(hidden_groups([set(pids[in_group]), set(pids[~in_group])], min_patients))


def reportable_metrics(metrics: dict, task: str, labels: np.ndarray, patient_ids,
                       min_patients: int = MIN_PATIENTS) -> dict:
    """evaluate_outcome's metrics of a split as they may be reported, under the 5-patient rule:
    every value HIDDEN when the split has fewer than min_patients patients; for binary and
    survival, the counts of the two outcome groups and the metrics computed on them HIDDEN when
    either group has fewer than min_patients patients; median_duration (survival: one patient's
    value) left out. labels and patient_ids are the split's, in the same order. evaluate_outcome
    itself is unchanged, so training and model selection see every value."""
    out = {k: v for k, v in metrics.items() if k != "median_duration"}
    if len(set(np.asarray(patient_ids).tolist())) < min_patients:
        return {k: HIDDEN for k in out}
    if _small_outcome_group(task, labels, patient_ids, min_patients):
        out.update({k: HIDDEN for k in _GROUP_COUNTS[task] + _GROUP_METRICS[task] if k in out})
    return out


# ============================================================================ #
# The test-evaluation figure and its tables
# ============================================================================ #
_GRID = np.linspace(0.0, 1.0, 101)            # the FPR and recall grid of the ROC/PR tables
_SMALL_NOTE = f"Not shown: an outcome group has fewer than {MIN_PATIENTS} patients"


def plot_outcome_eval(task: str, labels: np.ndarray, preds: np.ndarray, patient_ids, out_dir) -> None:
    """test_eval.png/.svg in out_dir, the task's evaluation figure of the test split, and the
    tables it is drawn from (test_eval_<table>.csv), under the 5-patient rule: binary (a) ROC,
    (b) precision-recall, (c) calibration; regression calibration; multiclass the row-normalised
    confusion matrix; survival Kaplan-Meier curves by predicted-risk group. When the split or
    one of its outcome groups has fewer than MIN_PATIENTS patients, the figure shows a note and
    the tables have no rows, as reportable_metrics hides those metrics. labels, preds and
    patient_ids are the split's, in the same order."""
    out_dir = Path(out_dir)
    codes = pd.factorize(np.asarray(patient_ids))[0]          # one integer per patient
    small = (len(np.unique(codes)) < MIN_PATIENTS
             or _small_outcome_group(task, labels, patient_ids, MIN_PATIENTS))
    draw = {"binary": _binary_eval, "regression": _regression_eval,
            "multiclass": _multiclass_eval, "survival": _survival_eval}[task]
    with plt.rc_context(STYLE):
        fig, tables = draw(np.asarray(labels), np.asarray(preds), codes, small)
    for name, table in tables.items():
        table.to_csv(out_dir / f"test_eval_{name}.csv", index=False, float_format="%.6g")
    save_figure(fig, out_dir / "test_eval")


def _n_patients(codes: np.ndarray) -> int:
    """The number of distinct patients among these patient codes."""
    return len(np.unique(codes))


def _note(ax, text: str = _SMALL_NOTE) -> None:
    """Replace a panel by a centred note."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_axis_off()


def _calibration_bins(order: np.ndarray, ok) -> list[np.ndarray]:
    """Up to 10 bins of the predictions order indexes (sorted by prediction): neighbouring
    tenths are merged until ok(bin) holds, and what is left at the end joins the last bin, so no
    bin is too small and none is hidden (an empty list when no bin can be made)."""
    bins, current = [], np.empty(0, dtype=int)
    for part in np.array_split(order, 10):
        current = np.concatenate([current, part])
        if ok(current):
            bins.append(current)
            current = np.empty(0, dtype=int)
    if len(current) and bins:
        bins[-1] = np.concatenate([bins[-1], current])
    return bins


def _wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval of the proportion k / n."""
    centre = (k + z ** 2 / 2) / (n + z ** 2)
    half = z * np.sqrt(k * (n - k) / n + z ** 2 / 4) / (n + z ** 2)
    return centre - half, centre + half


def _binary_eval(labels, preds, codes, small):
    """(figure, tables) of a binary split: (a) ROC and (b) precision-recall on _GRID, each
    point drawn only when all four cells of its 2x2 table hold MIN_PATIENTS patients, and (c)
    calibration in bins of MIN_PATIENTS patients with and without the event."""
    y, p = labels.ravel().astype(int), preds.ravel()
    tables = {"roc": pd.DataFrame(columns=["fpr", "tpr"]),
              "pr": pd.DataFrame(columns=["recall", "precision"]),
              "calibration": pd.DataFrame(columns=["bin", "n_patients", "mean_predicted", "observed_rate",
                                                   "observed_low", "observed_high"])}
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.5), constrained_layout=True)
    if small or len(np.unique(y)) < 2:
        for ax in axes:
            ax.set_axis_off()
        _note(axes[1], _SMALL_NOTE if small else "Not shown: the test split has one outcome class")
        return fig, tables

    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = tpr * n_pos / (tpr * n_pos + fpr * n_neg)   # undefined where nothing is flagged
    checked = {}                                            # operating point -> cells_ok

    def cells_ok(i: int) -> bool:
        """True when every cell of operating point i's 2x2 table holds MIN_PATIENTS patients."""
        if i not in checked:
            flagged = p >= thresholds[i]
            checked[i] = all(_n_patients(codes[m]) >= MIN_PATIENTS for m in
                             (flagged & (y == 1), flagged & (y == 0), ~flagged & (y == 1), ~flagged & (y == 0)))
        return checked[i]

    roc_rows, pr_rows = [], []
    for f in _GRID:
        i = int(np.searchsorted(fpr, f, side="right")) - 1          # the last point with FPR <= f
        if cells_ok(i):
            roc_rows.append((f, tpr[i]))
    for r in _GRID:
        candidates = np.flatnonzero((tpr >= r) & np.isfinite(precision))
        if len(candidates):
            i = int(candidates[np.argmax(precision[candidates])])   # the best precision at recall >= r
            if cells_ok(i):
                pr_rows.append((r, precision[i]))
    tables["roc"] = pd.DataFrame(roc_rows, columns=["fpr", "tpr"])
    tables["pr"] = pd.DataFrame(pr_rows, columns=["recall", "precision"])

    bins = _calibration_bins(np.argsort(p, kind="stable"), lambda b: (
        _n_patients(codes[b][y[b] == 1]) >= MIN_PATIENTS and _n_patients(codes[b][y[b] == 0]) >= MIN_PATIENTS))
    tables["calibration"] = pd.DataFrame(
        [(k + 1, _n_patients(codes[b]), p[b].mean(), y[b].mean(), *_wilson(y[b].sum(), len(b)))
         for k, b in enumerate(bins)], columns=tables["calibration"].columns)

    roc, pr, cal = tables["roc"], tables["pr"], tables["calibration"]
    ax_roc, ax_pr, ax_cal = axes
    ax_roc.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=0.8)
    ax_roc.plot(roc["fpr"], roc["tpr"], color="C0", label=f"AUROC = {roc_auc_score(y, p):.3f}")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax_roc.set_title("a  ROC", loc="left", fontweight="bold")
    ax_pr.axhline(n_pos / len(y), color="0.6", linestyle="--", linewidth=0.8, label="Prevalence")
    ax_pr.plot(pr["recall"], pr["precision"], color="C0", label=f"AUPRC = {average_precision_score(y, p):.3f}")
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    ax_pr.set_title("b  Precision-recall", loc="left", fontweight="bold")
    ax_cal.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=0.8)
    ax_cal.errorbar(cal["mean_predicted"], cal["observed_rate"],
                    yerr=[cal["observed_rate"] - cal["observed_low"], cal["observed_high"] - cal["observed_rate"]],
                    fmt="o", markersize=3, color="C0", elinewidth=0.8)
    ax_cal.set(xlabel="Mean predicted probability", ylabel="Observed event rate", xlim=(0, 1), ylim=(0, 1))
    ax_cal.set_title("c  Calibration", loc="left", fontweight="bold")
    for ax in (ax_roc, ax_pr):
        ax.legend(frameon=False, loc="lower right" if ax is ax_roc else "upper right")
    return fig, tables


def _regression_eval(labels, preds, codes, small):
    """(figure, tables) of a regression split: the mean observed against the mean predicted
    value in bins of the prediction, each of MIN_PATIENTS patients, with 95% intervals."""
    y, p = labels.ravel().astype(float), preds.ravel()
    columns = ["bin", "n_patients", "mean_predicted", "mean_observed", "observed_low", "observed_high"]
    fig, ax = plt.subplots(figsize=(0.5 * FIG_WIDTH, 0.45 * FIG_WIDTH), constrained_layout=True)
    if small:
        _note(ax)
        return fig, {"calibration": pd.DataFrame(columns=columns)}
    bins = _calibration_bins(np.argsort(p, kind="stable"), lambda b: _n_patients(codes[b]) >= MIN_PATIENTS)
    rows = []
    for k, b in enumerate(bins):
        half = 1.96 * y[b].std(ddof=1) / np.sqrt(len(b))
        rows.append((k + 1, _n_patients(codes[b]), p[b].mean(), y[b].mean(), y[b].mean() - half, y[b].mean() + half))
    cal = pd.DataFrame(rows, columns=columns)
    lo = float(min(cal["mean_predicted"].min(), cal["observed_low"].min()))
    hi = float(max(cal["mean_predicted"].max(), cal["observed_high"].max()))
    ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=0.8)
    ax.errorbar(cal["mean_predicted"], cal["mean_observed"],
                yerr=[cal["mean_observed"] - cal["observed_low"], cal["observed_high"] - cal["mean_observed"]],
                fmt="o", markersize=3, color="C0", elinewidth=0.8)
    ax.set(xlabel="Mean predicted value", ylabel="Mean observed value")
    ax.grid(True, alpha=0.3)
    return fig, {"calibration": cal}


def _multiclass_eval(labels, preds, codes, small):
    """(figure, tables) of a multiclass split: the confusion matrix as the % of each true
    class, a cell of fewer than MIN_PATIENTS patients hidden with its row's complement
    (hidden_groups: a row adds up to 100%)."""
    y, pred = labels.astype(int), preds.argmax(axis=-1)
    K = preds.shape[1]
    columns = ["true_class", "predicted_class", "n_predictions", "row_percent"]
    fig, ax = plt.subplots(figsize=(min(FIG_WIDTH, 1.2 + 0.6 * K), min(FIG_WIDTH, 1.0 + 0.6 * K)),
                           constrained_layout=True)
    if small:
        _note(ax)
        return fig, {"confusion": pd.DataFrame(columns=columns)}
    percent = np.full((K, K), np.nan)
    rows = []
    for t in range(K):
        in_row = y == t
        cells = [codes[in_row & (pred == c)] for c in range(K)]
        hidden = hidden_groups([set(cell) for cell in cells])
        for c in range(K):
            share = 100 * len(cells[c]) / in_row.sum() if in_row.any() else np.nan
            if not hidden[c]:
                percent[t, c] = share
            # A column mixing numbers and HIDDEN escapes to_csv's float_format: round it here
            rows.append((t, c, HIDDEN if hidden[c] else len(cells[c]),
                         HIDDEN if hidden[c] else float(f"{share:.6g}")))
    cmap = plt.get_cmap("Blues").with_extremes(bad="0.88")    # hidden cells in grey
    mesh = ax.pcolormesh(np.ma.masked_invalid(percent), cmap=cmap, vmin=0, vmax=100,
                         edgecolors="white", linewidth=0.5)
    for t in range(K):
        for c in range(K):
            v = percent[t, c]
            ax.text(c + 0.5, t + 0.5, f"{v:.0f}%" if np.isfinite(v) else HIDDEN, ha="center", va="center",
                    fontsize=6, color="white" if np.isfinite(v) and v > 60 else "black")
    ax.set_xticks(np.arange(K) + 0.5, range(K))
    ax.set_yticks(np.arange(K) + 0.5, range(K))
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.set(xlabel="Predicted class", ylabel="True class")
    fig.colorbar(mesh, ax=ax, label="% of the true class")
    return fig, {"confusion": pd.DataFrame(rows, columns=columns)}


def _nice_step(x: float) -> float:
    """The smallest of 1, 2, 5 x 10^k that is at least x (1 for x <= 0)."""
    if x <= 0:
        return 1.0
    base = 10.0 ** np.floor(np.log10(x))
    return float(next(m * base for m in (1, 2, 5, 10) if m * base >= x))


def _km_points(durations, events, codes, grid, times, survival) -> list[tuple]:
    """(time, survival, n_at_risk) of one group's Kaplan-Meier curve at the grid times it may
    show: a time is kept once at least MIN_PATIENTS patients had an event since the last kept
    one, and the curve stops where fewer than MIN_PATIENTS patients remain at risk."""
    points, last = [(0.0, 1.0, len(durations))], 0.0
    for t in grid[1:]:
        at_risk = durations >= t
        if _n_patients(codes[at_risk]) < MIN_PATIENTS:
            break
        if _n_patients(codes[(events == 1) & (durations > last) & (durations <= t)]) >= MIN_PATIENTS:
            points.append((float(t), float(survival[np.searchsorted(times, t, side="right") - 1]),
                           int(at_risk.sum())))
            last = t
    return points


def _survival_eval(labels, preds, codes, small):
    """(figure, tables) of a survival split: observed Kaplan-Meier curves by predicted-risk
    group (tertiles of the risk score) on a fixed time grid, every step pooling MIN_PATIENTS
    patients' events, each curve stopping below MIN_PATIENTS at risk, with the numbers at risk
    under the axis. Ordered, separated curves mean the risk score discriminates."""
    columns = ["group", "time", "survival", "n_at_risk"]
    fig, ax = plt.subplots(figsize=(0.55 * FIG_WIDTH, 0.45 * FIG_WIDTH), constrained_layout=True)
    if small:
        _note(ax)
        return fig, {"km": pd.DataFrame(columns=columns)}
    durations, events = split_survival_labels(labels)
    n_groups = 3
    group = risk_groups(preds, n_groups)
    hidden = hidden_groups([set(codes[group == g]) for g in range(n_groups)])
    step = _nice_step(durations.max() / 50)
    grid = np.arange(0.0, durations.max() + step, step)
    names = {0: "low risk", n_groups - 1: "high risk"}
    shades = plt.cm.plasma(np.linspace(0.05, 0.75, n_groups))
    rows, drawn = [], []
    for g in range(n_groups):
        sel = group == g
        if hidden[g] or not sel.any():
            continue
        times, survival = kaplan_meier(durations[sel], events[sel])
        points = _km_points(durations[sel], events[sel], codes[sel], grid, times, survival)
        name = names.get(g, f"group {g + 1}")
        rows += [(name, t, s, n) for t, s, n in points]
        drawn.append((g, name, durations[sel], points[-1][0]))
        ax.plot([t for t, _, _ in points], [s for _, s, _ in points], drawstyle="steps-post",
                color=shades[g], label=name)
    if not drawn:
        _note(ax)
        return fig, {"km": pd.DataFrame(columns=columns)}
    x_max = max(end for *_, end in drawn) or step
    ticks = np.linspace(0.0, x_max, 5)
    ax.set_xticks(ticks)
    ax.set(xlabel="Time from anchor (days)", ylabel="Survival probability", ylim=(0, 1.02), xlim=(0, x_max))
    ax.legend(frameon=False)
    # Numbers at risk under the axis, only where the curve is shown (so each is >= MIN_PATIENTS)
    ax.text(0, -0.30, "At risk", transform=ax.transAxes, ha="right", fontsize=6)
    for row, (g, name, group_durations, end) in enumerate(drawn):
        y_pos = -0.38 - 0.08 * row
        for t in ticks:
            if t <= end:
                ax.text(t, y_pos, str(int((group_durations >= t).sum())), transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=6, color=shades[g])
    return fig, {"km": pd.DataFrame(rows, columns=columns)}
