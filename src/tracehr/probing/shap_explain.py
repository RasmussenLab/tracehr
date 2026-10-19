# src/tracehr/probing/shap_explain.py
"""Persist the downstream models trained in run_downstream_comparison (see its arms table)
and explain them with SHAP.

Each model is saved to disk (XGBoost json; LR + StandardScaler via joblib) and
explained on the test split: raw per-patient SHAP values (parquet), a ranked
top-feature summary (json), and a density-violin plot where every drawn part is
backed by at least `min_bin_count` observations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


def _slug(name: str) -> str:
    """File-name slug of an arm name: "Embedding + LR" -> "embedding_lr", "Sex+Age + LR" ->
    "sexage_lr"."""
    return name.lower().replace(" + ", "_").replace("+", "").replace(" ", "_")


# ============================================================================ #
# Model persistence (save only; reload is written separately when needed)
# ============================================================================ #
def save_downstream_models(results: dict[str, dict], output_dir) -> None:
    """Save each fitted model (and its scaler) so it can be reloaded later.

    XGBoost via booster.save_model, scikit-learn LR + scaler via joblib.dump.
    """
    output_dir = Path(output_dir)
    for name, r in results.items():
        slug = _slug(name)
        if r["kind"] == "tree":
            r["model"].save_model(str(output_dir / f"{slug}_model.json"))
        else:
            joblib.dump(r["model"], output_dir / f"{slug}_model.joblib")
            joblib.dump(r["scaler"], output_dir / f"{slug}_scaler.joblib")


# ============================================================================ #
# SHAP value computation
# ============================================================================ #
def _densify(X) -> np.ndarray:
    """SHAP needs a dense [N, F] block. Callers sample rows first, so this stays bounded."""
    return np.asarray(X.toarray() if sparse.issparse(X) else X, dtype=float)


def _subsample(X, n_max: int, seed: int = 0):
    """Take at most n_max rows, deterministically. Returns (rows, was_sampled)."""
    if X.shape[0] <= n_max:
        return X, False
    idx = np.sort(np.random.default_rng(seed).choice(X.shape[0], n_max, replace=False))
    return X[idx], True


# Background rows behind a linear model's SHAP reference point (the feature means): the
# shap package's LinearExplainer used 100; 1,000 makes that mean ~3x more precise.
LINEAR_BACKGROUND_ROWS = 1_000


def _linear_shap_values(model, X_bg, X_ex) -> np.ndarray:
    """Exact SHAP values of a linear model, as shap.LinearExplainer computes them:
    coef_j * (x_j - mean_j), the mean over at most LINEAR_BACKGROUND_ROWS background rows
    (a deterministic sample above that). LR, Ridge and the Cox arm all expose coef_ (binary
    LR as [1, F], so its values are in log-odds)."""
    X_bg, _ = _subsample(X_bg, LINEAR_BACKGROUND_ROWS)
    mean = np.asarray(X_bg.mean(axis=0), dtype=float).ravel()
    coef = np.asarray(model.coef_, dtype=float).ravel()
    return (_densify(X_ex) - mean) * coef


def _shap_values(kind: str, model, scaler, X_bg: np.ndarray, X_ex: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """SHAP values on the test matrix plus the raw feature values used for coloring.

    Tree models are explained on the raw matrix; linear models on the SAME scaled
    space the model was fit on (scaler.transform), while coloring keeps the raw
    (unscaled) feature values so "high/low" stays interpretable.
    """
    if kind == "tree":
        import xgboost as xgb
        # Exact TreeSHAP straight from the booster (the same C++ kernel the shap package's
        # TreeExplainer wraps). DMatrix takes the sparse matrix directly; only the returned
        # contributions and the coloring values are dense.
        contribs = np.asarray(model.predict(xgb.DMatrix(X_ex), pred_contribs=True))
        return contribs[:, :-1], _densify(X_ex)  # last column is the bias/base value
    Xs_bg = scaler.transform(X_bg) if scaler is not None else X_bg
    Xs_ex = scaler.transform(X_ex) if scaler is not None else X_ex
    return _linear_shap_values(model, Xs_bg, Xs_ex), _densify(X_ex)


def explain_downstream_models(results: dict[str, dict],
                              shap_inputs: dict[str, tuple[np.ndarray, np.ndarray, Sequence[str]]],
                              output_dir, max_display: int = 15,
                              max_rows: int = 20_000) -> None:
    """For each model: write raw SHAP values (parquet), a density-violin plot (png),
    and a mean|SHAP| top-feature ranking (json).

    SHAP output is per-row and dense, so a cohort of millions would need more memory for
    the explanation than for the model. Above max_rows the explain and background sets are
    deterministically subsampled; the ranking and the plot are distributional, so a sample
    of this size is representative. Smaller cohorts are explained in full.
    """
    output_dir = Path(output_dir)
    for name, r in results.items():
        if name not in shap_inputs:
            continue
        X_bg, X_ex, feat_names = shap_inputs[name]
        feat_names = list(feat_names)
        X_ex, sampled = _subsample(X_ex, max_rows)
        X_bg, _ = _subsample(X_bg, max_rows)
        if sampled:
            print(f"[shap] {name}: explaining a {max_rows}-row sample of the test split.")
        sv, fv = _shap_values(r["kind"], r["model"], r.get("scaler"), X_bg, X_ex)
        slug = _slug(name)

        pd.DataFrame(sv, columns=feat_names).to_parquet(
            output_dir / f"shap_{slug}_values.parquet", index=False)

        fig = plot_shap_density_violin(sv, feat_names, fv, max_display=max_display,
                                       title=f"SHAP — {name}")
        fig.savefig(output_dir / f"shap_{slug}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        order = np.argsort(np.nanmean(np.abs(sv), axis=0))[::-1][:max_display]
        top = {feat_names[i]: float(np.nanmean(np.abs(sv[:, i]))) for i in order}
        with open(output_dir / f"shap_{slug}_top_features.json", "w") as f:
            json.dump(top, f, indent=2)


# ============================================================================ #
# Density-violin plot
# ============================================================================ #
def plot_shap_density_violin(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    feature_values: np.ndarray | None = None,
    max_display: int = 15,
    title: str | None = None,
    min_bin_count: int = 5,
):
    """Density-violin figure of the max_display features with the largest mean |SHAP|. With
    feature_values, a violin is drawn from histogram bins of at least min_bin_count values,
    each colored by its mean feature value (low blue, high red); a feature with too few
    finite values, or a constant one, is drawn gray."""
    shap_values = np.asarray(shap_values, dtype=float)
    feature_names = list(feature_names)
    if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
        raise ValueError("SHAP values must be a 2D array matching feature_names.")
    if feature_values is not None:
        feature_values = np.asarray(feature_values, dtype=float)
        if feature_values.shape != shap_values.shape:
            raise ValueError("feature_values must have the same shape as shap_values.")

    mean_abs = np.nanmean(np.abs(shap_values), axis=0)
    order = np.argsort(np.nan_to_num(mean_abs, nan=-np.inf))[::-1][:max_display]

    def density_values(vals):
        """The finite values of vals (None when there are none); a constant set is widened
        by a tiny epsilon so its violin has a width to draw."""
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        if vals.size == 1 or np.nanmin(vals) == np.nanmax(vals):
            val = float(vals[0])
            eps = max(abs(val) * 1e-9, 1e-12)
            vals = np.array([val - eps, val, val + eps])
        return vals

    records = []
    for idx in order:
        vals = density_values(shap_values[:, idx])
        if vals is None:
            continue
        record = {"label": feature_names[idx], "shap": vals, "feature": None}
        if feature_values is None:
            records.append(record)
            continue

        fvals = feature_values[:, idx]
        mask = np.isfinite(shap_values[:, idx]) & np.isfinite(fvals)
        if mask.sum() >= min_bin_count * 2 and np.nanmin(fvals[mask]) != np.nanmax(fvals[mask]):
            record = {"label": feature_names[idx], "shap": shap_values[mask, idx], "feature": fvals[mask]}
        records.append(record)

    if not records:
        raise ValueError("No finite SHAP values to plot.")

    fig, ax = plt.subplots(figsize=(8, max(len(records) * 0.4 + 1, 3)))
    cmap = plt.get_cmap("coolwarm")
    used_gradient = False

    def draw_neutral(vals, position):
        """A gray violin of vals at row position."""
        parts = ax.violinplot(
            [vals],
            positions=[position],
            orientation="horizontal",
            widths=0.38,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#6A6A6A")
            body.set_edgecolor("#444444")
            body.set_alpha(0.75)

    def draw_gradient(vals, fvals, position):
        """The colored violin of vals at row position; gray (draw_neutral) when fewer than two
        bins hold min_bin_count values or either range is flat."""
        nonlocal used_gradient
        x_min, x_max = np.nanmin(vals), np.nanmax(vals)
        f_min, f_max = np.nanpercentile(fvals, [5, 95])
        if x_min == x_max or f_min == f_max:
            draw_neutral(density_values(vals), position)
            return

        n_bins = min(60, max(12, vals.size // min_bin_count))
        edges = np.linspace(x_min, x_max, n_bins + 1)
        counts, _ = np.histogram(vals, bins=edges)
        valid = counts >= min_bin_count
        if valid.sum() < 2:
            draw_neutral(density_values(vals), position)
            return

        smooth_counts = np.convolve(counts.astype(float), [0.25, 0.5, 0.25], mode="same")
        widths = smooth_counts / smooth_counts[valid].max() * 0.38
        for i in np.where(valid)[0]:
            left, right = edges[i], edges[i + 1]
            if i == len(counts) - 1:
                in_bin = (vals >= left) & (vals <= right)
            else:
                in_bin = (vals >= left) & (vals < right)
            mean_f = np.nanmean(fvals[in_bin])
            color_pos = np.clip((mean_f - f_min) / (f_max - f_min), 0, 1)
            ax.fill_between(
                [left, right],
                position - widths[i],
                position + widths[i],
                color=cmap(color_pos),
                linewidth=0,
                alpha=0.9,
            )
        used_gradient = True

    for pos, record in enumerate(records):
        if record["feature"] is None:
            draw_neutral(record["shap"], pos)
        else:
            draw_gradient(record["shap"], record["feature"], pos)

    ax.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_yticks(np.arange(len(records)))
    ax.set_yticklabels([record["label"] for record in records])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value")
    if used_gradient:
        from matplotlib.patches import Patch
        ax.legend(
            handles=[
                Patch(facecolor=cmap(0.05), edgecolor="none", label="Low feature value"),
                Patch(facecolor=cmap(0.95), edgecolor="none", label="High feature value"),
            ],
            loc="lower right",
            frameon=False,
        )
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig
