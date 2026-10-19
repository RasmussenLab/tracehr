# src/tracehr/probing/cross_validation.py
"""The cross-validation loops of pipeline 05's six arms: one for the scikit-learn-style
learners and one for XGBoost, shared by the binary, regression and survival tasks
(ml_classification, ml_regression and ml_survival supply the splitter, the labels and
the metric). Every arm prints the same lines and reports the same metrics per fold.

How the XGBoost loop picks a boosting-round count:

The obvious way - early-stop on the fold you are about to score - breaks the fold score
in two ways at once, and both are silent:

  1. Early stopping does not truncate the booster. When validation stops improving at
     round B, xgboost trains `early_stopping_rounds` more before giving up and merely
     records best_iteration=B, so the object holds B+21 trees. Since xgboost 2.0,
     predict() with no iteration_range uses ALL of them, so the fold is scored at the
     deliberately over-boosted end of training - while the model that finally ships has
     median(best_iteration) trees. The CV row and the test row then describe
     different-sized models. (Before 2.0, predict() honoured best_ntree_limit, so the
     same code reported different numbers - the bug was also version-dependent.)

  2. The stopping round was chosen to be the best point ON THE FOLD BEING SCORED, so the
     fold score is a best-of-300-checkpoints statistic rather than a held-out one. On a
     real cohort this was worth about +0.014 AUROC.

The two push in opposite directions, so the net bias has no predictable sign and cannot
be corrected after the fact.

fit_with_tuned_rounds fixes both by mirroring exactly what the final refit does: choose
the round count on data the scored fold never sees, then train on ALL of the fold's
training rows for exactly that many rounds. The scored booster therefore holds precisely
the tuned number of trees, which makes iteration_range unnecessary rather than merely
optional, and no scored row ever influenced its own stopping point.

It costs one extra fit per fold, but the second fit runs the tuned count (typically tens
of rounds) rather than the full num_boost_round, so the added time is well under 2x.
"""

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Below this many training rows a 20% holdout is too small to stop on; the fold then trains
# the full num_boost_round, which is at least honest - nothing is tuned on the scored rows.
_MIN_ROWS_FOR_HOLDOUT = 20


def fit_with_tuned_rounds(
    params: dict,
    X,
    label: np.ndarray,
    *,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
    stratify: np.ndarray | None = None,
    holdout_frac: float = 0.2,
) -> tuple[xgb.Booster, int]:
    """Train on all of (X, label) for a round count tuned on an inner holdout.

    Returns (booster, n_rounds). The booster holds exactly n_rounds trees, so a plain
    predict() on it is the tuned model - no iteration_range needed.

    `label` is what goes into the DMatrix (for survival that is the signed duration);
    `stratify` is what the inner split balances on (the class label, or the event
    indicator), which is a different array for survival and None for regression. A
    stratified split that cannot be made - a class with a single member - silently falls
    back to an unstratified one rather than failing the whole run.
    """
    n = X.shape[0]
    fit_idx = stop_idx = None
    if n >= _MIN_ROWS_FOR_HOLDOUT:
        idx = np.arange(n)
        for strat in (stratify, None):
            try:
                fit_idx, stop_idx = train_test_split(
                    idx, test_size=holdout_frac, random_state=random_state, stratify=strat)
                break
            except ValueError:
                fit_idx = stop_idx = None

    rounds = num_boost_round
    if fit_idx is not None and len(stop_idx) >= 2:
        probe = xgb.train(
            params, xgb.DMatrix(X[fit_idx], label=label[fit_idx]),
            num_boost_round=num_boost_round,
            evals=[(xgb.DMatrix(X[stop_idx], label=label[stop_idx]), "stop")],
            early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        # best_iteration is a 0-based index; +1 is the number of trees to keep. The old
        # code fed the index straight to the refit, so the shipped model was one tree
        # short of the one early stopping actually chose.
        rounds = int(probe.best_iteration) + 1

    booster = xgb.train(params, xgb.DMatrix(X, label=label), num_boost_round=rounds,
                        verbose_eval=False)
    return booster, rounds


def print_cv_header(model_name: str, n_folds: int) -> None:
    """The banner every cross-validation run opens with."""
    print(f"\n{'='*60}")
    print(f"Cross-validating {model_name} ({n_folds}-fold)...")
    print(f"{'='*60}")


def _print_cv_summary(cv_metrics: dict[str, list[float]]) -> None:
    """Print each cross-validation metric as its mean ± std over the folds."""
    for metric_name, values in cv_metrics.items():
        print(f"  CV {metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def cross_validate_sklearn(
    est,
    X,
    y: np.ndarray,
    model_name: str,
    folds: Iterable,
    *,
    predict: Callable,
    fold_metrics: Callable,
    metric_keys: Sequence[str],
    headline: tuple[str, str],
    n_folds: int,
    scale_features: bool,
) -> tuple[object, StandardScaler | None, dict[str, list[float]]]:
    """Cross-validate a scikit-learn-style estimator, then refit it on all of (X, y).

    folds yields (train_idx, val_idx) pairs. Each fold fits a clone of est (on standardized
    features when scale_features, the scaler fit on the fold's training rows only), scores
    the held-out rows with predict(est, X) and keeps fold_metrics(y_val, prediction)[k] for
    every k in metric_keys. headline = (metric key, label) names the per-fold line. Returns
    (refit estimator, its scaler or None, {metric: [one value per fold]}). The caller prints
    the header (print_cv_header) first.
    """
    cv_metrics = {k: [] for k in metric_keys}
    key, label = headline
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_v = X[train_idx], X[val_idx]
        y_tr, y_v = y[train_idx], y[val_idx]

        fold_est = clone(est)

        if scale_features:
            fold_scaler = StandardScaler()
            X_tr = fold_scaler.fit_transform(X_tr)
            X_v = fold_scaler.transform(X_v)

        fold_est.fit(X_tr, y_tr)
        scores = fold_metrics(y_v, predict(fold_est, X_v))
        for k in metric_keys:
            cv_metrics[k].append(scores[k])

        print(f"  Fold {fold_idx+1}/{n_folds}: {label}={cv_metrics[key][-1]:.4f}")

    _print_cv_summary(cv_metrics)

    # Refit on ALL training data
    print(f"\n  Refitting {model_name} on all {len(y)} training samples...")
    scaler = None
    X_final = X
    if scale_features:
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X)

    final_est = clone(est)
    final_est.fit(X_final, y)

    return final_est, scaler, cv_metrics


def cross_validate_xgboost(
    X,
    y: np.ndarray,
    labels: np.ndarray,
    params: dict,
    folds: Iterable,
    *,
    stratify: np.ndarray | None,
    fold_metrics: Callable,
    metric_keys: Sequence[str],
    headline: tuple[str, str],
    n_folds: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    random_state: int,
) -> tuple[xgb.Booster, dict[str, list[float]]]:
    """Cross-validate XGBoost, then refit it on all rows for the median tuned round count.

    labels is what goes into the DMatrix (the signed duration for survival, else y); y is
    what fold_metrics scores against; stratify is what each fold's inner early-stopping
    split balances on (or None). Each fold trains with fit_with_tuned_rounds, so no scored
    row ever chooses its own stopping point. The refit round count is floored at 1: a median
    of 0 would build a tree-less booster that predicts a constant and crashes SHAP's
    TreeExplainer. Returns (booster, {metric: [one value per fold]}). The caller prints the
    header (print_cv_header) and prepares params first.
    """
    cv_metrics = {k: [] for k in metric_keys}
    key, label = headline
    fold_rounds = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_v = X[train_idx], X[val_idx]

        dval = xgb.DMatrix(X_v, label=labels[val_idx])

        # The round count is tuned on a holdout inside THIS fold's training rows, then the
        # booster is trained on all of them for exactly that many rounds - see the header
        # for what early-stopping on the scored fold silently did to these numbers instead.
        bst, rounds = fit_with_tuned_rounds(
            params, X_tr, labels[train_idx], num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds, random_state=random_state,
            stratify=stratify[train_idx] if stratify is not None else None)
        fold_rounds.append(rounds)

        scores = fold_metrics(y[val_idx], bst.predict(dval))
        for k in metric_keys:
            cv_metrics[k].append(scores[k])

        print(f"  Fold {fold_idx+1}/{n_folds}: {label}={cv_metrics[key][-1]:.4f} "
              f"(rounds={rounds})")

    _print_cv_summary(cv_metrics)

    median_rounds = max(1, int(np.median(fold_rounds)))
    print(f"\n  Refitting XGBoost on all {len(y)} samples "
          f"(num_boost_round={median_rounds}, median of CV)...")

    dtrain_all = xgb.DMatrix(X, label=labels)
    final_bst = xgb.train(
        params, dtrain_all,
        num_boost_round=median_rounds,
        evals=[(dtrain_all, 'train')],
        verbose_eval=False
    )

    return final_bst, cv_metrics
