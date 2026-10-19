# src/tracehr/probing/ml_classification.py
"""Binary-classification helpers of pipeline 05's comparison (run_downstream_comparison):
the learners, the metrics, and stratified cross-validation of a scikit-learn classifier or
an XGBoost booster, each then refit on all rows."""

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .cross_validation import cross_validate_sklearn, cross_validate_xgboost, print_cv_header

# The binary metrics every cross-validation fold records, in this order
CV_METRICS = ("auc_roc", "avg_precision", "f1_score", "sensitivity", "specificity")


def get_classifiers(
    class_weight_strategy: str = "balanced",
    random_state: int = 42,
    xgb_params: dict | None = None,
) -> dict[str, object]:
    """The binary learners of the comparison arms: LogisticRegression (weighted by
    class_weight_strategy; "none" turns the weighting off), and XGBoost as its parameter
    dict (xgb_params, or the defaults below; cross_validate_xgboost_classifier sets its
    class weight, scale_pos_weight)."""
    cw = class_weight_strategy if class_weight_strategy != "none" else None

    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight=cw,
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs',
            C=1.0
        ),
    }

    # XGBoost handled separately (different API)
    if xgb_params is None:
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            # L1 on leaf weights, on top of XGBoost's default L2 (lambda=1), aimed at the
            # bag-of-words arm, whose matrix is as wide as the vocabulary: alpha zeroes out
            # leaf weights whose gain does not clear the penalty, so splits on tokens carried
            # by a handful of patients stop paying for themselves. Every XGBoost arm gets it.
            # Note this penalises leaf values, not features: it does not select a sparse
            # feature set the way L1 does in a linear model.
            'alpha': 1.0,
            'seed': random_state
        }

    # Store XGBoost params for later use
    classifiers["XGBoost"] = xgb_params

    return classifiers


def classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Binary metrics of positive-class probabilities: the test split's full set and, through
    CV_METRICS, every cross-validation fold's."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # Both ranking metrics need both classes present. Left to sklearn the two disagree
    # about what to do when they are not: roc_auc_score returns NaN with a warning while
    # average_precision_score returns 0.0, which reads like a measured failure rather than
    # an absent measurement. NaN for both, so they can never tell different stories about
    # the same split; json_safe turns it into null on the way out.
    both = len(np.unique(y)) > 1
    return {
        "auc_roc": float(roc_auc_score(y, proba)) if both else float("nan"),
        "avg_precision": float(average_precision_score(y, proba)) if both else float("nan"),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_pos": int(y.sum()), "n_total": int(len(y)),
    }


def _stratified_folds(X, y: np.ndarray, n_folds: int, random_state: int,
                      groups: np.ndarray | None = None):
    """StratifiedKFold splits of X on y, or StratifiedGroupKFold when groups (patient ids) is
    given, which keeps each patient on one side of every fold."""
    skf = (StratifiedGroupKFold if groups is not None else StratifiedKFold)(
        n_splits=n_folds, shuffle=True, random_state=random_state)
    return skf.split(X, y, groups)


def _positive_proba(est, X) -> np.ndarray:
    """est's positive-class probabilities on X."""
    return est.predict_proba(X)[:, 1]


def cross_validate_sklearn_classifier(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int = 10,
    scale_features: bool = True,
    random_state: int = 42,
    groups: np.ndarray | None = None
) -> tuple[object, StandardScaler | None, dict[str, list[float]]]:
    """Stratified k-fold CV of a scikit-learn classifier (folds grouped by patient when
    groups is given), then a refit on all rows. Returns (classifier, scaler, cv_metrics)."""
    print_cv_header(model_name, n_folds)
    return cross_validate_sklearn(
        clf, X, y, model_name, _stratified_folds(X, y, n_folds, random_state, groups),
        predict=_positive_proba, fold_metrics=classification_metrics, metric_keys=CV_METRICS,
        headline=("auc_roc", "AUC"), n_folds=n_folds, scale_features=scale_features)


def cross_validate_xgboost_classifier(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_folds: int = 10,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
    scale_pos_weight: float | None = None,
    groups: np.ndarray | None = None
) -> tuple[xgb.Booster, dict[str, list[float]]]:
    """Stratified k-fold CV of an XGBoost classifier, then a refit on all rows. Without
    scale_pos_weight the class imbalance sets it (negatives / positives, printed)."""
    print_cv_header("XGBoost", n_folds)

    params = params.copy()

    if scale_pos_weight is None:
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0:
            scale_pos_weight = n_neg / n_pos
            print(f"  Auto scale_pos_weight: {scale_pos_weight:.2f}")
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight

    return cross_validate_xgboost(
        X, y, y, params, _stratified_folds(X, y, n_folds, random_state, groups), stratify=y,
        fold_metrics=classification_metrics, metric_keys=CV_METRICS, headline=("auc_roc", "AUC"),
        n_folds=n_folds, num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds, random_state=random_state)
