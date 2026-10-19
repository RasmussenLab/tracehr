# src/tracehr/probing/ml_regression.py
"""Regression counterparts of the three ml_classification helpers, used by
run_downstream_comparison when the outcome target is continuous. Same return shapes as
get_classifiers / cross_validate_sklearn_classifier / cross_validate_xgboost_classifier,
and the same signatures less the class weighting, so the comparison code differs only in
which trio it calls. Differences are inherent to the task: KFold instead of
StratifiedKFold (continuous y cannot be stratified), r2/mae/mse instead of
auc/ap/f1, and no class weighting.
"""

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from .cross_validation import cross_validate_sklearn, cross_validate_xgboost, print_cv_header

# The regression metrics every cross-validation fold records, in this order
CV_METRICS = ("r2", "mae", "mse")


def get_regressors(
    random_state: int = 42,
    xgb_params: dict | None = None
) -> dict[str, object]:
    """The regression learners: Ridge, and XGBoost as its parameter dict (xgb_params, or the
    defaults below). Ridge mirrors the classifiers' LogisticRegression(C=1.0): the same L2
    penalty, which the d_model-wide embedding features need."""
    regressors = {
        "Ridge": Ridge(
            alpha=1.0,
            random_state=random_state
        ),
    }

    # XGBoost handled separately (different API)
    if xgb_params is None:
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            # L1 on leaf weights; see get_classifiers for why the wide bag-of-words matrix
            # wants it and what it does (and does not) do.
            'alpha': 1.0,
            'seed': random_state
        }

    # Store XGBoost params for later use
    regressors["XGBoost"] = xgb_params

    return regressors


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Regression metrics (the same set the finetuning path reports for task="regression"):
    the test split's and, through CV_METRICS, every cross-validation fold's."""
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "mse": float(mean_squared_error(y, pred)),
        "n_total": int(len(y)),
    }


def _kfolds(X, y: np.ndarray, n_folds: int, random_state: int,
            groups: np.ndarray | None = None):
    """KFold splits of X, or GroupKFold when groups (patient ids) is given, which keeps each
    patient on one side of every fold (KFold ignores y and groups)."""
    kf = (GroupKFold if groups is not None else KFold)(
        n_splits=n_folds, shuffle=True, random_state=random_state)
    return kf.split(X, y, groups)


def _predict(est, X) -> np.ndarray:
    """est's predicted values on X."""
    return est.predict(X)


def cross_validate_sklearn_regressor(
    reg,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int = 10,
    scale_features: bool = True,
    random_state: int = 42,
    groups: np.ndarray | None = None
) -> tuple[object, StandardScaler | None, dict[str, list[float]]]:
    """k-fold CV of a scikit-learn regressor (folds grouped by patient when groups is given),
    then a refit on all rows. Returns (regressor, scaler, cv_metrics)."""
    print_cv_header(model_name, n_folds)
    return cross_validate_sklearn(
        reg, X, y, model_name, _kfolds(X, y, n_folds, random_state, groups),
        predict=_predict, fold_metrics=regression_metrics, metric_keys=CV_METRICS,
        headline=("r2", "R2"), n_folds=n_folds, scale_features=scale_features)


def cross_validate_xgboost_regressor(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_folds: int = 10,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
    groups: np.ndarray | None = None
) -> tuple[xgb.Booster, dict[str, list[float]]]:
    """k-fold CV of an XGBoost regressor, then a refit on all rows. The inner early-stopping
    split is not stratified: a continuous target cannot be."""
    print_cv_header("XGBoost", n_folds)
    return cross_validate_xgboost(
        X, y, y, params.copy(), _kfolds(X, y, n_folds, random_state, groups), stratify=None,
        fold_metrics=regression_metrics, metric_keys=CV_METRICS, headline=("r2", "R2"),
        n_folds=n_folds, num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds, random_state=random_state)
