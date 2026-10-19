# src/tracehr/probing/ml_survival.py
"""Survival counterparts of the ml_classification / ml_regression helpers, used by
run_downstream_comparison when the outcome is a censored time to event. Same return
shapes as their twins, so the comparison code differs only in which trio it calls.

Differences are inherent to the task:
  - the label is TWO columns, passed as y[:, 0] = duration, y[:, 1] = event (1/0).
    survival_core owns that contract; the stage-06 Cox head reads the same one.
  - folds are stratified on the EVENT indicator (not the duration), which keeps the
    event rate balanced across folds the way StratifiedKFold does for a class label.
  - the metric is Harrell's concordance index; there is no accuracy or R2 analogue.
  - the linear arm is a penalised Cox model (fit below with scipy), because scikit-learn
    has no survival estimator.

The label contract and the C-index live in survival_core, which pulls in neither xgboost
nor sklearn.
"""

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..finetuning.survival_core import split_survival_labels, survival_metrics
from .cross_validation import cross_validate_sklearn, cross_validate_xgboost, print_cv_header

# The survival metric every cross-validation fold records
CV_METRICS = ("cindex",)


def to_signed_labels(y: np.ndarray) -> np.ndarray:
    """XGBoost's survival:cox label encoding: +t for an event, -t for a censored row.

    Kept here so the signed form never escapes into the rest of the pipeline, which
    works exclusively with the explicit (duration, event) pair.

    THE ENCODING IS NOT INJECTIVE AT t=0. A sign cannot separate +0.0 from -0.0: they
    compare equal, and neither is < 0 or > 0, so no comparison-based reader can tell an
    event at the anchor from a censoring at it. XGBoost resolves both to CENSORED -
    measured, not assumed: a row labelled +0.0 trains a model bit-identical to the same
    row labelled -0.0 and to a plainly censored control, while at t=1e-6 the sign changes
    the fit as expected. A duration-0 EVENT is therefore discarded by the two XGBoost arms
    while CoxPH (which receives the explicit pair) and the C-index both keep it, so one
    patient carries two different labels across arms of the same comparison. Censoring at
    0 is unaffected - censored is what it already resolves to. Nothing inside this encoding
    can fix that; only durations bounded away from 0 avoid it, which is a property of how
    the outcomes table is built. detect_survival_task reports the duration-0 count.
    """
    durations, events = split_survival_labels(y)
    return np.where(events == 1, durations, -durations)


def fit_cox_l2(X: np.ndarray, durations: np.ndarray, events: np.ndarray,
               penalizer: float) -> np.ndarray:
    """Coefficients of a Cox model with Efron's tie handling and an L2 penalty.

    The objective is lifelines' CoxPHFitter(penalizer) one, so the fit lands on the same
    coefficients: the columns standardised (mean 0, sample std 1), the log partial
    likelihood minus n * penalizer / 2 * ||beta||^2 maximised, beta scaled back to the
    original columns. The penalty makes it strictly concave, so the optimum is unique
    whatever the optimiser; L-BFGS needs only the gradient, which keeps memory at the size
    of X even for a d_model-wide embedding. X must have no constant column.
    """
    std = X.std(axis=0, ddof=1)
    n = len(durations)
    order = np.argsort(durations, kind="stable")
    Z, T, died = ((X - X.mean(axis=0)) / std)[order], durations[order], events[order] == 1

    # Event times t_k with d_k tied deaths. Efron's correction gives each death of a tie its
    # own denominator, S0(risk set) - (l / d_k) * S0(the tied deaths), l = 0..d_k-1, so the
    # deaths are expanded into one row each, grouped by event time (k_rep, frac = l / d_k).
    event_times, n_tied = np.unique(T[died], return_counts=True)
    n_times = len(event_times)
    risk_start = np.searchsorted(T, event_times, side="left")   # risk set k = rows risk_start[k]:
    k_of_death = np.searchsorted(event_times, T[died])
    k_rep = np.repeat(np.arange(n_times), n_tied)
    frac = (np.arange(len(k_rep)) - np.repeat(np.cumsum(n_tied) - n_tied, n_tied)) / n_tied[k_rep]
    times_upto = np.searchsorted(event_times, T, side="right")   # event times <= each row's T
    z_died_sum = Z[died].sum(axis=0)

    def objective(beta):
        """(-penalised log partial likelihood / n, its gradient)."""
        eta = Z @ beta
        shift = eta.max()                                        # exp without overflow
        w = np.exp(eta - shift)
        s0_risk = np.cumsum(w[::-1])[::-1][risk_start]
        s0_died = np.bincount(k_of_death, weights=w[died], minlength=n_times)
        den = s0_risk[k_rep] - frac * s0_died[k_rep]
        loglik = eta[died].sum() - (np.log(den) + shift).sum()
        # Gradient: sum over deaths of z minus, per event time, the weighted mean z of its
        # risk set (Efron-corrected). Collected per row so it is one matrix-vector product.
        a = np.bincount(k_rep, weights=1.0 / den, minlength=n_times)
        b = np.bincount(k_rep, weights=frac / den, minlength=n_times)
        row = w * np.r_[0.0, np.cumsum(a)][times_upto]
        row[died] -= w[died] * b[k_of_death]
        grad = z_died_sum - Z.T @ row
        return -loglik / n + penalizer / 2 * beta @ beta, -grad / n + penalizer * beta

    res = minimize(objective, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 10_000, "gtol": 1e-9, "ftol": 1e-15})
    # L-BFGS-B can stop at the floating-point limit with a line-search "failure" after it
    # has in fact converged, so the gradient decides.
    if not res.success and np.abs(res.jac).max() > 1e-6:
        raise RuntimeError(
            f"The Cox fit did not converge on a {X.shape[0]}x{X.shape[1]} matrix with "
            f"{int(died.sum())} events ({res.message}). Collinear feature columns are the "
            f"usual cause; a larger penalizer (currently {penalizer}) is the usual fix.")
    return res.x / std


class CoxPHRegressor(BaseEstimator):
    """A penalised Cox model (fit_cox_l2) behind the scikit-learn estimator interface.

    Subclasses BaseEstimator because the shared CV helper calls sklearn's clone(),
    which needs get_params/set_params. Exposes coef_ and intercept_ so the existing
    linear SHAP path in shap_explain.py works on this arm unchanged - a Cox
    model has no intercept, so it is fixed at 0.0 and predict() is exactly the linear
    predictor those SHAP values decompose.

    `penalizer` is load-bearing rather than cosmetic: an unpenalised Cox fit on a
    d_model-wide embedding with few events does not converge. It mirrors the
    Ridge(alpha=1.0) / LogisticRegression(C=1.0) the other two learners use for the
    same reason.
    """

    def __init__(self, penalizer: float = 0.1):
        """penalizer is the L2 penalty weight (see fit_cox_l2)."""
        self.penalizer = penalizer

    def fit(self, X, y):
        """Fit the Cox model on X and the [N, 2] (duration, event) labels y; a constant
        column is left out of the fit with coefficient 0. Returns self."""
        durations, events = split_survival_labels(y)
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)

        # A constant column cannot be standardised (its std is 0, so the fit would see NaN),
        # and one reaches this fit in ordinary cohorts: Sex in a single-sex study, or an
        # embedding dimension that never varies after pooling. LogisticRegression and Ridge
        # absorb such a column silently, so the survival arms must too, or the Sex+Age
        # baseline every lift is measured against takes the whole run down. Dropped columns
        # keep a coefficient of exactly 0, which is both the truth and what preserves the
        # coef_[j] <-> column j contract that the linear SHAP values rely on.
        keep = X.std(axis=0) > 0
        if not keep.any():
            raise ValueError(
                f"Every one of the {X.shape[1]} feature columns is constant, so a Cox model "
                f"has nothing to fit. Check that this arm's feature matrix was built from "
                f"the intended columns.")
        if not keep.all():
            print(f"  [survival] {int((~keep).sum())}/{len(keep)} feature column(s) are "
                  f"constant and were dropped from the Cox fit (coefficient 0): "
                  f"{np.flatnonzero(~keep)[:10].tolist()}")

        # Full width with zeros in the dropped positions, so coef_[j] always belongs to
        # column j of X and predict() stays a plain dot product over the original matrix. A
        # fit that does not converge raises rather than silently bumping the penalizer: a
        # different penalty on one arm would make the six arms incomparable, which is the
        # entire point of the comparison.
        self.coef_ = np.zeros(X.shape[1], dtype=np.float64)
        self.coef_[keep] = fit_cox_l2(X[:, keep], durations, events, self.penalizer)
        self.intercept_ = 0.0
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X) -> np.ndarray:
        """Log relative risk of X (the linear predictor; a Cox model has no intercept)."""
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


def get_survival_models(
    random_state: int = 42,
    penalizer: float = 0.1,
    xgb_params: dict | None = None,
) -> dict[str, object]:
    """The survival learners: the penalised CoxPHRegressor, and XGBoost (survival:cox) as its
    parameter dict (xgb_params, or the defaults below). The same {named estimator,
    "XGBoost": params dict} contract as get_classifiers and get_regressors, so
    run_downstream_comparison unpacks all three identically."""
    models = {
        "CoxPH": CoxPHRegressor(penalizer=penalizer),
    }

    # XGBoost handled separately (different API)
    if xgb_params is None:
        xgb_params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            # L1 on leaf weights; see get_classifiers for why the wide bag-of-words
            # matrix wants it and what it does (and does not) do.
            'alpha': 1.0,
            'seed': random_state
        }

    models["XGBoost"] = xgb_params

    return models


def _event_folds(X, y: np.ndarray, n_folds: int, random_state: int,
                 groups: np.ndarray | None = None):
    """Folds stratified on the event indicator: the survival analogue of stratifying on a
    class label. Without it a fold can draw few or no events and its C-index is noise.
    groups (patient ids) switches to StratifiedGroupKFold, which keeps each patient on one
    side of every fold."""
    _, events = split_survival_labels(y)
    skf = (StratifiedGroupKFold if groups is not None else StratifiedKFold)(
        n_splits=n_folds, shuffle=True, random_state=random_state)
    return skf.split(X, events, groups)


def _predict(est, X) -> np.ndarray:
    """est's risk scores on X."""
    return est.predict(X)


def cross_validate_sklearn_survival(
    est,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int = 10,
    scale_features: bool = True,
    random_state: int = 42,
    groups: np.ndarray | None = None
) -> tuple[object, StandardScaler | None, dict[str, list[float]]]:
    """k-fold CV of a scikit-learn-compatible survival model (folds stratified on the event,
    grouped by patient when groups is given), then a refit on all rows. Returns
    (model, scaler, cv_metrics)."""
    print_cv_header(model_name, n_folds)
    return cross_validate_sklearn(
        est, X, y, model_name, _event_folds(X, y, n_folds, random_state, groups),
        predict=_predict, fold_metrics=survival_metrics, metric_keys=CV_METRICS,
        headline=("cindex", "C-index"), n_folds=n_folds, scale_features=scale_features)


def cross_validate_xgboost_survival(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_folds: int = 10,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
    groups: np.ndarray | None = None
) -> tuple[xgb.Booster, dict[str, list[float]]]:
    """k-fold CV of an XGBoost survival:cox model, then a refit on all rows. survival:cox
    reads one signed label column (to_signed_labels); the inner early-stopping split
    balances on the event indicator, the quantity the outer folds stratify on. The booster
    predicts the hazard ratio exp(margin), monotone in the risk score, so the C-index is
    unaffected by the exponential."""
    print_cv_header("XGBoost", n_folds)

    params = params.copy()

    # survival:cox reads one signed column: +t for an event, -t for a censored row.
    signed = to_signed_labels(y)
    _, all_events = split_survival_labels(y)

    return cross_validate_xgboost(
        X, y, signed, params, _event_folds(X, y, n_folds, random_state, groups),
        stratify=all_events, fold_metrics=survival_metrics, metric_keys=CV_METRICS,
        headline=("cindex", "C-index"), n_folds=n_folds, num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds, random_state=random_state)
