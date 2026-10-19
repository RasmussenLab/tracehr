# src/tracehr/probing/downstream_comparison.py
"""Pipeline 05's comparison: six models trained on the ORIGINAL train+val patients and
evaluated on the ORIGINAL test patients, from the frozen transformer's embeddings
(anchor_embeddings). Demographics mean [Sex, age at the last token]; BoW means per-token
counts over the full truncated history:
  1. Demographics              -> Linear model (LR)    (the floor)
  2. Embedding                 -> Linear model (LR)    (linear probe)
  3. Embedding + demographics  -> Linear model (LR)
  4. Embedding                 -> XGBoost
  5. Embedding + demographics  -> XGBoost
  6. BoW + demographics        -> XGBoost

The grid is what makes the comparison readable: 5 vs 6 differ only in the representation
(same learner, same params, same demographic block), and each "+demographics" arm against
its bare twin gives the demographic increment for that learner.

Three tasks share those six arms:
  binary      target is 0/1        -> LogisticRegression / XGBoost-logistic, AUROC/AUPRC/F1
  regression  target is continuous -> Ridge / XGBoost-squarederror,          R2/MAE/MSE
  survival    censored time to event -> penalised CoxPH / XGBoost-survival:cox, C-index

Binary and regression are autodetected from the target values by detect_target_task.
Survival cannot be: its label is TWO columns, so it is selected by naming
anchor_columns.censor_timestamp_col, and then target_col carries the 0/1 event indicator
while the duration - the days from the anchor to the censor timestamp (the event or the
censoring), computed by load_anchors - travels as "duration". Internally that pair travels
as a [N, 2] array ordered (duration, event) - the contract survival_core owns and the
stage-06 Cox head reads. Both stages report their C-index through
survival_core.survival_metrics, so the two numbers are directly comparable.

All three tasks keep the same six model names, result shape, SHAP artifacts and output
filenames, and all report every arm on cross-validation (shared folds) as well as on the
held-out test split. The transformer is a fixed feature extractor here (no gradient);
this is the classic "embed then fit an external classifier" setup, distinct from the
neural fine-tuning probe of pipeline 06.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import precision_recall_curve, roc_curve

from ..finetuning.survival_core import km_by_risk_panel, split_survival_labels, survival_metrics
from ..generation.model_io import json_safe
from .ml_classification import (
    classification_metrics,
    cross_validate_sklearn_classifier,
    cross_validate_xgboost_classifier,
    get_classifiers,
)
from .ml_regression import (
    cross_validate_sklearn_regressor,
    cross_validate_xgboost_regressor,
    get_regressors,
    regression_metrics,
)

from .ml_survival import (
    cross_validate_sklearn_survival,
    cross_validate_xgboost_survival,
    get_survival_models,
)
from .shap_explain import explain_downstream_models, save_downstream_models

# One color per arm (display name), in arm order, so every figure draws an arm in the same
# color; a name missing here is drawn gray.
MODEL_COLORS = {
    "Sex+Age + LR":                "#276abd",  # blue
    "Embedding + LR":              "#ffff00",  # yellow
    "Embedding+Sex+Age + LR":      "#e69d01",  # orange
    "Embedding + XGBoost":         "#b50000",  # red
    "Embedding+Sex+Age + XGBoost": "#e87ba4",  # magenta
    "BoW+Sex+Age + XGBoost":       "#007800",  # green
}


# ============================================================================ #
# Feature builders for the non-embedding blocks
# ============================================================================ #
def bag_of_words_matrix(token_id_lists: list[list[int]], count_lists: list[list[int]],
                        vocab_size: int, exclude_ids: set[int] | None = None
                        ) -> sparse.csr_matrix:
    """Count matrix [N, vocab_size]: how often each token id occurs in the patient's full
    truncated history. Callers pass ids; extract_anchor_embeddings' bow_tokens column holds
    the matching token NAMES, so it is mapped back to ids before reaching here.

    CSR, not dense: a patient touches a few hundred distinct tokens out of the whole
    vocabulary, so the dense form is ~99% zeros and becomes unallocatable at cohort scale
    (5M patients x 20k vocab = 400 GB). XGBoost's DMatrix consumes CSR natively.

    A count of 0 means the token never occurs in the patient's history, so leaving it
    unstored says exactly that. XGBoost's sparsity-aware split finding sums the gradients
    of the absent rows and learns which side they belong on - absence is used as evidence,
    not imputed and not dropped - so nothing is lost relative to a dense matrix.

    The fitted trees do differ from a dense build's, because dense presents 0 as a measured
    value and splits on a threshold instead. Both reach the same partitions; the greedy
    search just takes a different route, so BoW numbers from a dense build are not
    comparable with these. A dense build is xgb.DMatrix(X.toarray()) (or equivalently
    xgb.DMatrix(dense, missing=0.0) to go the other way).

    The caller excludes the window-artifact specials (<PAD>/<SOS>/<CON>/<MASK>), which can
    never occur in a patient's history anyway, plus the sex tokens, which the demographic
    block carries for every arm. <UNK> is kept: it is the one special that does occur, and
    its count says how many of the patient's events fell outside the vocabulary - a genuine
    signal (record completeness, coding drift), not an artifact.
    """
    exclude_ids = exclude_ids or set()
    indptr = np.zeros(len(token_id_lists) + 1, dtype=np.int64)
    indices, data = [], []
    for i, (ids, counts) in enumerate(zip(token_id_lists, count_lists)):
        keep = [(int(t), float(c)) for t, c in zip(ids, counts)
                if t not in exclude_ids and 0 <= t < vocab_size]
        indices.extend(t for t, _ in keep)
        data.extend(c for _, c in keep)
        indptr[i + 1] = len(indices)
    X = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32), indptr),
        shape=(len(token_id_lists), vocab_size))
    # A repeated id would sum rather than overwrite; np.unique upstream makes ids distinct,
    # and summing duplicates is the correct count semantics either way.
    X.sum_duplicates()
    return X


def demographic_matrix(df: pd.DataFrame, sex_categories: list[str] | None = None) -> np.ndarray:
    """[N, 2] = [sex code, age at last token].

    Sex is encoded by its position in `sex_categories` (the distinct Sex token strings
    observed across the cohort), so the mapping is language/coding agnostic and identical
    for train and test: the first two categories -> 0.0 / 1.0, anything else or unknown
    -> 0.5. When sex_categories is None it falls back to the values observed in `df`.
    """
    if sex_categories is None:
        sex_categories = sorted(df["Sex"].dropna().unique().tolist())
    code = {c: float(i) for i, c in enumerate(sex_categories[:2])}
    sex01 = df["Sex"].map(lambda s: code.get(s, 0.5))
    return np.column_stack([sex01.to_numpy(dtype=np.float32),
                            df["last_token_age"].to_numpy(dtype=np.float32)])


def dense_to_stored_csr(A: np.ndarray) -> sparse.csr_matrix:
    """Dense [N, d] -> CSR with EVERY entry stored, zeros included.

    sparse.hstack routes a dense block through COO, which drops zeros - and an unstored CSR
    entry reads as MISSING to XGBoost. That is the right meaning for an absent token count
    and the wrong one for a measured demographic: sex code 0.0 is the first category (about
    half the cohort) and age 0.0 is a real age. Storing every row keeps them values. A NaN
    stays NaN and is still correctly read as missing.
    """
    n, d = A.shape
    return sparse.csr_matrix(
        (A.astype(np.float32).ravel(), np.tile(np.arange(d, dtype=np.int32), n),
         np.arange(0, n * d + 1, d, dtype=np.int64)), shape=(n, d))


# ============================================================================ #
# Task autodetection
# ============================================================================ #
def detect_target_task(values, label: str = "target", source: str | None = None) -> str:
    """Decide "binary" vs "regression" from the outcome values and print the evidence.

    Binary = every finite value is 0 or 1; anything else is a continuous target. NaN is
    rejected here (before the expensive embedding extraction) because no estimator accepts
    a NaN label; integer-looking class codes are flagged loudly since they are numerically
    indistinguishable from a small-count regression target.
    """
    y = np.asarray(values, dtype=np.float64).ravel()
    where = f" in {source}" if source else ""
    n_nan = int(np.isnan(y).sum())
    if n_nan:
        raise ValueError(
            f"Outcome column '{label}'{where} has {n_nan} of {len(y)} values missing (NaN). "
            f"No estimator accepts NaN targets: drop or impute those patients before running "
            f"the downstream stage.")

    uniq = np.unique(y)
    if np.isin(uniq, (0.0, 1.0)).all():
        print(f"[downstream] task=binary (autodetected from '{label}': values "
              f"{{{', '.join(f'{v:g}' for v in uniq)}}}, {int(y.sum())}/{len(y)} positive)")
        return "binary"

    print(f"[downstream] task=regression (autodetected from '{label}': {len(y)} values, "
          f"{len(uniq)} distinct, range [{uniq[0]:.4g}, {uniq[-1]:.4g}], mean {y.mean():.4g})")
    if len(uniq) <= 10 and (uniq >= 0).all() and np.equal(np.mod(uniq, 1), 0).all():
        print("\n" + "!" * 78)
        print(f"!! WARNING: '{label}' looks like integer class codes "
              f"({{{', '.join(f'{v:g}' for v in uniq)}}}) but is being treated as a")
        print("!! CONTINUOUS target: the models are fit by regression, not classification.")
        print("!! That is correct for a count/score outcome. For multiclass classification")
        print("!! use pipeline 06 (finetuning.task: multiclass) instead.")
        print("!" * 78 + "\n")
    return "regression"


# ============================================================================ #
# Train the six arms and evaluate on cross-validation and the test split
# ============================================================================ #
def _predict(kind: str, task: str, model, scaler, X: np.ndarray) -> np.ndarray:
    """Test-split output of one fitted arm: binary -> positive-class probability;
    regression -> predicted value; survival -> relative risk. The booster needs no branch:
    its objective already decides which it returns (for survival:cox, the hazard ratio,
    monotone in the risk score)."""
    if kind == "xgb":
        import xgboost as xgb
        return model.predict(xgb.DMatrix(X))
    X_ = scaler.transform(X) if scaler is not None else X
    # Only binary has predict_proba; regression and survival both return predict().
    return (model.predict_proba(X_)[:, 1] if task == "binary"
            else model.predict(X_))


def _fold_cap(n_folds: int, n_minority: int, what: str) -> int:
    """Lower n_folds until every fold can draw at least one of the scarce label.

    Both stratified splitters spread the scarce label as evenly as they can, so
    n_folds <= n_minority is exactly the condition under which no fold comes back with an
    undefined metric. Decided once per run from the shared training labels: all six arms
    must cross-validate on identical folds, and capping per-arm would break the
    fold-by-fold pairing the comparison rests on.

    Returns n_folds unchanged - the common case - whenever the labels are not that scarce.
    """
    if n_minority >= n_folds:
        return n_folds
    reduced = max(2, n_minority)
    print(f"[downstream] only {n_minority} {what}: cross-validating on {reduced} folds "
          f"instead of {n_folds}, so no fold is left with an undefined metric.")
    return reduced


class _Learners(NamedTuple):
    """The task-matched learner pair and metric function every arm shares."""
    lr_proto: object          # the linear learner (cloned by the CV helper before every fit)
    xgb_params: dict          # the XGBoost parameters (copied by the CV helper before use)
    cv_linear: Callable       # cross-validates, then refits, the linear learner
    cv_tree: Callable         # cross-validates, then refits, XGBoost
    metric_fn: Callable       # test-split metrics of one arm's output


def _describe_labels(y: np.ndarray, task: str) -> str:
    """One-line summary of a split's labels, for the task line."""
    if task == "regression":
        return f"mean={y.mean():.4g}, range [{y.min():.4g}, {y.max():.4g}]"
    if task == "survival":
        d, e = split_survival_labels(y)
        return (f"events={int(e.sum())}/{len(e)} ({e.mean():.1%}), "
                f"median duration={np.median(d):.4g}")
    return f"pos={int(y.sum())}"


def _split_labels(tr: pd.DataFrame, te: pd.DataFrame, emb_df: pd.DataFrame,
                  task: str) -> tuple[np.ndarray, np.ndarray]:
    """(y_train, y_test), with the task line printed. Survival has two labels per patient:
    column order [duration, event] is the contract ml_survival and the stage-06 head both
    read; target_col carries the event."""
    if task == "survival":
        if "duration" not in emb_df.columns:
            raise KeyError(
                "task='survival' needs a 'duration' column in the embedding frame; "
                "pass outcome_duration_col to extract_anchor_embeddings (config: "
                "anchor_columns.censor_timestamp_col).")
        y_tr = tr[["duration", "target"]].to_numpy(dtype=np.float64)
        y_te = te[["duration", "target"]].to_numpy(dtype=np.float64)
    else:
        y_tr, y_te = tr["target"].to_numpy(), te["target"].to_numpy()
    print(f"[downstream] task={task}: train anchors={len(tr)} ({_describe_labels(y_tr, task)}), "
          f"test anchors={len(te)} ({_describe_labels(y_te, task)})")
    return y_tr, y_te


def _task_learners(task: str, random_state: int) -> _Learners:
    """The task-matched estimator pair and metric function; everything downstream of this is
    shared by the three tasks."""
    if task == "regression":
        protos = get_regressors(random_state=random_state)
        return _Learners(protos["Ridge"], protos["XGBoost"], cross_validate_sklearn_regressor,
                         cross_validate_xgboost_regressor, regression_metrics)
    if task == "survival":
        protos = get_survival_models(random_state=random_state)
        return _Learners(protos["CoxPH"], protos["XGBoost"], cross_validate_sklearn_survival,
                         cross_validate_xgboost_survival, survival_metrics)
    protos = get_classifiers(random_state=random_state)
    return _Learners(protos["Logistic Regression"], protos["XGBoost"],
                     cross_validate_sklearn_classifier, cross_validate_xgboost_classifier,
                     classification_metrics)


def _capped_folds(task: str, n_folds: int, train_ids: pd.Series, y_tr: np.ndarray) -> int:
    """n_folds, lowered (_fold_cap) when the scarce label cannot reach every fold.
    Regression folds are never capped."""
    if task == "survival":
        # A fold with no events has no comparable pairs, so its C-index is undefined.
        # The folds are stratified on the event indicator, so one event per fold is
        # guaranteed as long as there are at least n_folds of them. Counted in patients:
        # grouped folds can only move whole patients (one anchor each: the same number).
        return _fold_cap(n_folds, train_ids[split_survival_labels(y_tr)[1] == 1].nunique(),
                         "patients with an event in the training split")
    if task == "regression":
        return n_folds
    # Same argument for a class label: AUROC needs both classes in every fold, and
    # StratifiedKFold can only deliver that with at least n_folds of the rarer one.
    # Counted in patients, as above.
    n_pos = train_ids[y_tr == 1].nunique()
    n_neg = train_ids[y_tr != 1].nunique()
    return _fold_cap(n_folds, min(n_pos, n_neg),
                     "patients of the rarer class in the training split")


def _feature_arms(tr: pd.DataFrame, te: pd.DataFrame, emb_df: pd.DataFrame, itos: list[str],
                  vocab_size: int, special_ids: set[int]) -> list[tuple]:
    """The six arms as (display name, train X, test X, learner, SHAP feature names), built
    from the embeddings, the demographic block and the bag-of-words counts (train + test
    share identical patient rows/order within each split)."""
    X_emb_tr = np.stack(tr["Embedding"].to_numpy())
    X_emb_te = np.stack(te["Embedding"].to_numpy())
    # Encode Sex by its distinct token strings across BOTH splits, so the demographic
    # baseline works for any sex vocabulary (e.g. "mand"/"kvinde"), not just "male"/"female",
    # and train/test share one mapping. sorted(...) codes "female"/"male" as 0/1.
    sex_categories = sorted(emb_df["Sex"].dropna().unique().tolist())
    if len(sex_categories) < 2:
        print(f"[downstream] WARNING: Sex has <2 distinct values {sex_categories}; "
              f"the Sex+Age baseline is effectively age-only.")
    elif len(sex_categories) > 2:
        print(f"[downstream] WARNING: Sex has >2 distinct values {sex_categories}; "
              f"only the first two are coded 0/1, the rest map to 0.5.")
    X_demo_tr = demographic_matrix(tr, sex_categories)
    X_demo_te = demographic_matrix(te, sex_categories)
    # Token counts only: sex and age reach every arm through the demographic block below,
    # so the sex tokens are dropped from the counts. Carrying sex twice - once as a count
    # column, once as a code - would split both the trees' splits and their SHAP attribution
    # across two encodings of one fact. A sex string absent from the vocabulary excludes
    # nothing.
    tok_id = {t: i for i, t in enumerate(itos)}
    bow_exclude = special_ids | {tok_id[s] for s in sex_categories if s in tok_id}

    # bow_tokens holds token NAMES so the saved parquet is readable without a vocabulary;
    # bag_of_words_matrix keeps its integer-id contract (it owns the CSR semantics), so the
    # names are mapped back here. A name outside this vocabulary means the frame and the
    # vocabulary disagree - the encoder turns unknown events into <UNK>, never into an
    # unlisted name - so count them rather than dropping them quietly.
    unmapped = set()

    def _bow(df):
        """Bag-of-words CSR matrix of a split's bow_tokens/bow_token_counts, the names
        mapped to their ids; a name outside the vocabulary is collected in unmapped and
        left out."""
        ids, counts = [], []
        for toks, cs in zip(df["bow_tokens"], df["bow_token_counts"]):
            unmapped.update(t for t in toks if t not in tok_id)
            keep = [(tok_id[t], int(c)) for t, c in zip(toks, cs) if t in tok_id]
            ids.append([i for i, _ in keep])
            counts.append([c for _, c in keep])
        return bag_of_words_matrix(ids, counts, vocab_size, bow_exclude)

    X_bow_tr, X_bow_te = _bow(tr), _bow(te)
    if unmapped:
        print(f"[downstream] WARNING: {len(unmapped)} bow_tokens are absent from the "
              f"vocabulary and were dropped, e.g. {sorted(unmapped)[:5]}. The embeddings "
              f"were probably built against a different vocabulary than the one passed here.")
    # How much out-of-vocabulary mass the <UNK> column carries is data-dependent and decides
    # whether that feature means anything, so report it rather than leaving it implicit.
    if "<UNK>" in itos:
        unk_col = X_bow_tr[:, itos.index("<UNK>")]
        n_unk = unk_col.nnz
        print(f"[downstream] BoW <UNK> column: {n_unk}/{X_bow_tr.shape[0]} train patients have "
              f">=1 out-of-vocabulary event"
              + (f" (mean {unk_col.sum() / n_unk:.1f} each)" if n_unk else ""))

    # Embedding + demographics, and BoW + demographics. Both are built once and shared by
    # their LR and XGBoost arms, so the two learners see byte-identical features. The demo
    # block is converted to stored-CSR rather than passed dense: sparse.hstack would route a
    # dense block through COO and silently drop its zeros (see dense_to_stored_csr).
    X_ed_tr = np.hstack([X_emb_tr, X_demo_tr])
    X_ed_te = np.hstack([X_emb_te, X_demo_te])
    X_bd_tr = sparse.hstack([X_bow_tr, dense_to_stored_csr(X_demo_tr)], format="csr")
    X_bd_te = sparse.hstack([X_bow_te, dense_to_stored_csr(X_demo_te)], format="csr")

    emb_names = [f"emb_{i}" for i in range(X_emb_tr.shape[1])]
    demo_names = ["Sex", "last_token_age"]
    bow_names = [itos[i] for i in range(vocab_size)]
    # One row per arm: this is the only place a model's features are assembled and _fit_arms
    # the only place a model is fit and a SHAP input registered, so an arm can never be
    # explained on a matrix it was not trained on. Order is the legend/panel/table order,
    # and the order MODEL_COLORS assigns its palette slots along.
    #
    # The pairs that carry the comparison: arm 5 vs arm 6 differ only in the representation
    # (same learner, same params, same demographic block), and each "+Sex+Age" arm against
    # its bare twin gives the demographic increment for that learner.
    return [
        ("Sex+Age + LR",                X_demo_tr, X_demo_te, "lr",  demo_names),
        ("Embedding + LR",              X_emb_tr,  X_emb_te,  "lr",  emb_names),
        ("Embedding+Sex+Age + LR",      X_ed_tr,   X_ed_te,   "lr",  emb_names + demo_names),
        ("Embedding + XGBoost",         X_emb_tr,  X_emb_te,  "xgb", emb_names),
        ("Embedding+Sex+Age + XGBoost", X_ed_tr,   X_ed_te,   "xgb", emb_names + demo_names),
        ("BoW+Sex+Age + XGBoost",       X_bd_tr,   X_bd_te,   "xgb", bow_names + demo_names),
    ]


def _fit_arms(arms: list[tuple], task: str, learners: _Learners, y_tr: np.ndarray,
              y_te: np.ndarray, *, n_folds: int, random_state: int,
              groups: np.ndarray | None) -> tuple[dict[str, dict], dict[str, tuple]]:
    """(results, SHAP inputs) per arm: cross-validated and refit on the training split, then
    scored on the test split.

    "proba" holds the model's test-set output: positive-class probability for binary,
    predicted value for regression. Every arm gets the same n_folds and random_state, and
    the splitters key off y (stratified) or n_samples (KFold) but never off X - so all six
    are cross-validated on byte-identical folds and fold k is comparable across arms. Do
    not reseed per arm; that would break the pairing. lr_proto and xgb_params are reused
    safely: the sklearn helpers clone() before every fit, and both XGBoost helpers copy()
    the params before touching them. scale_features is load-bearing for the LR arms -
    last_token_age is O(1e4) days against O(1) embedding components, so unscaled the L2
    penalty would see the age column alone.
    """
    results: dict[str, dict] = {}
    shap_inputs: dict[str, tuple] = {}
    for name, Xtr, Xte, learner, feat_names in arms:
        if learner == "lr":
            est, scaler, cv = learners.cv_linear(learners.lr_proto, Xtr, y_tr, name, n_folds=n_folds,
                                                 scale_features=True, random_state=random_state,
                                                 groups=groups)
            kind = "linear"
        else:
            est, cv = learners.cv_tree(Xtr, y_tr, learners.xgb_params, n_folds=n_folds,
                                       random_state=random_state, groups=groups)
            scaler, kind = None, "tree"
        p = _predict(learner, task, est, scaler, Xte)
        results[name] = {"metrics": learners.metric_fn(y_te, p), "cv": cv, "proba": p, "y": y_te,
                         "model": est, "scaler": scaler, "kind": kind}
        shap_inputs[name] = (Xtr, Xte, feat_names)
    return results, shap_inputs


def _write_metrics(results: dict[str, dict], output_dir: Path) -> None:
    """downstream_metrics.json: each arm's test metrics and every CV metric (not just the
    headline) with its raw per-fold scores - the folds are shared across arms, so keeping
    them is what lets two arms be compared fold-by-fold later without re-running anything."""
    metrics_out = {name: {"test": r["metrics"],
                          "cv": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                                     "folds": [float(x) for x in v]}
                                 for k, v in r["cv"].items()}}
                   for name, r in results.items()}
    with open(output_dir / "downstream_metrics.json", "w") as f:
        # json_safe: an undefined metric is NaN in numpy and null on disk. json.dump would
        # otherwise write a bare NaN, which no strict JSON parser will read back.
        json.dump(json_safe(metrics_out), f, indent=2)


def _plot_results(results: dict[str, dict], task: str, y_te: np.ndarray, output_dir: Path) -> None:
    """The task's comparison figures. The CV bars carry the fold spread as error bars, which
    is what says whether a gap between two arms is bigger than fold-to-fold noise; the test
    bars are the held-out read of the same comparison."""
    if task == "regression":
        _plot_pred_vs_true(results, output_dir / "downstream_pred_vs_true.png")
        r2_only = [("r2", "R²")]
        _plot_metric_bars(results, output_dir / "downstream_metric_bars.png",
                          metrics=r2_only, ylim=None)
        _plot_metric_bars(results, output_dir / "downstream_metric_bars_cv.png",
                          metrics=r2_only, ylim=None, source="cv")
    elif task == "survival":
        # ROC/PR/confusion are binary-only and pred-vs-true is regression-only; the
        # rank-based read is the C-index bars plus survival curves by predicted risk.
        cindex_only = [("cindex", "C-index")]
        _plot_metric_bars(results, output_dir / "downstream_metric_bars.png",
                          metrics=cindex_only, ylim=(0.4, 1.0))
        _plot_metric_bars(results, output_dir / "downstream_metric_bars_cv.png",
                          metrics=cindex_only, ylim=(0.4, 1.0), source="cv")
        _plot_km_by_risk(results, output_dir / "downstream_km_by_risk.png")
    else:
        _plot_roc(results, output_dir / "downstream_roc.png")
        _plot_pr(results, output_dir / "downstream_pr.png", prevalence=float(y_te.mean()))
        _plot_metric_bars(results, output_dir / "downstream_metric_bars.png")
        _plot_metric_bars(results, output_dir / "downstream_metric_bars_cv.png", source="cv")
        _plot_confusion(results, output_dir / "downstream_confusion_matrices.png")


def _print_summary(results: dict[str, dict], task: str) -> None:
    """The CV (mean±std over the shared folds) and test-set table of every arm. Every column
    exists in r["cv"] for its task (binary CV also carries sensitivity/specificity;
    regression CV is exactly r2/mae/mse), so one accessor covers every path without
    branching."""
    print("\n[downstream] CV (mean±std over shared folds) and test-set summary")
    if task == "regression":
        cols = [("r2", "R2"), ("mae", "MAE"), ("mse", "MSE")]
    elif task == "survival":
        # "C-idx", not "C-index": the shared header format below allows 12 columns for
        # "<label> test", which a 7-character label fills exactly, running it into the
        # preceding cv column. Same width class as AUROC, so it lines up.
        cols = [("cindex", "C-idx")]
    else:
        cols = [("auc_roc", "AUROC"), ("avg_precision", "AUPRC"), ("f1_score", "F1")]
    print(f"{'model':<28}" + "".join(f"{lab + ' cv':>17}{lab + ' test':>12}"
                                     for _, lab in cols))
    for name, r in results.items():
        cells = "".join(f"{np.mean(r['cv'][key]):>10.4f}±{np.std(r['cv'][key]):<6.4f}"
                        f"{r['metrics'][key]:>12.4f}" for key, _ in cols)
        print(f"{name:<28}" + cells)


def run_downstream_comparison(
    emb_df: pd.DataFrame,
    output_dir: str,
    vocab_size: int,
    special_ids: set[int],
    *,
    itos: list[str],
    n_folds: int = 5,
    random_state: int = 42,
    task: str | None = None,
    id_col: str = "subject_id",
) -> dict[str, dict]:
    """Train the six arms (see the module docstring); report each on CV and the test split.

    `task` is "binary", "regression" or "survival"; None autodetects between the first two
    from the target values (the pipeline passes it explicitly, resolved from the outcomes
    file before extraction - survival is never autodetected, naming a duration selects it).
    `itos` (id->token) is REQUIRED: emb_df's bow_tokens column holds token names, so the
    vocabulary is what turns them back into the column indices the BoW matrix needs. It also
    names the SHAP features and identifies the sex tokens to exclude from the counts.
    `id_col` is emb_df's patient column: when a training patient has several anchors (rows),
    the cross-validation folds are grouped by it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if task is None:
        task = detect_target_task(emb_df["target"].to_numpy())
    tr = emb_df[emb_df["split"] == "train"].reset_index(drop=True)
    te = emb_df[emb_df["split"] == "test"].reset_index(drop=True)
    # A patient with several anchors must never sit on both sides of a CV fold, or the fold
    # scores anchors of a patient the model was fit on. Grouping is switched on only when a
    # patient repeats, so a one-anchor-per-patient file keeps exactly the folds it had.
    train_ids = tr[id_col]
    groups = train_ids.to_numpy() if train_ids.duplicated().any() else None
    if groups is not None:
        print(f"[downstream] CV folds grouped by patient: "
              f"{int((train_ids.value_counts() > 1).sum())} of {train_ids.nunique()} train "
              f"patients have more than one anchor")
    y_tr, y_te = _split_labels(tr, te, emb_df, task)
    learners = _task_learners(task, random_state)
    n_folds = _capped_folds(task, n_folds, train_ids, y_tr)

    arms = _feature_arms(tr, te, emb_df, itos, vocab_size, special_ids)
    results, shap_inputs = _fit_arms(arms, task, learners, y_tr, y_te, n_folds=n_folds,
                                     random_state=random_state, groups=groups)
    _write_metrics(results, output_dir)
    _plot_results(results, task, y_te, output_dir)
    _print_summary(results, task)

    # Save the fitted models and run SHAP (test-split matrices as the explain set, train
    # matrices as the linear-explainer background), on the shap_inputs the fitting loop
    # registered. Runs last so metrics/plots above are already persisted regardless.
    save_downstream_models(results, output_dir)
    explain_downstream_models(results, shap_inputs, output_dir)
    return results


# ============================================================================ #
# Plots
# ============================================================================ #
def _plot_roc(results: dict[str, dict], path: Path) -> None:
    """Binary: every arm's ROC curve on the test split, with its AUROC in the legend."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y"], r["proba"])
        ax.plot(fpr, tpr, lw=2, color=MODEL_COLORS.get(name, "gray"),
                label=f"{name} (AUROC={r['metrics']['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — test set")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_pr(results: dict[str, dict], path: Path, prevalence: float) -> None:
    """Binary: every arm's precision-recall curve on the test split, with its AUPRC in the
    legend and the test prevalence as the no-skill line."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(r["y"], r["proba"])
        ax.plot(rec, prec, lw=2, color=MODEL_COLORS.get(name, "gray"),
                label=f"{name} (AUPRC={r['metrics']['avg_precision']:.3f})")
    ax.axhline(prevalence, ls="--", color="#888", lw=1, label=f"prevalence={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — test set")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_bars(results: dict[str, dict], path: Path,
                      metrics: list[tuple[str, str]] | None = None,
                      ylim: tuple[float, float] | None = (0, 1.08),
                      source: str = "test") -> None:
    """Grouped bars of every arm's metrics (default: AUROC, AUPRC, F1). source="test" plots
    the held-out scores; source="cv" plots the fold mean with the fold spread as error
    bars - the read that says whether a gap between two arms is bigger than fold-to-fold
    noise."""
    names = list(results.keys())
    metrics = metrics or [("auc_roc", "AUROC"), ("avg_precision", "AUPRC"), ("f1_score", "F1")]
    x = np.arange(len(names))
    w = 0.26
    # Width tracks the arm count: six labels at fontsize 9 need ~10.5in of axis.
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(names)), 5))
    for j, (key, lab) in enumerate(metrics):
        if source == "cv":
            folds = [results[n]["cv"][key] for n in names]
            vals = [float(np.mean(f)) for f in folds]
            err = [float(np.std(f)) for f in folds]
        else:
            vals, err = [results[n]["metrics"][key] for n in names], None
        # Centred offset: identical to (j-1)*w for three metrics, correct for one
        bars = ax.bar(x + (j - (len(metrics) - 1) / 2) * w, vals, w, label=lab, yerr=err,
                      ecolor="#52514e", capsize=2,
                      color=["#2a78d6", "#1baf7a", "#eda100"][j], zorder=3)
        for b, v, e in zip(bars, vals, err or [0.0] * len(vals)):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#52514e")
    # Arm names are long enough to collide head-on at six arms; a slight rotation clears
    # them without widening the figure past what fits in a document.
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel("score")
    ax.set_title("Downstream CV metrics (mean ± SD over folds)" if source == "cv"
                 else "Downstream test metrics", pad=24)
    # Above the axes: with six arms every bar starts at 0, so there is no in-plot gap left.
    ax.legend(frameon=False, ncol=len(metrics), loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", color="#e6e6e3", lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_vs_true(results: dict[str, dict], path: Path) -> None:
    """Regression counterpart of _plot_confusion: one predicted-vs-true scatter panel per
    arm on the test split, with its R² and MAE in the title."""
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        m, y, pred = results[name]["metrics"], results[name]["y"], results[name]["proba"]
        ax.scatter(y, pred, s=14, alpha=0.45, color=MODEL_COLORS.get(name, "gray"))
        lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1)
        ax.set_xlabel("True value")
        ax.set_ylabel("Predicted value")
        ax.set_title(f"{name}\nR²={m['r2']:.3f}  MAE={m['mae']:.3g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_km_by_risk(results: dict[str, dict], path: Path, n_groups: int = 3) -> None:
    """Survival counterpart of _plot_confusion: one Kaplan-Meier panel per arm.

    Patients are split into equal-sized groups by PREDICTED risk and each group's
    observed survival is drawn (km_by_risk_panel, shared with the stage-06 report so the
    two read identically). A model that discriminates produces ordered, well-separated
    curves with the high-risk group lowest; overlapping curves say the C-index is coming
    from noise no matter what the bar chart shows.
    """
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        m, y, risk = results[name]["metrics"], results[name]["y"], results[name]["proba"]
        durations, events = split_survival_labels(y)
        km_by_risk_panel(ax, durations, events, risk, n_groups=n_groups)
        ax.set_title(f"{name}\nC-index={m['cindex']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(results: dict[str, dict], path: Path) -> None:
    """Binary: one test-split confusion-matrix panel per arm (at the metrics' 0.5
    threshold), with its AUROC in the title."""
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        m = results[name]["metrics"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "#0b0b0b", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(f"{name}\nAUROC={m['auc_roc']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
