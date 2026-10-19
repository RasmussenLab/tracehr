# src/tracehr/probing/embedding_outcome_prediction.py
"""Pipeline 05's stage: downstream outcome prediction from a trained NTPTransformer used as a
frozen feature extractor. For each outcomes file (patient id, anchor timestamp, target)
every anchor row's history is truncated before the anchor and embedded
(anchor_embeddings), then the six models of downstream_comparison are trained on the
ORIGINAL train+val patients and evaluated on the ORIGINAL test patients. A patient may have
several anchors - one row each, all in the patient's split, with cross-validation folds
grouped by patient. The pipeline module (pipelines/outcome_probe_prediction.py) supplies the
config, the loaded model and the output directory; this module loops over the outcomes
file(s), extracts, compares and summarizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset.cohort import (
    anchors_by_patient,
    load_anchors,
    load_patient_id_filter,
    validate_cutoff,
)
from ..dataset.column_mapping import enabled_col, resolve_anchor_columns
from ..dataset.vocabulary import load_vocab, vocab_fingerprint
from ..finetuning.survival_core import detect_survival_task
from .anchor_embeddings import extract_anchor_embeddings
from .anchor_windows import (
    anchor_fingerprint,
    anchor_window_store_mismatch,
    build_anchor_window_store,
)
from .downstream_comparison import detect_target_task, run_downstream_comparison
from .embeddings import write_extraction_meta


# ============================================================================ #
# Stage orchestration (pipeline entry point)
# ============================================================================ #
def _prepare_window_store(dcfg, common, anchor_table, acols, name, odir, split_by_pid, model,
                          vocab_path, event_times=None):
    """Reuse a matching anchor window store, or build one. Returns its directory.

    Same exists -> validate -> regenerate contract training uses for its window npz, with
    the anchor cohort folded into the fingerprint so a changed anchor age, target set or
    split is caught too. anchor_table is the load_anchors table of this outcomes file.
    """
    base = dcfg.get("windows_dir") or odir
    windows_dir = Path(base) / f"anchor_windows_{name}"

    id_col, age_col = acols["id_col"], acols["age_col"]
    cohort = anchor_table[anchor_table[id_col].isin(set(split_by_pid))]
    anchors = anchors_by_patient(cohort, id_col, age_col)
    targets = dict(zip(zip(cohort[id_col], cohort[age_col].astype(float)),
                       cohort[acols["target_col"]]))

    expected = {
        "max_seq_len": int(common["max_seq_len"]),
        "overlap_percent": float(common["overlap_percent"]),
        "start_from_end": bool(common["start_from_end"]),
        # Part of the geometry: it gates which patients survive AND the head window's
        # width. Omitting it here would silently reuse a store built under a different
        # floor, and the extraction would read windows the current config never asked for.
        "min_seq_len": int(common["min_seq_len"]),
        # Same argument: it changes the window count for an unchanged cohort.
        "add_head_window": bool(common["add_head_window"]),
        "time_before_event": float(common["time_before_event"]),
        "remove_cutoff_token": bool(common["remove_cutoff_token"]),
        "use_time_abs": bool(common["use_time_abs"]),
        "use_type": bool(common["use_type"]),
        "vocab_hash": vocab_fingerprint(vocab_path),
        "anchor_hash": anchor_fingerprint(anchors, split_by_pid),
    }
    if windows_dir.is_dir():
        mismatch = anchor_window_store_mismatch(windows_dir, **expected)
        if mismatch is None:
            print(f"[anchor-windows] {windows_dir} matches the config, reusing it.")
            return str(windows_dir)
        print(f"[anchor-windows] {windows_dir} is stale ({mismatch}); regenerating.")

    build_anchor_window_store(
        windows_dir, common["parquet_path"], vocab_path,
        anchors=anchors, targets=targets, split_by_pid=split_by_pid,
        max_seq_len=common["max_seq_len"], use_time_abs=common["use_time_abs"],
        use_type=common["use_type"], overlap_percent=common["overlap_percent"],
        start_from_end=common["start_from_end"], min_seq_len=common["min_seq_len"],
        add_head_window=common["add_head_window"],
        time_before_event=common["time_before_event"],
        remove_cutoff_token=common["remove_cutoff_token"],
        id_col=common["id_col"], tok_col=common["tok_col"],
        time_age_col=common["time_age_col"], time_abs_col=common["time_abs_col"],
        type_col=common["type_col"], event_times=event_times)
    return str(windows_dir)


def check_split_labels(anchors: pd.DataFrame, acols: dict[str, str | None],
                       split_by_pid: dict, task: str) -> None:
    """Reject a cohort whose splits cannot support the task's metrics, before extracting.

    `anchors` is the outcomes file as load_anchors returns it. The label counts are in
    patients, not rows: cross-validation folds are grouped by patient when a patient has
    several anchors, so they can only move whole patients (one anchor each: the same count).

    Cross-validation folds can be made smaller until every one of them is scorable (see
    _fold_cap), but two cases no fold arithmetic reaches: a training split too one-sided to
    form even two folds, and a TEST split with a single class - which leaves the headline
    column undefined no matter what happens during training. Both are data problems, and
    the only useful moment to report them is here, before an embedding pass that can take
    hours, rather than as a table of nulls afterwards.

    This counts the labels as they sit in the outcomes file, and nothing re-counts them
    after extraction drops patients for having too little pre-anchor history - so a split
    that clears the floor of 2 here can still reach the CV loop holding 1 or 0. That
    residue is NOT caught by the NaN -> null path, which only turns an unscorable
    VALIDATION fold into a null. StratifiedKFold splits such a cohort without complaint
    (it raises only when n_splits exceeds EVERY class count), so the damage lands on a
    TRAINING fold left with a single class: binary fails inside the LR arms' fit
    ("needs samples of at least 2 classes"), mid-CV and after the embedding pass this
    check exists to run ahead of, while survival fits a zero-event fold quietly and
    returns a constant risk, making that fold's C-index noise rather than an error.
    Reaching either needs the outcomes file to hold exactly 2 of the rarer label in the
    split and extraction to drop one of them.
    """
    if task not in ("binary", "survival"):
        return                                  # a continuous target has no class balance
    oc = anchors
    split = oc[acols["id_col"]].map(split_by_pid)
    label = "event" if task == "survival" else "positive"

    for name, floor in (("train", 2), ("test", 1)):
        ids = oc.loc[split == name, acols["id_col"]]
        y = oc.loc[split == name, acols["target_col"]].to_numpy(dtype=np.float64)
        n_patients = ids.nunique()
        n_pos = ids[y == 1].nunique()
        n_neg = ids[y != 1].nunique()
        print(f"[downstream] split '{name}': {len(y)} anchors from {n_patients} patients; "
              f"patients {label}={n_pos}"
              f"{'' if task == 'survival' else f', negative={n_neg}'}")
        if len(y) == 0:
            raise ValueError(
                f"Split '{name}' is empty: no patient of the outcomes file appears in the "
                f"{name} id file(s). Check that the id columns share a dtype and an id space.")
        scarce = n_pos if task == "survival" else min(n_pos, n_neg)
        if scarce < floor:
            raise ValueError(
                f"Split '{name}' has only {scarce} patient(s) with a {label} label out of "
                f"{n_patients} patients. "
                f"{'Harrell C' if task == 'survival' else 'AUROC and AUPRC'} "
                f"cannot be computed on it, so this run would report undefined metrics for "
                f"the {name} column. Use a larger cohort, or a different anchor/outcome "
                f"definition.")


def _load_outcomes(opath: str, name: str, config: dict, split_by_pid: dict, event_times):
    """(anchor columns, anchors, task) of one outcomes file: its schema validated, its rows
    read once and its task resolved and checked against the splits - all before the
    expensive embedding extraction."""
    # Validates this outcomes file's schema (id, anchor age, target) up front
    acols = resolve_anchor_columns(config, parquet_path=opath, require_target=True)
    print(f"\n===== Downstream outcome prediction: {name} =====")
    # Naming anchor_columns.censor_timestamp_col is what selects survival: the
    # two-column label cannot be autodetected from target values alone. Without it the
    # task is autodetected below (binary or regression).
    duration_col = acols["duration_col"]
    # Read once: rows missing the anchor or the outcome, and repeated (patient, anchor)
    # rows, are dropped and reported; several anchors per patient are kept. For survival
    # load_anchors computes the duration (days from the anchor to the censor timestamp).
    anchors = load_anchors(opath, acols["id_col"], acols["age_col"], [acols["target_col"]],
                           event_times=event_times, timestamp_col=acols["timestamp_col"],
                           censor_timestamp_col=acols["censor_timestamp_col"])
    if duration_col is not None:
        task = detect_survival_task(
            anchors, event_col=acols["target_col"], duration_col=duration_col, source=opath)
    else:
        # Binary vs regression, decided (and any bad target rejected) before the
        # expensive embedding extraction.
        task = detect_target_task(anchors[acols["target_col"]].to_numpy(),
                                  label=acols["target_col"], source=opath)

    # The task tells us which metrics this run will report; this tells us whether the
    # splits can actually support them. Still before the expensive extraction.
    check_split_labels(anchors, acols, split_by_pid, task)
    return acols, anchors, task


def _extraction_kwargs(model, dcfg: dict, cols: dict[str, str | None], acols: dict, *,
                       raw_parquet: str, vocab_path: str, min_seq_len: int,
                       time_before_event: float, remove_cutoff_token: bool) -> dict:
    """The fully resolved, fully defaulted keyword arguments of extract_anchor_embeddings for
    one outcomes file; everything but the model also describes the run in the sidecar."""
    return dict(
        model=model,
        parquet_path=raw_parquet,
        vocab_path=vocab_path,
        pooling=dcfg.get("pooling", "mean"),
        overlap_percent=dcfg.get("overlap_percent", 0),
        start_from_end=dcfg.get("start_from_end", True),
        combine_windows=dcfg.get("combine_windows", "recency_weighted"),
        max_seq_len=model.max_seq_len,
        min_seq_len=min_seq_len,
        add_head_window=dcfg.get("add_head_window", False),
        time_before_event=time_before_event,
        remove_cutoff_token=remove_cutoff_token,
        use_time_abs=model.use_time_abs,
        use_type=model.use_type,
        id_col=cols["id_col"],
        tok_col=cols["tok_col"],
        time_age_col=cols["age_col"],
        time_abs_col=enabled_col(cols, "abstime_col", model.use_time_abs, "model.use_time_abs"),
        type_col=enabled_col(cols, "type_col", model.use_type, "model.use_type"),
        outcome_patient_col=acols["id_col"],
        outcome_age_col=acols["age_col"],
        outcome_col=acols["target_col"],
        outcome_duration_col=acols["duration_col"],   # None for every non-survival outcome
        batch_size=dcfg.get("batch_size", 64),
        chunk_windows=dcfg.get("chunk_windows", 2048),
        save_sequences=dcfg.get("save_sequences", False),
    )


def run_downstream_stage(model, dcfg: dict, config: dict, *, raw_parquet: str,
                         cols: dict[str, str | None], out_dir, vocab_path: str,
                         min_seq_len: int = 1, event_times=None) -> tuple[dict, dict]:
    """Extract anchor embeddings and compare the six arms (see the module docstring).

    Runs once per outcomes file (dcfg["outcomes_path"] may be a single parquet or a list);
    all artifacts land under `out_dir` (per-outcome subfolders when a list is given).
    `raw_parquet` is streamed and encoded on the fly (no intermediate file). `cols` names
    the sequence parquet columns; the outcomes-file columns come from `config`'s
    anchor_columns block. Each file's task is resolved and printed before the embeddings
    are extracted: naming anchor_columns.censor_timestamp_col selects survival, otherwise
    binary 0/1 vs continuous regression is autodetected from the target column. event_times
    (load_event_times) drives the readers and the anchor time_age.

    min_seq_len is the window floor. It is NOT a config knob: the pipeline reads it off the
    checkpoint's training config (training_min_seq_len), because inference windowing has to
    match the geometry the model was trained on - the same reason pipeline 02 reads it back.
    Returns ({outcome_name: results}, {outcome_name: task}) for the summary.
    """
    out_dir = Path(out_dir)

    # Config-only values first, before any file is opened: a mistyped fold count should not
    # be reported after a vocabulary load and three parquet reads. Shared by every outcomes
    # file in this run, so a list of them stays comparable.
    n_folds = int(dcfg.get("n_folds", 5))
    random_state = int(dcfg.get("random_state", 42))
    if n_folds < 2:
        raise ValueError(f"outcome.n_folds must be at least 2, got {n_folds}. It is the "
                         f"cross-validation fold count shared by all six arms.")
    # Checked here, not in the extractor: a matching window store is reused before any
    # extractor runs, and a bad value would otherwise slip through with it.
    time_before_event, remove_cutoff_token = validate_cutoff(
        dcfg.get("time_before_event", 0), dcfg.get("remove_cutoff_token", True))
    # The floor comes from the checkpoint, so a key here could only ever disagree with the
    # geometry the model was trained on. Silently ignoring it is the exact failure this
    # value exists to prevent, so say so instead.
    if "min_seq_len" in dcfg:
        raise ValueError(
            f"outcome.min_seq_len is not a config knob (found {dcfg['min_seq_len']!r}). The "
            f"window floor must match training, so it is read from the checkpoint's "
            f"sequences.min_seq_len (currently {min_seq_len}). Remove the key; to change "
            f"the floor, retrain with a different sequences.min_seq_len.")

    stoi, itos, _ = load_vocab(vocab_path)
    vocab_size = len(itos)
    # <UNK> is deliberately NOT excluded. It is the only special that can occur in a
    # patient's history - the encoder maps every out-of-vocabulary event to it
    # (COALESCE(v.tid, unk_id) in stream_encoded_patient_sequences) - so its count is a
    # real feature: how much of this patient's record the vocabulary failed to cover.
    # The rest are window artifacts written by fill_window and never reach work.toks.
    special_ids = {stoi[t] for t in ("<PAD>", "<SOS>", "<CON>", "<MASK>") if t in stoi}

    def load_ids(key):
        """The patient ids of the id file dcfg[key], as a set."""
        return set(load_patient_id_filter(dcfg[key]).tolist())

    # Classifier train set = original train + val; test set = original test.
    # One map drives a single extraction pass over both splits.
    train_ids = load_ids("train_ids") | load_ids("val_ids")
    test_ids = load_ids("test_ids")
    split_by_pid = {**{p: "train" for p in train_ids}, **{p: "test" for p in test_ids}}

    mode = dcfg.get("mode", "streaming")
    if mode not in {"streaming", "save_intermediate"}:
        raise ValueError(
            f"outcome.mode must be one of ['save_intermediate', 'streaming'], got {mode!r}.")
    # Fail before building a store that the extraction would then refuse to read.
    if mode == "save_intermediate" and dcfg.get("save_sequences", False):
        raise ValueError(
            "outcome.save_sequences is not available with mode: save_intermediate - the "
            "store keeps windows, not token strings. Use mode: streaming, or turn "
            "save_sequences off.")

    paths = dcfg["outcomes_path"]
    paths = [paths] if isinstance(paths, str) else list(paths)
    all_results, all_tasks = {}, {}
    for opath in paths:
        name = Path(opath).stem
        odir = out_dir / name if len(paths) > 1 else out_dir
        odir.mkdir(parents=True, exist_ok=True)
        acols, anchors, task = _load_outcomes(opath, name, config, split_by_pid, event_times)
        common = _extraction_kwargs(model, dcfg, cols, acols, raw_parquet=raw_parquet,
                                    vocab_path=vocab_path, min_seq_len=min_seq_len,
                                    time_before_event=time_before_event,
                                    remove_cutoff_token=remove_cutoff_token)
        windows_dir = None
        if mode == "save_intermediate":
            windows_dir = _prepare_window_store(
                dcfg, common, anchors, acols, name, odir, split_by_pid, model, vocab_path,
                event_times=event_times)

        print(f"Extracting anchor embeddings (pooling={common['pooling']}) ...")
        emb_df = extract_anchor_embeddings(
            anchors=anchors, patient_ids=None, split_name=None, split_by_pid=split_by_pid,
            windows_dir=windows_dir, event_times=event_times, **common)

        emb_path = odir / "downstream_embeddings.parquet"
        emb_df.to_parquet(emb_path, index=False)
        # `common` is already the fully resolved, fully defaulted kwarg set; everything in it
        # except the model itself describes the run, which is exactly what the sidecar is for.
        write_extraction_meta(odir, emb_df, outcomes_path=opath,
                              **{k: v for k, v in common.items() if k != "model"})
        print(f"Saved embeddings to {emb_path} ({len(emb_df)} rows)")

        # Window-artifact specials are never bag-of-words features; <UNK> is one (see above).
        all_results[name] = run_downstream_comparison(emb_df, str(odir), vocab_size, special_ids,
                                                     itos=itos, task=task, n_folds=n_folds,
                                                     random_state=random_state,
                                                     id_col=cols["id_col"])
        all_tasks[name] = task
        print(f"Downstream results saved to {odir}")
    return all_results, all_tasks


def summarize_downstream(results: dict, tasks: dict) -> dict:
    """Per-outcome model metrics plus the task's headline lift over the Sex+Age baseline.

    The lift key names the metric it is computed from: auroc_lift_over_sex_age for a
    binary task (AUROC), r2_lift_over_sex_age for a regression task (R2),
    cindex_lift_over_sex_age for a survival task (Harrell's C).
    """
    _LIFT = {
        "regression": ("r2", "r2_lift_over_sex_age"),
        "survival": ("cindex", "cindex_lift_over_sex_age"),
    }
    summary = {}
    for name, res in results.items():
        s = {clf: r["metrics"] for clf, r in res.items()}
        metric, key = _LIFT.get(tasks.get(name), ("auc_roc", "auroc_lift_over_sex_age"))
        s[key] = (res["Embedding + LR"]["metrics"][metric]
                  - res["Sex+Age + LR"]["metrics"][metric])
        summary[name] = s
    return summary
