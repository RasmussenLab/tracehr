# src/tracehr/dataset/cohort.py
"""Who is in each split and where each history is cut: the patient-id files, the anchors (or
outcomes) table, and the cutoff rule every anchored pipeline (03, 04, 05, 06) shares.
"""

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .meds_reader import days_between

# The anchors table's survival duration in days, computed by load_anchors from the anchor
# and censor timestamps (never read from the outcomes file). The package prefix keeps it from
# colliding with an outcomes column, and resolve_anchor_columns refuses an id or target column
# with this name (ignoring case).
DURATION_COL = "tracehr_survival_duration"

# What a pipeline does with an anchors row whose anchor Age is missing (NaN/None).
# Generation and embedding extraction (pipelines 03, 04) keep the whole history; the
# prediction pipelines (05, 06) drop the row when the file is loaded (load_anchors).
# The outcome extractors skip one that is passed to them directly.
ANCHOR_KEEPS_FULL_HISTORY = "using the full history for those patients"
ANCHOR_SKIPPED = "those patients are skipped"
ANCHOR_DROPPED = "those rows are dropped"


def load_patient_id_filter(ids_path: str | None) -> np.ndarray | None:
    """The patient ids of a split file (the first column of a parquet), or None without a path."""
    if not ids_path:
        return None
    return pd.read_parquet(ids_path).iloc[:, 0].to_numpy()


def report_missing_anchors(anchor_ages: Iterable, consequence: str, label: str = "") -> int:
    """Print how many anchors are missing and what the pipeline does with them; return the
    count. Silent when none is missing. Reported because a broken anchors file (a failed
    merge, a bad export) looks the same as an intended "no anchor"."""
    ages = pd.Series(list(anchor_ages), dtype=object)
    n_missing = int(ages.isna().sum())
    if n_missing:
        where = f" ({label})" if label else ""
        print(f"Anchors{where}: {n_missing:,} of {len(ages):,} anchors are missing: {consequence}")
    return n_missing


def load_anchors(path: str, id_col: str, age_col: str,
                 outcome_cols: Iterable[str | None] = (), *,
                 event_times, timestamp_col: str,
                 censor_timestamp_col: str | None = None) -> pd.DataFrame:
    """Read an anchors/outcomes parquet: one row per (patient, anchor), and a patient may
    have several anchors. Every pipeline that reads anchors loads them here.

    The anchor timestamp (timestamp_col) is turned into the anchor time_age under the name
    age_col, with the run's births (see _anchor_ages_from_timestamps); for survival, the
    censor timestamp becomes the duration in days from the anchor, DURATION_COL. Only
    those columns and the outcome columns are read. outcome_cols is given by the prediction
    pipelines (05, 06), which then drop rows missing the anchor or any outcome - first,
    so a usable row is never lost to an unusable duplicate of it. Rows repeating a
    (patient, anchor) pair are then dropped, keeping the first. Every drop is reported.
    """
    outcome_cols = [c for c in outcome_cols if c is not None]
    read = [id_col, timestamp_col, *outcome_cols] + ([censor_timestamp_col] if censor_timestamp_col else [])
    df = pd.read_parquet(path, columns=read)
    print(f"Anchors: {len(df):,} rows read from {path}")
    df = _anchor_ages_from_timestamps(df, path, id_col, age_col, timestamp_col, event_times,
                                      censor_timestamp_col)
    if outcome_cols:
        report_missing_anchors(df[age_col], ANCHOR_DROPPED)
        df = df[df[age_col].notna()]
        for col in outcome_cols:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                print(f"  dropped {n_missing:,} row(s) with no outcome ('{col}')")
                df = df[df[col].notna()]
    n_dup = int(df.duplicated([id_col, age_col]).sum())
    if n_dup:
        print(f"  dropped {n_dup:,} duplicate ({id_col}, {age_col}) row(s), keeping the first")
        df = df.drop_duplicates([id_col, age_col])
    per_patient = df[id_col].value_counts()
    print(f"  kept {len(df):,} rows: {len(per_patient):,} patients, "
          f"{int((per_patient > 1).sum()):,} of them with more than one anchor")
    return df.reset_index(drop=True)


def _anchor_ages_from_timestamps(df: pd.DataFrame, path: str, id_col: str, age_col: str,
                                 timestamp_col: str, event_times,
                                 censor_timestamp_col: str | None = None) -> pd.DataFrame:
    """Replace the anchor timestamp with the anchor time_age (days since the patient's birth)
    and, for survival, the censor timestamp with the duration in days from the anchor.

    Same births and same day formula as the event times (days_between), so an anchor at an
    event's moment gets exactly that event's time_age and remove_cutoff_token=False keeps the
    event. Rows of patients without a usable birth - dropped from the event data by
    load_event_times - are dropped and reported; a missing anchor timestamp gives a missing
    (NaN) anchor, which each pipeline then treats as it treats a missing anchor. Survival
    rows whose censor timestamp is missing or before the anchor are dropped and reported.
    """
    births = event_times.births
    row = pd.Index(births.column(id_col).to_pandas()).get_indexer(df[id_col])
    has_birth = row >= 0
    n_lost = int((~has_birth).sum())
    if n_lost and n_lost == len(df):
        raise ValueError(
            f"None of the {n_lost:,} anchors in {path} belongs to a patient of the event data. "
            f"The id columns likely use different dtypes or id spaces.")
    if n_lost:
        print(f"  dropped {n_lost:,} row(s) of patients with no usable birth in the event data")
    df = df[has_birth]
    t = df[timestamp_col].to_numpy(dtype="datetime64[us]")
    birth_us = births.column("birth_us").to_numpy()[row[has_birth]]
    ages = np.full(len(df), np.nan)
    known = ~np.isnat(t)
    ages[known] = days_between(t[known].astype(np.int64), birth_us[known])
    df = df.drop(columns=[timestamp_col]).assign(**{age_col: ages})
    if censor_timestamp_col is None:
        return df
    c = df[censor_timestamp_col].to_numpy(dtype="datetime64[us]")
    durations = np.full(len(df), np.nan)
    both = known & ~np.isnat(c)
    durations[both] = days_between(c[both].astype(np.int64), t[both].astype(np.int64))
    no_censor = known & np.isnat(c)
    before = both & (durations < 0)
    if no_censor.any():
        print(f"  dropped {int(no_censor.sum()):,} row(s) with no censor timestamp "
              f"('{censor_timestamp_col}')")
    if before.any():
        print(f"  dropped {int(before.sum()):,} row(s) whose censor timestamp is before the anchor")
    keep = ~(no_censor | before)
    return df.drop(columns=[censor_timestamp_col]).assign(**{DURATION_COL: durations})[keep]


def anchors_by_patient(anchors: pd.DataFrame, id_col: str, age_col: str) -> dict:
    """{patient id: [anchor ages]} from a load_anchors table (missing ages kept as NaN)."""
    return {pid: [float("nan") if pd.isna(a) else float(a) for a in ages]
            for pid, ages in anchors.groupby(id_col, sort=False)[age_col]}


def validate_cutoff(time_before_event, remove_cutoff_token) -> tuple[float, bool]:
    """The two cutoff settings every anchored pipeline shares (03/04/05/06), checked and
    returned as (float, bool). Kept events are Age < anchor - time_before_event, or <= when
    remove_cutoff_token is False. A NaN or infinite gap keeps the whole history or none of
    it, a negative one moves the cutoff past the anchor, and a quoted "false" is truthy -
    each would silently change which events every row keeps."""
    gap = float(time_before_event)
    if not (gap >= 0 and math.isfinite(gap)):
        raise ValueError(f"time_before_event must be a finite number of days >= 0, "
                         f"got {time_before_event!r}.")
    if not isinstance(remove_cutoff_token, bool):
        raise ValueError(f"remove_cutoff_token must be true or false (unquoted in YAML), "
                         f"got {remove_cutoff_token!r}.")
    return gap, remove_cutoff_token


def cutoff_index(ages: np.ndarray, cutoff, remove_cutoff_token: bool) -> int:
    """How many leading events a cutoff keeps: those with time_age < cutoff, or <= cutoff
    when remove_cutoff_token is False. The one implementation of the cutoff rule in Python
    (stream_encoded_patient_sequences applies the same rule in SQL).

    ages must be sorted ascending, as the canonical event order (canonical_sort) leaves
    them, so the kept events are a prefix. A missing (None/NaN) cutoff keeps every event.
    """
    if cutoff is None or pd.isna(cutoff):
        return len(ages)
    return int(np.searchsorted(ages, cutoff, side="left" if remove_cutoff_token else "right"))
