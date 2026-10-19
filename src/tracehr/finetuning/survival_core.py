# src/tracehr/finetuning/survival_core.py
"""The survival label contract and everything that reads it, shared by the two stages that
model a censored time to event:
  stage 05 (outcome prediction) the frozen-embedding comparison arms
  stage 06 (finetuning)         a Cox head on the transformer itself

A censored label is TWO numbers, and every mistake in this file's subject matter is
silent rather than loud: a transposed pair still fits, an unnegated risk score still
scores, and both produce a plausible-looking C-index. So the contract lives in exactly
one place:

  labels are [N, 2], column 0 = duration, column 1 = event (1/0)
  a model's output is a RISK score - higher risk means SHORTER survival

Both stages import their metric from here, which is what makes their C-indices
comparable by construction rather than by convention.

It sits in finetuning/ as one of the shared modules, and stage 05 imports it from here.
Keep it light: only numpy is imported at module scope, and torch and matplotlib each load
lazily inside the one function that needs them.
"""

import numpy as np


def split_survival_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(durations, events) from the [N, 2] survival label array.

    One place owns the column order, so a transposed label array fails here rather
    than silently producing a C-index near 0.5 downstream.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(
            f"Survival labels must be a [N, 2] array of (duration, event), got shape "
            f"{y.shape}. Column 0 is the time from the anchor to the event or to "
            f"censoring; column 1 is the 1/0 event indicator.")
    durations, events = y[:, 0], y[:, 1]
    if not np.isin(events, (0.0, 1.0)).all():
        raise ValueError(
            f"The event column (labels[:, 1]) must be 0/1, found "
            f"{np.unique(events)[:6].tolist()}. Column 0 must be the duration and "
            f"column 1 the event indicator, not the other way round.")
    if not np.isfinite(durations).all():
        raise ValueError("Survival durations contain NaN or inf.")
    if (durations < 0).any():
        raise ValueError("Survival durations contain negative values.")
    return durations, events


def concordance(durations: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index for a RISK score (higher risk = shorter survival).

    A pair is comparable when the patient with the shorter duration had the event; a
    censored patient tied in time with an event counts as outliving it, and two events at
    the same time are not compared. The pair is concordant when the earlier event has the
    higher risk, and a tied risk counts half: (concordant + tied / 2) / comparable. These
    are lifelines' rules (its concordance_index, which reads a HIGHER score as LONGER
    survival, i.e. the negated risk), and the counts match it exactly.

    With no comparable pairs the C-index does not exist and NaN is returned. 0.5 would be
    the tempting substitute - "uninformative" - but it is a number nobody measured, and it
    survives every downstream mean, plot and summary looking exactly like a real result.
    NaN propagates instead, and json_safe writes it as null.
    """
    durations = np.asarray(durations, dtype=np.float64).ravel()
    events = np.asarray(events, dtype=np.float64).ravel()
    risk = np.asarray(risk, dtype=np.float64).ravel()
    if not durations.shape == events.shape == risk.shape:
        raise ValueError(f"durations, events and risk must have the same length, got "
                         f"{len(durations)}, {len(events)} and {len(risk)}.")
    if np.isnan(durations).any() or np.isnan(events).any() or np.isnan(risk).any():
        raise ValueError("NaNs in the C-index inputs (durations, events or risk scores).")

    concordant, tied, comparable = _concordance_counts(durations, events.astype(bool), risk)
    if comparable == 0:
        print("  [survival] WARNING: no admissible pairs for the C-index "
              f"({int(events.sum())} events in {len(events)} rows); reporting it as "
              f"undefined (NaN).")
        return float("nan")
    return float((concordant + tied / 2) / comparable)


def _concordance_counts(durations: np.ndarray, died: np.ndarray, risk: np.ndarray
                        ) -> tuple[int, int, int]:
    """(concordant, tied, comparable) pair counts behind concordance, in O(n log n).

    Walks the durations from the longest down. Everyone already walked past outlived the
    current time, so they form each event's comparison pool, held as counts per risk rank
    in a Fenwick tree (prefix sums in O(log n)). At each time the censored rows join the
    pool first (they outlive an event at the same time), then every event there is
    compared against the pool, and only then do the events join (two events at the same
    time are not compared).
    """
    ranks = np.unique(risk, return_inverse=True)[1].ravel() + 1   # 1-based Fenwick index
    tree = [0] * (int(ranks.max(initial=0)) + 1)

    def add(i):
        while i < len(tree):
            tree[i] += 1
            i += i & -i

    def count_up_to(i):                                           # pool members with rank <= i
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    order = np.argsort(-durations, kind="stable")
    times, ranks, died = durations[order], ranks[order], died[order]
    starts = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
    pool = concordant = tied = comparable = 0
    for start, end in zip(starts, np.r_[starts[1:], len(times)]):
        group_ranks, group_died = ranks[start:end].tolist(), died[start:end].tolist()
        for r, d in zip(group_ranks, group_died):
            if not d:
                add(r)
                pool += 1
        for r, d in zip(group_ranks, group_died):
            if d:
                below = count_up_to(r - 1)                        # lower risk: concordant
                concordant += below
                tied += count_up_to(r) - below
                comparable += pool
        for r, d in zip(group_ranks, group_died):
            if d:
                add(r)
                pool += 1
    return concordant, tied, comparable


def survival_metrics(y: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    """Metric set for a survival prediction. Mirrors classification_metrics / regression_metrics.

    Both stages report through this function, so "cindex" means the same thing in
    stage 06's test_metrics.json as in stage 05's downstream_metrics.json.
    """
    durations, events = split_survival_labels(y)
    return {
        "cindex": concordance(durations, events, risk),
        "n_events": int(events.sum()),
        "n_censored": int(len(events) - events.sum()),
        "event_rate": float(events.mean()),
        "median_duration": float(np.median(durations)),
        "n_total": int(len(events)),
    }


def detect_survival_task(df, event_col: str, duration_col: str,
                         source: str | None = None) -> str:
    """Validate a survival outcomes table and print the evidence. Always returns "survival".

    The survival counterpart of detect_target_task: it does not decide anything (naming a
    censor timestamp column already did), it rejects a malformed label before the
    expensive work starts and reports the censoring the run will actually see. The most
    valuable check is the swapped-column one - a duration in the event column and vice
    versa is silently fittable and produces a meaningless C-index.
    """
    where = f" in {source}" if source else ""
    events = np.asarray(df[event_col], dtype=np.float64).ravel()
    durations = np.asarray(df[duration_col], dtype=np.float64).ravel()

    for label, arr in ((event_col, events), (duration_col, durations)):
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            raise ValueError(
                f"Survival column '{label}'{where} has {n_nan} of {len(arr)} values "
                f"missing (NaN). No estimator accepts a NaN label: drop those patients "
                f"before running this stage.")

    uniq = np.unique(events)
    if not np.isin(uniq, (0.0, 1.0)).all():
        raise ValueError(
            f"For a survival outcome, anchor_columns.target_col ('{event_col}'){where} "
            f"must be the 0/1 EVENT indicator, found values {uniq[:6].tolist()}. The "
            f"time of the event or of the censoring belongs in "
            f"anchor_columns.censor_timestamp_col - the two may be the wrong way round.")
    if (durations < 0).any():
        raise ValueError(
            f"Survival durations '{duration_col}'{where} contain negative values "
            f"(min {durations.min():.4g}). A duration is the time from the anchor "
            f"forward to the event or to censoring.")
    if events.sum() == 0:
        raise ValueError(
            f"No events in '{event_col}'{where}: every patient is censored, so a Cox "
            f"model has nothing to fit.")

    n_zero = int((durations == 0).sum())
    print(f"[survival] task=survival (anchor_columns.censor_timestamp_col is named; durations "
          f"in days from the anchor): "
          f"{int(events.sum())}/{len(events)} events ({events.mean():.1%}), "
          f"{len(events) - int(events.sum())} censored, duration range "
          f"[{durations.min():.4g}, {durations.max():.4g}], median {np.median(durations):.4g}")
    if n_zero:
        # Only the censored half of this group is inert, so the warning must not call all
        # of it harmless. An EVENT at 0 is a real failure and the C-index forms comparable
        # pairs from it like any other (ranking that one patient first vs last swings the
        # C-index from 1.0 to 0.0 on a three-row example) - but the signed encoding
        # to_signed_labels feeds XGBoost cannot represent it, because +0.0 and -0.0 are
        # indistinguishable under every comparison, so the two XGBoost arms read it as
        # censored while CoxPH keeps it. One patient, two labels, different arms.
        print(f"[survival] WARNING: {n_zero} patients have duration 0 (the event or the "
              f"censoring falls on the anchor). Censored ones carry no follow-up and "
              f"contribute nothing; EVENTS at 0 are kept by CoxPH and the C-index but are "
              f"read as CENSORED by the XGBoost arms - see to_signed_labels.")
    return "survival"


def cox_ph_loss(risk, durations, events):
    """Cox negative partial log-likelihood (Breslow) over one batch. Returns a scalar tensor.

    Sorting by descending duration makes the risk set of each row a prefix, so a single
    logcumsumexp gives every row's log sum of exp(risk) over the patients still at risk when
    it fails. The loss averages -(risk - log risk-set sum) over the events only; censored
    rows contribute by sitting in the risk sets of the events that precede them.

    THE RISK SET IS THE BATCH, not the cohort. This is the standard minibatch approximation
    (what DeepSurv-style training does), and it makes batch_size statistically load-bearing:
    a small batch estimates each partial likelihood term from few comparisons, and a final
    batch with no events contributes nothing at all. Prefer the largest batch that fits.

    TIES MATTER HERE. The risk set is {j : duration_j >= duration_i}, so tied rows belong
    to each other's risk sets - but a bare prefix cuts the group off partway through,
    dropping the tied rows that happen to sort later. Durations in whole days tie
    constantly, so the prefix value is broadcast back over each tied group below, which is
    exactly Breslow's approximation.
    """
    import torch

    risk = risk.reshape(-1)
    durations = durations.reshape(-1).to(risk.dtype)
    events = events.reshape(-1).to(risk.dtype)

    n_events = events.sum()
    if n_events == 0:
        # No events, no partial likelihood. Return a zero that still carries grad, so an
        # unlucky batch is a no-op rather than a NaN that poisons every later step.
        return (risk * 0.0).sum()

    order = torch.argsort(durations, descending=True)
    r, e, d = risk[order], events[order], durations[order]
    # logcumsumexp over the descending-duration order = log sum of exp(risk) over every row
    # up to here. Computed in log space, so a large risk score cannot overflow the way exp
    # would. Sorting descending makes equal durations contiguous, so taking each group's
    # LAST prefix value gives every tied row the full risk set it is entitled to.
    log_cum = torch.logcumsumexp(r, dim=0)
    _, inverse, counts = torch.unique_consecutive(d, return_inverse=True, return_counts=True)
    log_risk_set = log_cum[torch.cumsum(counts, dim=0) - 1][inverse]
    return -((r - log_risk_set) * e).sum() / n_events


def risk_groups(risk: np.ndarray, n_groups: int = 3) -> np.ndarray:
    """Equal-sized groups of RANKED risk: 0 = lowest risk, n_groups-1 = highest.

    Rank rather than value cuts, because the two stages produce different scales for the
    same ordering - a log relative risk (Cox head, CoxPH arm) or a hazard ratio
    (XGBoost survival:cox) - and only the ordering is comparable between them.
    """
    risk = np.asarray(risk, dtype=np.float64).ravel()
    order = np.argsort(np.argsort(risk))
    return np.minimum((order * n_groups) // max(len(risk), 1), n_groups - 1)


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Kaplan-Meier survival curve as (times, survival): times are 0 and every distinct
    duration, survival the product-limit estimate S(t) = prod over t_k <= t of
    (1 - deaths_k / at_risk_k). Computed as exp(cumsum(log(at_risk - deaths) - log(at_risk))),
    the form lifelines' KaplanMeierFitter uses, so the values match it exactly."""
    durations = np.asarray(durations, dtype=np.float64).ravel()
    died = np.asarray(events, dtype=np.float64).ravel() == 1
    times = np.unique(np.r_[0.0, durations])
    at_risk = len(durations) - np.searchsorted(np.sort(durations), times, side="left")
    deaths = np.bincount(np.searchsorted(times, durations[died]), minlength=len(times))
    with np.errstate(divide="ignore"):                   # log(0) = -inf -> survival 0
        log_steps = np.log((at_risk - deaths).astype(np.float64)) - np.log(at_risk.astype(np.float64))
    return times, np.exp(np.cumsum(log_steps))


def km_by_risk_panel(ax, durations: np.ndarray, events: np.ndarray, risk: np.ndarray,
                     n_groups: int = 3) -> None:
    """Draw observed Kaplan-Meier curves per predicted-risk group onto one axis.

    A model that discriminates produces ordered, well-separated curves with the high-risk
    group lowest; overlapping curves say the C-index is coming from noise no matter what
    the bar chart shows. Shared by both stages so the read is the same in either report.
    """
    import matplotlib.pyplot as plt

    labels = {0: "low risk", n_groups - 1: "high risk"}
    # plasma, not coolwarm: coolwarm passes through white at its midpoint, which makes the
    # middle group invisible on a white axis. This ramp stays dark enough to read at every
    # group count while still ordering low -> high.
    shades = plt.cm.plasma(np.linspace(0.05, 0.75, n_groups))
    group = risk_groups(risk, n_groups)

    for g in range(n_groups):
        sel = group == g
        if not sel.any():
            continue
        times, survival = kaplan_meier(durations[sel], events[sel])
        ax.plot(times, survival, drawstyle="steps-post", color=shades[g],
                label=labels.get(g, f"group {g + 1}"))
    ax.set_xlabel("Time from anchor")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
