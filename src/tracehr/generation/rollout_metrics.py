# src/tracehr/generation/rollout_metrics.py
"""Pipeline 03's rollouts scored against the real continuation, the way pipeline 02 scores
next-token predictions: generation_metrics.json, generation_metrics.txt (the report, also
printed) and three figures, generation_tokens_by_step, generation_tokens_by_target and
generation_times (each a 400-dpi PNG and an SVG).

A step is scored when its real event is known and is not <UNK> (score_generations' `correct`
is set). On the scored steps:
  tokens  the real event under the step's full distribution: cross-entropy and perplexity,
          top-1 accuracy (the generated token: `correct`), top-5 and top-10 accuracy and the
          mean reciprocal rank (from true_token_prob and true_token_rank); next to pipeline 02's
          two references on the same events: the context copy (the prompt's codes, the latest
          first; only the allowed codes when generation is restricted, as are the real events)
          and the unigram (the codes most frequent among all scored real events, the best
          a predictor that ignores the history can do on them). Per step, with pipeline 03's
          mean cumulative overlap, and by target in 02's four cells: new or repeat (the code is
          or is not in the prompt), same time or later (the event has or has not the time of the
          real event before it), at step 1 and over steps 2 on.
  times   for a model that advances time, on the steps with a real event: the generated and the
          real gaps by gap band, and per step the gap's error and the time elapsed since the
          prompt's last event (days).
From step 2 the model reads its own generated history, so these are rollout scores, not 02's
teacher-forced ones; step 1 is the next-token prediction at the anchor.

Everything written follows the 5-patient rule (finetune_plots): a group of fewer than
MIN_PATIENTS patients is written as HIDDEN and left out of the figures, and so are the groups
needed to keep it from being worked out by subtraction (hidden_groups). No prompt or patient
total is written, as the steps without a scored event could be counted from it.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from ..finetuning.finetune_plots import FIG_WIDTH, HIDDEN, MIN_PATIENTS, STYLE, hidden_groups, save_figure
from .evaluation import never_target_ids
from .model_io import save_metrics

MINUTE = 1.0 / 1440                    # days
GAP_BANDS = (                          # (label, lower bound, upper bound) in days
    ("< 1 min", 0.0, MINUTE),
    ("1 min-1 h", MINUTE, 1.0 / 24),
    ("1 h-1 d", 1.0 / 24, 1.0),
    ("1-7 d", 1.0, 7.0),
    ("7-30 d", 7.0, 30.0),
    ("30-180 d", 30.0, 180.0),
    (">= 180 d", 180.0, math.inf),
)
# Pipeline 02's four target cells (evaluation.TARGET_GROUPS): (key, new, same time, label)
CELLS = (("new_same_time", True, True, "New, same time"), ("new_later", True, False, "New, later"),
         ("repeat_same_time", False, True, "Repeat, same time"),
         ("repeat_later", False, False, "Repeat, later"))
# The predictors, drawn in colours C0-C2 as in pipeline 02's figure
PREDICTORS = (("model", "Model"), ("context_copy", "Context copy"), ("unigram", "Unigram"))
CLOCKS = {"t2e": "the model's predicted gap (T2E)",
          "history_median": "each prompt's median gap (no T2E)"}
_HIDDEN_BLOCK = {"events": HIDDEN, "model": HIDDEN, "context_copy": HIDDEN, "unigram": HIDDEN}


# ============================================================================ #
# The scored steps
# ============================================================================ #
def _copy_ranking(context_tokens, never: set, allowed: set | None = None) -> dict:
    """Pipeline 02's context copy on a prompt (evaluation._context_copy): {code: rank}, its codes
    ranked by their latest position, the most recent first; never-targets left out, and so are
    the codes outside allowed when generation is restricted, as no real event is one of them."""
    ranking = {}
    for token in reversed(list(context_tokens)):
        if token not in never and token not in ranking and (allowed is None or token in allowed):
            ranking[token] = len(ranking) + 1
    return ranking


def _unigram(true_ids: np.ndarray, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline 02's unigram (evaluation._unigram_sums) fitted on true_ids, the scored real
    events: (log probability, rank) of every id, its share of the events and its place when the
    codes are ranked by their count, the lower id first among equal counts."""
    counts = np.bincount(true_ids, minlength=vocab_size).astype(float)
    ids = np.arange(vocab_size)
    rank = np.empty(vocab_size)
    rank[np.lexsort((ids, -counts))] = ids + 1
    log_prob = np.full(vocab_size, -np.inf)          # an id never scored is never looked up
    seen = counts > 0
    log_prob[seen] = np.log(counts[seen] / counts.sum())
    return log_prob, rank


def _step_table(gen_df: pd.DataFrame, stoi: dict[str, int], never: set,
                allowed: set | None) -> pd.DataFrame:
    """One row per generated step, in prompt and step order: the prompt (one per patient and
    anchor), whether the step is scored, the real event's rank under the model, the context copy
    (0 = not in the prompt) and the unigram, and the generated and the real gap and time elapsed
    since the prompt's last event (days). allowed: the codes generation may pick, None for all."""
    df = gen_df[["patient_id", "anchor_age", "step", "true_token", "true_age", "true_token_prob",
                 "true_token_rank", "correct", "cum_overlap", "predicted_time_delta",
                 "generated_age"]].copy()
    # One prompt per (patient, anchor age); dropna=False keeps a full-history prompt (NaN anchor)
    df["prompt"] = df.groupby(["patient_id", "anchor_age"], dropna=False, sort=False).ngroup()
    first = df["step"] == 1
    rankings = dict(zip(df.loc[first, "prompt"],
                        (_copy_ranking(c, never, allowed) for c in gen_df.loc[first, "context_tokens"])))
    df["copy_rank"] = [rankings[p].get(t, 0) for p, t in zip(df["prompt"], df["true_token"])]
    df["scored"] = df["correct"].notna()

    scored = df["scored"]
    true_ids = df.loc[scored, "true_token"].map(stoi).to_numpy(dtype=np.int64)
    log_prob, rank = _unigram(true_ids, len(stoi))
    df.loc[scored, "unigram_rank"] = rank[true_ids]
    df.loc[scored, "unigram_log_prob"] = log_prob[true_ids]

    df = df.sort_values(["prompt", "step"])
    # Step 1's gaps start at the prompt's last event (generated_age - predicted_time_delta; NaN
    # for a time-free model), a later real gap at the real event before it
    start = ((df["generated_age"] - df["predicted_time_delta"]).where(df["step"] == 1)
             .groupby(df["prompt"]).transform("first"))
    df["gen_gap"] = df["predicted_time_delta"]
    df["true_gap"] = np.where(df["step"] == 1, df["true_age"] - start,
                              df["true_age"] - df.groupby("prompt")["true_age"].shift(1))
    df["gen_elapsed"] = df["generated_age"] - start
    df["true_elapsed"] = df["true_age"] - start
    return df


# ============================================================================ #
# The metrics
# ============================================================================ #
def _ranked(rank: pd.Series, top1: pd.Series) -> dict:
    """Top-1, top-5 and top-10 accuracy and the mean reciprocal rank, from the ranks (0 = a
    miss) and the top-1 hits."""
    rank = rank.astype(float)
    hit = rank > 0
    return {"accuracy": float(top1.astype(float).mean()),
            "top5_accuracy": float((hit & (rank <= 5)).mean()),
            "top10_accuracy": float((hit & (rank <= 10)).mean()),
            "mrr": float((1.0 / rank).where(hit, 0.0).mean())}


def _metrics(g: pd.DataFrame) -> dict:
    """Pipeline 02's metrics of the scored steps g: the events, and the ranked metrics of the
    model, the context copy and the unigram, with the cross-entropy and perplexity of the model
    and the unigram (the context copy gives no probabilities). NaN for an empty group."""
    model_ce = float((-np.log(g["true_token_prob"])).mean())
    unigram_ce = float((-g["unigram_log_prob"]).mean())
    return {"events": int(len(g)),
            "model": {"cross_entropy": model_ce, "perplexity": float(np.exp(model_ce)),
                      **_ranked(g["true_token_rank"], g["correct"])},
            "context_copy": _ranked(g["copy_rank"], g["copy_rank"] == 1),
            "unigram": {"cross_entropy": unigram_ce, "perplexity": float(np.exp(unigram_ce)),
                        **_ranked(g["unigram_rank"], g["unigram_rank"] == 1)}}


def _partition(groups: list[pd.DataFrame]) -> list[dict]:
    """_metrics of every group of a partition of the scored steps, under the 5-patient rule."""
    hidden = hidden_groups([set(g["patient_id"]) for g in groups])
    return [dict(_HIDDEN_BLOCK) if h else _metrics(g) for g, h in zip(groups, hidden)]


def _token_metrics(df: pd.DataFrame) -> dict:
    """The scored steps' metrics over all steps, per step (with the mean cumulative overlap) and
    by target cell at step 1 and over steps 2 on."""
    s = df[df["scored"]]
    n_steps = int(df["step"].max())
    by_step = []
    for k, block in enumerate(_partition([s[s["step"] == k] for k in range(1, n_steps + 1)]), start=1):
        overlap = df[(df["step"] == k) & df["cum_overlap"].notna()]
        n = overlap["patient_id"].nunique()
        by_step.append({"step": k, **block, "mean_cum_overlap": (
            float(overlap["cum_overlap"].mean()) if n >= MIN_PATIENTS else HIDDEN if n else float("nan"))})

    new = s["copy_rank"] == 0
    # A step-1 event never has the prompt's last time (the cutoff keeps every event of one time
    # on one side), so a NaN step-1 gap (a time-free model) is later too
    same = s["true_gap"] == 0
    scopes = [("step_1", s["step"] == 1)] + ([("steps_2_on", s["step"] > 1)] if n_steps > 1 else [])
    by_target = {scope: dict(zip([key for key, *_ in CELLS], _partition(
                     [s[in_scope & (new == is_new) & (same == is_same)] for _, is_new, is_same, _ in CELLS])))
                 for scope, in_scope in scopes}
    return {"all_steps": _partition([s])[0], "by_step": by_step, "by_target": by_target}


def _band_shares(gaps: pd.Series, patients: pd.Series) -> list:
    """The share of gaps in every GAP_BANDS band, under the 5-patient rule (a band's patients
    are those of its gaps)."""
    in_band = [(gaps >= lo) & (gaps < hi) for _, lo, hi in GAP_BANDS]
    hidden = hidden_groups([set(patients[m]) for m in in_band])
    return [HIDDEN if h else float(m.mean()) for m, h in zip(in_band, hidden)]


def _quartiles(x: pd.Series) -> list[float]:
    return [float(v) for v in x.quantile([0.25, 0.5, 0.75])]


def _time_metrics(df: pd.DataFrame) -> dict:
    """The generated against the real times, on the steps with a real event: the share of their
    generated and of their real gaps in every band and, per step, the quartiles of both gaps and
    of both elapsed times, their mean absolute errors and the median error of the elapsed time
    (days). A step of fewer than MIN_PATIENTS patients is HIDDEN; no count is written, so none can
    be worked out from it."""
    real = df[df["true_gap"].notna()]
    bands = [{"band": label, "generated_share": gen, "real_share": true}
             for (label, _, _), gen, true in zip(GAP_BANDS, _band_shares(real["gen_gap"], real["patient_id"]),
                                                 _band_shares(real["true_gap"], real["patient_id"]))]
    by_step = []
    for k in range(1, int(df["step"].max()) + 1):
        r = real[real["step"] == k]
        error = r["gen_elapsed"] - r["true_elapsed"]
        values = {"generated_gap_quartiles_days": _quartiles(r["gen_gap"]),
                  "real_gap_quartiles_days": _quartiles(r["true_gap"]),
                  "gap_mae_days": float((r["gen_gap"] - r["true_gap"]).abs().mean()),
                  "generated_elapsed_quartiles_days": _quartiles(r["gen_elapsed"]),
                  "real_elapsed_quartiles_days": _quartiles(r["true_elapsed"]),
                  "elapsed_mae_days": float(error.abs().mean()),
                  "elapsed_median_error_days": float(error.median())}
        if 0 < r["patient_id"].nunique() < MIN_PATIENTS:
            values = dict.fromkeys(values, HIDDEN)
        by_step.append({"step": k, **values})
    return {"gap_bands": bands, "by_step": by_step}


def rollout_metrics(gen_df: pd.DataFrame, stoi: dict[str, int], stats: dict,
                    allowed_token_ids: set[int] | None = None) -> dict:
    """The metrics of gen_df, generate_tokens' rows scored by score_generations (see the module
    docstring): {min_patients, steps, time_source, restricted, tokens, times}, times None for a
    time-free model. stoi and stats are the run's vocabulary (load_vocab); allowed_token_ids is
    the restriction generate_tokens was given (None: none)."""
    never_ids = set(never_target_ids(stats))
    allowed = (None if allowed_token_ids is None
               else {t for t, i in stoi.items() if i in allowed_token_ids})
    df = _step_table(gen_df, stoi, {t for t, i in stoi.items() if i in never_ids}, allowed)
    source = gen_df["time_source"].iloc[0]           # one time source per run
    time_source = None if pd.isna(source) else str(source)
    return {"min_patients": MIN_PATIENTS, "steps": int(df["step"].max()), "time_source": time_source,
            "restricted": allowed is not None,
            "tokens": _token_metrics(df), "times": _time_metrics(df) if time_source else None}


# ============================================================================ #
# The report
# ============================================================================ #
_TABLE_HEAD = (f"  {'':<20}{'events':>9}  model: acc top5 top10 MRR (ppl)  |  "
               "context copy: acc top5 top10 MRR  |  unigram: acc top5 top10 MRR (ppl)")


def _row(name: str, block: dict) -> str:
    """One line of a token table: a group's events and its predictors' metrics."""
    if block["events"] in (HIDDEN, 0):
        return f"  {name:<20}{block['events']:>9}"
    cells = []
    for predictor, _ in PREDICTORS:
        m = block[predictor]
        cell = " ".join(f"{m[k]:.3f}" for k in ("accuracy", "top5_accuracy", "top10_accuracy", "mrr"))
        cells.append(cell + (f" ({m['perplexity']:.2f})" if "perplexity" in m else ""))
    return f"  {name:<20}{block['events']:>9,}  " + "  |  ".join(cells)


def _fmt(v) -> str:
    """A reported share: HIDDEN as is, NaN (nothing to measure) as '-'."""
    if v == HIDDEN:
        return HIDDEN
    return "-" if math.isnan(v) else f"{v:.3f}"


def rollout_report(metrics: dict) -> str:
    """The text report of rollout_metrics' result (generation_metrics.txt)."""
    tokens, times = metrics["tokens"], metrics["times"]
    lines = ["=" * 60, "GENERATION METRICS: the rollouts against the real continuation", "=" * 60,
             f"{metrics['steps']} generated step(s) per prompt; clock: "
             f"{CLOCKS.get(metrics['time_source'], 'none (a time-free model)')}",
             "Scored: the steps whose real event is known and not <UNK>. Model top-1 is the generated",
             "token; top-5/10, MRR and perplexity use the step's full distribution. From step 2 the",
             "model reads its own generated history (rollout scores, not pipeline 02's teacher-forced",
             "ones). References: the context copy (the prompt's codes, the latest first) and the",
             "unigram (fitted on the scored real events). Step 1's real event is the first after the",
             "cutoff: with remove_cutoff_token true and an anchor on an event, the anchor's own event.",
             f"{HIDDEN}: hidden under the 5-patient rule."]
    if metrics["restricted"]:
        lines += ["Restricted (allowed_token_names): the real events and the context copy hold only the",
                  "allowed codes, the generated token is picked among them; the model's top-5/10, MRR",
                  "and perplexity still use the full distribution, so its top-1 can exceed its top-5."]
    lines += ["", "Tokens", _TABLE_HEAD, _row("all steps", tokens["all_steps"])]
    lines += [_row(f"step {r['step']}", r) for r in tokens["by_step"]]
    lines.append("  mean cumulative overlap by step: " + ", ".join(
        f"{r['step']}: {_fmt(r['mean_cum_overlap'])}" for r in tokens["by_step"]))
    for scope, title in (("step_1", "step 1"), ("steps_2_on", "steps 2 on")):
        if scope in tokens["by_target"]:
            lines += ["", f"Tokens by target, {title}", _TABLE_HEAD]
            lines += [_row(label, tokens["by_target"][scope][key]) for key, *_, label in CELLS]
    if times is None:
        return "\n".join(lines + ["", "Times: none (a time-free model generates no times)"])

    lines += ["", "Gaps by band (the steps with a real event: share of their generated gaps; of their real gaps)"]
    lines += [f"  {b['band']:<12}generated {_fmt(b['generated_share']):>6}  real {_fmt(b['real_share']):>6}"
              for b in times["gap_bands"]]
    lines += ["", "Times by step (days; median [25th-75th percentile]; real = the patient's own continuation)"]
    for r in times["by_step"]:
        if r["gap_mae_days"] == HIDDEN or math.isnan(r["gap_mae_days"]):
            lines.append(f"  step {r['step']:>2}: {HIDDEN if r['gap_mae_days'] == HIDDEN else 'no real event'}")
            continue
        q = {k: "{1:.4g} [{0:.4g}-{2:.4g}]".format(*r[k]) for k in r if k.endswith("quartiles_days")}
        lines.append(f"  step {r['step']:>2}: gap generated {q['generated_gap_quartiles_days']}, real "
                     f"{q['real_gap_quartiles_days']} (MAE {r['gap_mae_days']:.4g}); elapsed generated "
                     f"{q['generated_elapsed_quartiles_days']}, real {q['real_elapsed_quartiles_days']} "
                     f"(MAE {r['elapsed_mae_days']:.4g}, median error {r['elapsed_median_error_days']:+.4g})")
    return "\n".join(lines)


# ============================================================================ #
# The figures
# ============================================================================ #
def _shown(block: dict) -> bool:
    """Whether a group's metrics may be drawn: neither hidden nor empty."""
    return block["events"] not in (HIDDEN, 0)


def _number(v) -> float:
    """v as a number to draw, NaN when it is HIDDEN or not finite."""
    return v if isinstance(v, (int, float)) and math.isfinite(v) else float("nan")


def plot_tokens_by_step(metrics: dict, out_stem) -> bool:
    """out_stem.png/.svg: per generated step, (a) top-1 accuracy with the model's cumulative
    overlap, (b) top-10 accuracy, (c) mean reciprocal rank of the model, the context copy and the
    unigram, (d) the perplexity of the real event under the model and the unigram. False (no
    figure) when no step may be drawn."""
    rows = metrics["tokens"]["by_step"]
    if not any(_shown(r) for r in rows):
        return False
    steps = [r["step"] for r in rows]

    def series(predictor, key):
        return [r[predictor][key] if _shown(r) else float("nan") for r in rows]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 4.6), constrained_layout=True)
        for ax, (key, title) in zip(axes.flat, (("accuracy", "a  Top-1 accuracy"),
                                                ("top10_accuracy", "b  Top-10 accuracy"),
                                                ("mrr", "c  Mean reciprocal rank"))):
            for i, (predictor, label) in enumerate(PREDICTORS):
                ax.plot(steps, series(predictor, key), marker="o", markersize=3, color=f"C{i}", label=label)
            ax.set_ylim(0, 1)
            ax.set_title(title, loc="left", fontweight="bold")
        axes[0, 0].plot(steps, [_number(r["mean_cum_overlap"]) for r in rows], linestyle="--",
                        color="C0", label="Model, cumulative overlap")
        for i, (predictor, label) in enumerate(PREDICTORS):
            if predictor != "context_copy":             # no probabilities
                axes[1, 1].plot(steps, series(predictor, "perplexity"), marker="o", markersize=3,
                                color=f"C{i}", label=label)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("d  Perplexity of the real event", loc="left", fontweight="bold")
        for ax in axes.flat:
            ax.set_xlabel("Generated step")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(frameon=False)
    save_figure(fig, out_stem)
    return True


def plot_tokens_by_target(metrics: dict, out_stem) -> bool:
    """out_stem.png/.svg: the top-1 accuracy of the model, the context copy and the unigram in
    each of pipeline 02's four target cells, (a) at step 1 and (b) over steps 2 on. A cell that
    may not be drawn is labelled. False (no figure) when no cell may be drawn."""
    by_target = metrics["tokens"]["by_target"]
    scopes = [(key, title) for key, title in (("step_1", "Step 1"), ("steps_2_on", "Steps 2 on"))
              if key in by_target]
    if not any(_shown(block) for key, _ in scopes for block in by_target[key].values()):
        return False
    x = np.arange(len(CELLS))
    width = 0.8 / len(PREDICTORS)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(scopes), figsize=(FIG_WIDTH, 2.8), sharey=True, squeeze=False,
                                 constrained_layout=True)
        for letter, ax, (key, title) in zip("ab", axes[0], scopes):
            cells = [by_target[key][cell] for cell, *_ in CELLS]
            for i, (predictor, label) in enumerate(PREDICTORS):
                ax.bar(x + (i - 1) * width, [c[predictor]["accuracy"] if _shown(c) else float("nan")
                                             for c in cells], width, color=f"C{i}", label=label)
            for xi, c in zip(x, cells):
                if not _shown(c):
                    ax.text(xi, 0.02, f"{HIDDEN} patients" if c["events"] == HIDDEN else "No events",
                            ha="center", va="bottom", fontsize=6, color="0.4")
            ax.set_xticks(x, [label.replace(", ", ",\n") for *_, label in CELLS])
            ax.set_ylim(0, 1)
            ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
        axes[0, 0].set_ylabel("Top-1 accuracy")
        axes[0, 0].legend(frameon=False)
    save_figure(fig, out_stem)
    return True


def plot_times(metrics: dict, out_stem) -> bool:
    """out_stem.png/.svg: (a) the share of the generated and the real gaps in every gap band, on
    the steps with a real event,
    (b) per generated step the time elapsed since the prompt's last event, generated and real
    (median and 25th-75th percentile). False (no figure) when nothing may be drawn."""
    times = metrics["times"]
    bands, steps = times["gap_bands"], times["by_step"]
    kinds = (("generated", "Generated", "C0"), ("real", "Real", "0.45"))
    if all(math.isnan(_number(b[f"{kind}_share"])) for b in bands for kind, *_ in kinds) and \
            all(math.isnan(_number(r["elapsed_mae_days"])) for r in steps):
        return False
    with plt.rc_context(STYLE):
        fig, (ax_band, ax_time) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.8), constrained_layout=True)
        x = np.arange(len(bands))
        for k, (kind, label, color) in enumerate(kinds):
            offset = (k - 0.5) * 0.4
            ax_band.bar(x + offset, [_number(b[f"{kind}_share"]) for b in bands], 0.4, color=color, label=label)
            for xi, b in zip(x, bands):
                if b[f"{kind}_share"] == HIDDEN:
                    ax_band.text(xi + offset, 0.005, HIDDEN, ha="center", va="bottom", fontsize=6, color="0.4")
        ax_band.set_xticks(x, [b["band"] for b in bands], rotation=30, ha="right")
        ax_band.set_ylabel("Share of the gaps")
        ax_band.set_title("a  Gap to the next event", loc="left", fontweight="bold")
        ax_band.legend(frameon=False)
        s = [r["step"] for r in steps]
        for kind, label, color in kinds:
            q = np.array([[_number(v) for v in r[f"{kind}_elapsed_quartiles_days"]]
                          if isinstance(r[f"{kind}_elapsed_quartiles_days"], list) else [float("nan")] * 3
                          for r in steps])
            ax_time.plot(s, q[:, 1], marker="o", markersize=3, color=color, label=f"{label}, median")
            ax_time.fill_between(s, q[:, 0], q[:, 2], color=color, alpha=0.2, linewidth=0)
        ax_time.set_yscale("symlog", linthresh=MINUTE)   # log above one minute; a zero still shows
        ax_time.set_xlabel("Generated step")
        ax_time.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_time.set_ylabel("Days since the prompt's last event")
        ax_time.set_title("b  Elapsed time", loc="left", fontweight="bold")
        ax_time.legend(frameon=False)
        for ax in (ax_band, ax_time):
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Clock: {CLOCKS[metrics['time_source']]}", fontsize=8)
    save_figure(fig, out_stem)
    return True


def write_rollout_metrics(gen_df: pd.DataFrame, stoi: dict[str, int], stats: dict, output_dir,
                          allowed_token_ids: set[int] | None = None) -> None:
    """Score gen_df (generate_tokens' rows after score_generations, allowed_token_ids its
    restriction) and write to output_dir generation_metrics.json, generation_metrics.txt (the
    report, also printed) and the figures that have something to show. The two files come
    first, so a plotting failure never costs them."""
    output_dir = Path(output_dir)
    metrics = rollout_metrics(gen_df, stoi, stats, allowed_token_ids)
    save_metrics(metrics, output_dir / "generation_metrics.json")
    report = rollout_report(metrics)
    (output_dir / "generation_metrics.txt").write_text(report + "\n")
    print("\n" + report)
    figures = [("generation_tokens_by_step", plot_tokens_by_step),
               ("generation_tokens_by_target", plot_tokens_by_target)]
    if metrics["times"] is not None:
        figures.append(("generation_times", plot_times))
    for name, plot in figures:
        drawn = plot(metrics, output_dir / name)
        print(f"{name}: " + (f"saved to {output_dir / name}.png and .svg" if drawn else
                             f"not drawn (no group of at least {MIN_PATIENTS} patients to show)"))
