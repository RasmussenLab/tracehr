#!/usr/bin/env python3
# src/tracehr/pipelines/generate_tokens.py
"""Autoregressive generation from anchor-truncated patient histories.

Inputs: the MEDS input and an anchors parquet with a patient id and an anchor
timestamp (an outcomes file without its target column works). Each anchor row is one
prompt: the patient's history cut at that anchor's time_age (the cutoff rule every anchored
pipeline shares; a patient may have several anchors, a missing anchor keeps the full
history, and the run reports how many there are), from which the model greedily generates
num_generated_tokens events (1-20, default 1). Every prediction step is logged to
generations.parquet: context seen, position, predicted token, the FULL raw
probability distribution keyed by token string, and the time provenance. Each step is
also scored against the real continuation right after the prompt (true_token /
true_age, correct, cum_accuracy / cum_overlap; empty past the end of the record).
Optional explainability (explainability.enabled) attributes every generated token to
the context it was generated from (attr_<method> / top_<method> columns, optional plots).
"""

import argparse
from pathlib import Path

import yaml

from tracehr.dataset.cohort import ANCHOR_KEEPS_FULL_HISTORY, load_anchors, report_missing_anchors
from tracehr.dataset.column_mapping import enabled_col, resolve_anchor_columns, resolve_columns
from tracehr.dataset.meds_reader import load_event_times
from tracehr.dataset.streaming_datasets import StreamingAnchorsDataset
from tracehr.dataset.vocabulary import load_vocab, type_id_map
from tracehr.generation.explainability import plot_generated_explanations, resolve_explainability
from tracehr.generation.model_io import load_checkpoint
from tracehr.generation.predict import (
    allowed_token_ids_from_names,
    generate_tokens,
    score_generations,
    token_modal_types,
    write_generations_parquet,
)
from tracehr.training.runtime import resolve_device


def _report_continuation(gen_df, n_gen: int) -> None:
    """Print the cohort summary of the comparison with the real continuation: how many
    prompts have events after them, the accuracy at each step, and the mean cumulative
    accuracy and overlap over the n_gen steps."""
    scored = gen_df[gen_df["correct"].notna()]
    if scored.empty:
        print("Real continuation: no prompt has a scorable event after it (columns left empty)")
        return
    final = gen_df[gen_df["step"] == n_gen]                     # each prompt's cumulative scores
    n_scored = scored.groupby(["patient_id", "anchor_age"], dropna=False).ngroups
    print(f"Real continuation: {n_scored} of {len(final)} prompts have events after the prompt")
    by_step = scored.groupby("step")["correct"].agg(["mean", "size"])
    print("  accuracy by step: " + ", ".join(
        f"{step}: {mean:.3f} (n={n})" for step, mean, n in by_step.itertuples()))
    print(f"  over {n_gen} step(s): mean cum_accuracy {final['cum_accuracy'].mean():.3f}, "
          f"mean cum_overlap {final['cum_overlap'].mean():.3f}")


def main(config_path: str):
    """Generate from every anchor of the yaml at config_path and write generations.parquet."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device_str = resolve_device(config["hardware"]["device"])
    print(f"Using device: {device_str}")

    pcfg = config["prediction"]
    output_dir = Path(pcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gen = pcfg.get("num_generated_tokens")
    n_gen = 1 if n_gen is None else int(n_gen)           # None/absent -> 1 (next position only)
    if not 1 <= n_gen <= 20:
        raise ValueError(f"num_generated_tokens must be in [1, 20], got {n_gen}: it is the number of "
                         f"tokens generated per prompt (1 = predict only the next token).")

    # Optional explainability, validated before the slow steps (None = off)
    explain = resolve_explainability(config.get("explainability"))
    print("Explainability: off" if explain is None else
          f"Explainability: on ({', '.join(explain['methods'])}; n_steps={explain['n_steps']}, "
          f"top_k_tokens={explain['top_k_tokens']}, plot_first_k={explain['plot_first_k']})")

    print("\nLoading vocabulary...")
    stoi, itos, stats = load_vocab(config["data"]["vocab_path"])
    print(f"Vocabulary size: {len(itos)}")

    # Optional restriction of the generated tokens, by name (resolved before the slow steps).
    # The id-based key it replaced is refused when set: ignoring it would silently lift the
    # restriction.
    if pcfg.get("allowed_token_ids"):
        raise ValueError("prediction.allowed_token_ids was replaced by allowed_token_names, which "
                         "lists token names (e.g. ['e11.9', 'death']) instead of vocabulary ids. "
                         "Rename the key and list the tokens by name.")
    allowed_ids = allowed_token_ids_from_names(
        pcfg.get("allowed_token_names"), stoi,
        stats.get("sex_token_ids", []) + stats.get("static_token_ids", []))
    if allowed_ids is not None:
        print(f"Generation restricted to {len(allowed_ids):,} token(s) from allowed_token_names "
              f"(probs stay full-vocabulary)")

    print("\nLoading model...")
    model, checkpoint = load_checkpoint(
        checkpoint_path=config["checkpoint"]["model_path"],
        vocab_path=config["data"]["vocab_path"],
        device=device_str,
    )
    print(f"Model loaded from epoch {checkpoint.get('epoch', '?')} "
          f"(loss_mode={model.loss_mode}, t2e_weight={model.t2e_weight:g})")

    # Anchor-truncated prompt windows streamed from the MEDS input (encoded on the fly, cut
    # by the cutoff rule every anchored pipeline shares; no intermediate file)
    raw_parquet = pcfg["raw_parquet_path"]
    cols = resolve_columns(config, parquet_path=raw_parquet)
    acols = resolve_anchor_columns(config, parquet_path=pcfg["anchors_path"])
    id_col, age_col = acols["id_col"], acols["age_col"]
    time_abs_col = enabled_col(cols, "abstime_col", model.use_time_abs, "model.use_time_abs")
    event_times = load_event_times(raw_parquet, cols, config["data"]["vocab_path"])
    anchors = load_anchors(pcfg["anchors_path"], id_col, age_col,
                           event_times=event_times, timestamp_col=acols["timestamp_col"])
    gen_ds = StreamingAnchorsDataset(
        parquet_path=raw_parquet,
        vocab_path=config["data"]["vocab_path"],
        anchors=anchors,
        patient_ids=set(anchors[id_col]),
        time_before_event=pcfg.get("time_before_event", 0),
        remove_cutoff_token=pcfg.get("remove_cutoff_token", True),
        max_seq_len=model.max_seq_len,
        overlap_percent=0,
        id_col=cols["id_col"],
        tok_col=cols["tok_col"],
        time_age_col=cols["age_col"],
        time_abs_col=time_abs_col,
        type_col=enabled_col(cols, "type_col", model.use_type, "model.use_type"),
        outcome_patient_col=id_col,
        outcome_age_col=age_col,
        start_from_end=True,
        final_windows_only=True,
        event_times=event_times,
    )
    print(f"Prompt windows: {len(gen_ds)} prompts ({len(set(gen_ds.patient_table))} patients) "
          f"survived truncation")
    if len(gen_ds) == 0:
        raise ValueError("No anchored patient has an event before their cutoff (anchor age - "
                         "time_before_event), so there is nothing to generate from. Check the anchor "
                         "ages and time_before_event.")
    # A missing (NaN) anchor keeps the whole history, so generation predicts the next event
    # after the end of the record.
    report_missing_anchors(gen_ds.cutoff_table, ANCHOR_KEEPS_FULL_HISTORY)

    # Rollout and save (one row per prediction step per prompt)
    type_to_id = type_id_map(stats) if model.use_type else {}
    type_lookup = (token_modal_types(raw_parquet, stoi, int(stoi["<UNK>"]), type_to_id, len(itos),
                                     event_times, tok_col=cols["tok_col"],
                                     type_col=cols["type_col"])
                   if model.use_type else None)
    print(f"\nGenerating {n_gen} token(s) per prompt (greedy, deterministic)...")
    gen_df = generate_tokens(
        model, gen_ds, itos, stoi,
        num_tokens=n_gen,
        batch_size=pcfg["batch_size"],
        type_lookup=type_lookup,
        allowed_token_ids=allowed_ids,
        explain=explain,
    )
    gen_df = score_generations(gen_df)
    out_path = output_dir / "generations.parquet"
    write_generations_parquet(gen_df, out_path)
    print(f"\nComplete! Saved {len(gen_df)} generation steps ({len(gen_ds)} prompts x {n_gen}) -> {out_path}")

    _report_continuation(gen_df, n_gen)

    # Plots after the table is saved, so a plotting failure never costs the generations
    if explain is not None and explain["plot_first_k"] > 0:
        plot_dir = output_dir / "explainability_plots"
        n_plots = plot_generated_explanations(gen_df, explain["methods"], explain["plot_first_k"], plot_dir)
        print(f"Saved {n_plots} explainability plot(s) -> {plot_dir}")


def cli():
    """Console entry point (tracehr-generate): run main on --config."""
    parser = argparse.ArgumentParser(description="Autoregressive generation from anchor-truncated histories")
    parser.add_argument("--config", type=str, required=True, help="Path to the stage's YAML config")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
