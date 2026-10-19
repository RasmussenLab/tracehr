#!/usr/bin/env python3
# src/tracehr/pipelines/outcome_probe_prediction.py
"""Pipeline 05: downstream outcome prediction from a trained NTPTransformer, used as a
frozen feature extractor. For each outcomes table (patient id, anchor timestamp, target)
every anchor row's history is cut at its anchor (the cutoff rule every anchored pipeline
shares), embedded, and six models are compared.
Demographics = [Sex, age at last token]; BoW = counts over the full truncated history:
  1. Demographics             -> linear (LogReg / Ridge)     (the floor)
  2. Embedding                -> linear                      (linear probe)
  3. Embedding + demographics -> linear
  4. Embedding                -> XGBoost
  5. Embedding + demographics -> XGBoost
  6. BoW + demographics       -> XGBoost
5 vs 6 is the controlled representation comparison: same learner, same params, same
demographic block. Every arm is reported on cross-validation (shared folds) and on test.

Three tasks share those six arms. Binary vs regression is autodetected per outcomes file
from its target column. Survival cannot be autodetected - its label is two columns - so
it is selected by naming anchor_columns.censor_timestamp_col, after which target_col is
the 0/1 event indicator and the arms become penalised CoxPH / XGBoost survival:cox,
scored by Harrell's C-index against the Sex+Age floor (cindex_lift_over_sex_age in the
summary).

Two modes, both reading the MEDS input (encoded on the fly):
  streaming          one ordered pass; windows are built and embedded in flight
  save_intermediate  the anchor-truncated windows are written once to a reusable store,
                     then embedded from it - so a pooling / combine_windows / checkpoint
                     sweep pays the scan only once

Next-token test metrics and the vocabulary token-embedding analysis are pipeline 02.
"""

import argparse
from pathlib import Path

import yaml

from tracehr.dataset.column_mapping import resolve_columns
from tracehr.dataset.meds_reader import load_event_times
from tracehr.generation.model_io import load_checkpoint, save_metrics
from tracehr.probing.window_floor import report_head_window, window_floor_from_checkpoint
from tracehr.training.runtime import resolve_device


def main(config_path: str):
    """Run the downstream comparison of every outcomes file in the yaml at config_path and
    write outcome_summary.json."""
    # Imported lazily: tracehr/__init__.py imports the whole pipelines package, so a
    # module-level import here would pull xgboost into EVERY tracehr import - and
    # on macOS torch + xgboost together segfault the process (libomp).
    from tracehr.probing.embedding_outcome_prediction import (
        run_downstream_stage,
        summarize_downstream,
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    ocfg = config["outcome"]

    device_str = resolve_device(config["hardware"]["device"])
    print(f"Using device: {device_str}")

    output_dir = Path(ocfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    model, checkpoint = load_checkpoint(
        checkpoint_path=config["checkpoint"]["model_path"],
        vocab_path=config["checkpoint"]["vocab_path"],
        device=device_str,
    )
    print(f"Model loaded from epoch {checkpoint.get('epoch', '?')} "
          f"(loss_mode={model.loss_mode}, t2e_weight={model.t2e_weight:g})")

    # Window floor: taken from the checkpoint, never from this config. Windowing here has to
    # match what the model was trained on, so the training value is the only correct one and
    # a local knob could only ever disagree with it.
    min_seq_len = window_floor_from_checkpoint(checkpoint)
    report_head_window(ocfg.get("start_from_end", True), bool(ocfg.get("add_head_window", False)))

    # The input's column names, validated against its schema
    cols = resolve_columns(config, parquet_path=config["data"]["parquet_path"])
    # Each patient's rows are checked once; the readers leave out the rows and patients that
    # fail and compute time_age and time_abs from the timestamps (the anchors too)
    event_times = load_event_times(config["data"]["parquet_path"], cols,
                                   config["checkpoint"]["vocab_path"])

    results, tasks = run_downstream_stage(
        model, ocfg, config,
        raw_parquet=config["data"]["parquet_path"],
        cols=cols,
        out_dir=output_dir,
        vocab_path=config["checkpoint"]["vocab_path"],
        min_seq_len=min_seq_len,
        event_times=event_times,
    )

    summary = {"downstream": summarize_downstream(results, tasks), "tasks": tasks}
    save_metrics(summary, output_dir / "outcome_summary.json")
    print(f"\nOutcome summary saved to {output_dir / 'outcome_summary.json'}")


def cli():
    """Console entry point (tracehr-outcome): run main on --config."""
    parser = argparse.ArgumentParser(
        description="Downstream outcome prediction from a trained NTPTransformer")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the stage's YAML config")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
