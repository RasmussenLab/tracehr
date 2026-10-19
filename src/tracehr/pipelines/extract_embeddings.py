#!/usr/bin/env python3
# src/tracehr/pipelines/extract_embeddings.py
"""Anchor-grid embedding extraction from the MEDS input.

Inputs: the MEDS input and an anchors parquet with a patient id and an anchor timestamp
(an outcomes file without its target column works). Each anchor row's history is cut at
that anchor's time_age and the transformer embedding of that history is extracted (a
patient may have several anchors; a missing anchor embeds the full history). With a positive `timestep` (DAYS),
embeddings are extracted at grid anchors anchor_0 + s*timestep for
s = 0, 1, 2, ... until the complete history is consumed - a longitudinal
embedding trajectory per patient.

Outputs: anchor_embeddings.parquet, plus embedding_extraction_meta.json holding the
settings that describe the run (pooling, window aggregation, embedding_dim, ...) rather
than the patient. Its columns match pipeline 05's downstream_embeddings.parquet wherever
they mean the same thing - id, anchor_age, last_token_age, Sex, Embedding, bow_tokens,
bow_token_counts, n_windows, and the opt-in Sequence/Age - so the two are joinable;
step_index/timestep/anchor_offset/is_final_step are the grid-only extras.

DAYS CONTRACT: Age, anchors and timestep are all in DAYS.
Cutoff contract (same as pipelines 03/05/06): kept events satisfy
Age < (anchor - time_before_event); the event exactly AT the anchor age is
EXCLUDED. remove_cutoff_token: false flips to <=.
"""

import argparse
from pathlib import Path

import yaml

from tracehr.dataset.cohort import (
    ANCHOR_KEEPS_FULL_HISTORY,
    anchors_by_patient,
    load_anchors,
    report_missing_anchors,
)
from tracehr.dataset.column_mapping import enabled_col, resolve_anchor_columns, resolve_columns
from tracehr.dataset.meds_reader import load_event_times
from tracehr.dataset.vocabulary import load_vocab
from tracehr.generation.model_io import load_checkpoint
from tracehr.probing.embeddings import extract_anchor_grid_embeddings, write_extraction_meta
from tracehr.probing.window_floor import (
    reject_min_seq_len_key,
    report_head_window,
    window_floor_from_checkpoint,
)
from tracehr.training.runtime import resolve_device


def main(config_path: str):
    """Extract the anchor(-grid) embeddings of the yaml at config_path and write them, with
    the run-settings sidecar, to extraction.output_dir."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device_str = resolve_device(config["hardware"]["device"])
    print(f"Using device: {device_str}")

    ecfg = config["extraction"]
    output_dir = Path(ecfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestep = ecfg.get("timestep")
    print("[embeddings] time_age, the anchors and timestep are in DAYS (time_age and the "
          "anchors computed from the timestamps)")

    print("\nLoading vocabulary...")
    stoi, itos, stats = load_vocab(config["data"]["vocab_path"])
    print(f"Vocabulary size: {len(itos)}")

    print("\nLoading model...")
    model, checkpoint = load_checkpoint(
        checkpoint_path=config["checkpoint"]["model_path"],
        vocab_path=config["data"]["vocab_path"],
        device=device_str,
    )
    print(f"Model loaded from epoch {checkpoint.get('epoch', '?')} "
          f"(loss_mode={model.loss_mode}, t2e_weight={model.t2e_weight:g})")

    # Window floor: from the checkpoint, never from this config - the windows built here
    # have to match the geometry the model was trained on. Same lookup pipeline 05 uses.
    # A key here could only disagree with training, so reject it rather than ignore it.
    reject_min_seq_len_key(ecfg, "extraction")
    min_seq_len = window_floor_from_checkpoint(checkpoint)
    add_head_window = bool(ecfg.get("add_head_window", False))
    report_head_window(ecfg.get("start_from_end", True), add_head_window)

    cols = resolve_columns(config, parquet_path=ecfg["raw_parquet_path"])
    acols = resolve_anchor_columns(config, parquet_path=ecfg["anchors_path"])
    id_col, age_col = acols["id_col"], acols["age_col"]
    time_abs_col = enabled_col(cols, "abstime_col", model.use_time_abs, "model.use_time_abs")
    event_times = load_event_times(ecfg["raw_parquet_path"], cols, config["data"]["vocab_path"])

    # Anchors (their time_age computed from the anchor timestamps)
    anchors = load_anchors(ecfg["anchors_path"], id_col, age_col,
                           event_times=event_times, timestamp_col=acols["timestamp_col"])
    report_missing_anchors(anchors[age_col], ANCHOR_KEEPS_FULL_HISTORY)

    # Anchor-grid extraction from the MEDS input -> one row per (patient, anchor, grid step).
    # One resolved-settings dict drives the extraction AND the sidecar, so the file beside
    # the parquet can never drift from the run that produced it.
    settings = dict(
        timestep=timestep,
        pooling=ecfg.get("pooling", "mean"),
        overlap_percent=ecfg.get("overlap_percent", 0),
        start_from_end=ecfg.get("start_from_end", True),
        combine_windows=ecfg.get("combine_windows", "recency_weighted"),
        batch_size=ecfg.get("batch_size", 64),
        time_before_event=ecfg.get("time_before_event", 0),
        remove_cutoff_token=ecfg.get("remove_cutoff_token", True),
        save_sequences=ecfg.get("save_sequences", False),
        min_seq_len=min_seq_len,
        add_head_window=add_head_window,
    )
    # The rows are written to out_path a chunk at a time as they are produced, never held
    # all at once; what comes back is the row count and embedding width for the sidecar.
    out_path = output_dir / "anchor_embeddings.parquet"
    written = extract_anchor_grid_embeddings(
        model,
        parquet_path=ecfg["raw_parquet_path"],
        vocab_path=config["data"]["vocab_path"],
        anchors_by_pid=anchors_by_patient(anchors, id_col, age_col),
        id_col=cols["id_col"],
        tok_col=cols["tok_col"],
        time_age_col=cols["age_col"],
        time_abs_col=time_abs_col,
        type_col=enabled_col(cols, "type_col", model.use_type, "model.use_type"),
        out_path=out_path,
        event_times=event_times,
        **settings,
    )

    write_extraction_meta(output_dir, None, id_col=cols["id_col"],
                          max_seq_len=model.max_seq_len, use_time_abs=model.use_time_abs,
                          use_type=model.use_type, **settings, **written)
    print(f"\nComplete! Saved {written['n_rows']} embedding rows -> {out_path}")


def cli():
    """Console entry point (tracehr-embed): run main on --config."""
    parser = argparse.ArgumentParser(description="Anchor-grid embedding extraction from MEDS input")
    parser.add_argument("--config", type=str, required=True, help="Path to the stage's YAML config")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
