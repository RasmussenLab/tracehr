#!/usr/bin/env python3
# src/tracehr/pipelines/evaluate_model.py
"""Held-out evaluation of a trained NTPTransformer. Two jobs, both scored from the
checkpoint alone:
  1. Next-token test metrics (loss/perplexity/accuracy/top-k/MRR, plus t2e MAE when the
     model predicts time-to-event) over the test split -> test_metrics.json/.png
  2. Vocabulary token-embedding analysis (optional, evaluation.token_analysis): what the
     model learned about each token -> token_embeddings/

Downstream outcome prediction from patient embeddings is a SEPARATE pipeline
(05, outcome_probe_prediction); it needs an outcomes table and its own id splits.
"""

import argparse
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from tracehr.dataset.column_mapping import enabled_col, resolve_columns
from tracehr.dataset.meds_reader import load_event_times
from tracehr.dataset.ntp_batching import NTPDataset, collate_fn
from tracehr.dataset.streaming_datasets import RawStreamingNTPDataset
from tracehr.dataset.vocabulary import vocab_fingerprint
from tracehr.dataset.windows import open_window_store, window_store_rescoring
from tracehr.generation.evaluation import evaluate_model, plot_evaluation_metrics, print_metrics
from tracehr.generation.model_io import load_checkpoint, save_metrics
from tracehr.training.runtime import resolve_device


def _streaming_test_dataset(config: dict, cols: dict, checkpoint: dict, model):
    """(test dataset, event times): the test split's windows streamed on the fly from the
    MEDS input. Window geometry and model flags come from the checkpoint's training config
    so the test windows match what the model was trained on; the column names come from
    THIS config's `columns:` block (already schema-validated)."""
    train_cfg = checkpoint.get("config", {})
    model_cfg = train_cfg.get("model", {})
    seq_cfg = train_cfg.get("sequences", {})
    time_abs_col = enabled_col(cols, "abstime_col", model_cfg.get("use_time_abs", True),
                               "model.use_time_abs")
    event_times = load_event_times(config["data"]["parquet_path"], cols,
                                   config["checkpoint"]["vocab_path"])
    test_dataset = RawStreamingNTPDataset(
        parquet_path=config["data"]["parquet_path"],
        vocab_path=config["checkpoint"]["vocab_path"],
        ids_path=config["data"].get("test_ids"),
        max_seq_len=model_cfg.get("max_seq_len", model.max_seq_len),
        # Metrics must score each held-out token exactly once: use non-overlapping
        # partition windows. Training overlap AND the extra add_final_window tail
        # window both re-cover boundary/tail tokens, which would silently inflate the
        # token-weighted perplexity/accuracy. (Training itself keeps its overlap.)
        overlap_percent=0,
        add_final_window=False,
        id_col=cols["id_col"],
        tok_col=cols["tok_col"],
        time_age_col=cols["age_col"],
        time_abs_col=time_abs_col,
        type_col=enabled_col(cols, "type_col", model_cfg.get("use_type", True), "model.use_type"),
        min_seq_len=seq_cfg.get("min_seq_len", 1),
        compute_time_deltas=model.t2e_pred,
        shuffle_buffer_windows=0,
        event_times=event_times,
    )
    return test_dataset, event_times


def _store_test_dataset(config: dict, model) -> NTPDataset:
    """The test split's windows from the prebuilt window store data.test_npz, refused unless
    it was built from the checkpoint's vocabulary, carries type ids when the model uses them,
    fits the model's context and scores every test token exactly once."""
    test_dataset = NTPDataset(
        config["data"]["test_npz"],
        compute_time_deltas=model.t2e_pred,
    )
    # Fail fast: the windows must come from the checkpoint's vocabulary - its token ids,
    # front block and time origin - or the model would read them as other tokens
    store_meta = open_window_store(config["data"]["test_npz"])["meta"][0]
    stored_fp = store_meta.get("vocab_hash")
    if stored_fp != vocab_fingerprint(config["checkpoint"]["vocab_path"]):
        raise ValueError(
            f"{config['data']['test_npz']} was built with another vocabulary than "
            f"{config['checkpoint']['vocab_path']} (fingerprint {stored_fp!r}). Regenerate "
            f"it with create_sequence_windows(vocab_path=...) on this vocabulary, or set "
            f"data.streaming: true.")
    # Fail fast: a model with types would read a type-less store as type 0 everywhere
    if model.use_type and not store_meta.get("has_types"):
        raise ValueError(
            f"{config['data']['test_npz']} holds no type ids (built without a type column, "
            f"or before stores recorded it), but the model uses types. Regenerate it with "
            f"create_sequence_windows(type_col=...), or set data.streaming: true.")
    # Fail fast: windows wider than the model's context would crash mid-evaluation
    npz_width = test_dataset.input_ids.shape[1]
    if npz_width > model.max_seq_len:
        raise ValueError(
            f"{config['data']['test_npz']} has {npz_width}-wide windows but the model's "
            f"max_seq_len is {model.max_seq_len}. Regenerate the test windows with "
            f"create_sequence_windows(max_seq_len={model.max_seq_len}, ...)."
        )
    # Fail fast: test metrics must score each held-out token exactly once, as the
    # streaming path does. Overlapping or tail windows (training geometry) re-score
    # low-context copies of some tokens and silently bias every metric.
    rescoring = window_store_rescoring(config["data"]["test_npz"])
    if rescoring:
        raise ValueError(
            f"{config['data']['test_npz']} would score some test tokens more than once: "
            f"{rescoring}. Regenerate it as a partition with "
            f"create_sequence_windows(..., overlap_percent=0, add_final_window=False), "
            f"or set data.streaming: true."
        )
    return test_dataset


def main(config_path: str):
    """Evaluate the checkpoint named in the yaml at config_path on the test split."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # The input's column names, validated against its schema
    cols = resolve_columns(config, parquet_path=config["data"]["parquet_path"])
    token_analysis = bool(config["evaluation"].get("token_analysis", False))
    # Fail fast: the token analysis groups tokens by clinical Type, so a null type_col
    # would only blow up after the whole evaluation had already run.
    if token_analysis and cols["type_col"] is None:
        raise ValueError(
            "evaluation.token_analysis is true but columns.type_col is null; the "
            "analysis groups tokens by clinical Type. Name the column or turn it off.")

    # Setup device
    device_str = resolve_device(config["hardware"]["device"])
    print(f"Using device: {device_str}")

    # Create output directory
    ecfg = config["evaluation"]
    output_dir = Path(ecfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model, checkpoint = load_checkpoint(
        checkpoint_path=config["checkpoint"]["model_path"],
        vocab_path=config["checkpoint"]["vocab_path"],
        device=device_str
    )
    print(f"Model loaded from epoch {checkpoint.get('epoch', '?')}")
    if checkpoint.get("val_loss") is not None:
        print(f"Training val loss: {checkpoint['val_loss']:.4f}")

    # Test windows: streamed on the fly from the MEDS input (as in training's streaming
    # mode), or read from a prebuilt window store (data.test_npz)
    print("\nLoading test data...")
    event_times = None   # the run's checked rows: loaded for streaming, else for the token analysis
    if config["data"].get("streaming", False):
        test_dataset, event_times = _streaming_test_dataset(config, cols, checkpoint, model)
    else:
        test_dataset = _store_test_dataset(config, model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=ecfg["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        collate_fn=collate_fn,
    )
    print(f"Test set: {len(test_dataset)} sequences, {len(test_loader)} batches")

    # The loss is reported with the T2E weight the model was trained with (part of its config)
    if model.t2e_pred:
        print(f"Reporting loss with trained t2e_weight={model.t2e_weight}")
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_loader, device=device_str, show_progress=True)

    # Arm provenance: which model variant produced these numbers (for cross-version tables)
    mc = checkpoint.get("model_config", {})
    tc = checkpoint.get("config", {})
    metrics["model"] = {
        "loss_mode": mc.get("loss_mode"), "t2e_weight": mc.get("t2e_weight"),
        "d_model": mc.get("d_model"), "n_layers": mc.get("n_layers"),
        "shuffle_simultaneous": tc.get("data", {}).get("shuffle_simultaneous", False),
    }

    # Print results
    print_metrics(metrics)

    # Save metrics
    save_metrics(metrics, output_dir / "test_metrics.json")

    # Create plots
    plot_evaluation_metrics(metrics, str(output_dir / "test_metrics.png"))

    # ── Vocabulary token-embedding analysis (optional) ────────────────────────
    # What the model learned about each vocabulary token, read straight off the token
    # embedding table and the MEDS input (encoded on the fly via DuckDB - no intermediate
    # file). Lands under the SAME evaluation.output_dir.
    if token_analysis:
        from tracehr.generation.token_analysis import analyze_token_embeddings
        if event_times is None:
            event_times = load_event_times(config["data"]["parquet_path"], cols,
                                           config["checkpoint"]["vocab_path"])
        analyze_token_embeddings(
            model, config["checkpoint"]["vocab_path"], config["data"]["parquet_path"],
            output_dir / "token_embeddings",
            top_k=ecfg.get("token_top_k", 10),
            tok_col=cols["tok_col"], type_col=cols["type_col"], event_times=event_times)

    print(f"\nAll results saved to {output_dir}")


def cli():
    """Console entry point (tracehr-evaluate): run main on --config."""
    parser = argparse.ArgumentParser(description="Evaluate trained NTPTransformer on test set")
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation configuration file")
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
