# src/tracehr/generation/model_io.py
"""Loading a trained NTPTransformer checkpoint and checking it against the vocabulary, and
writing metrics files. Shared: every pipeline after training (02-06) uses it."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..dataset.vocabulary import load_vocab, vocabulary_fingerprint
from ..ntp_model.config import NTPTransformerConfig
from ..ntp_model.ntp_transformer import NTPTransformer
from ..training.runtime import resolve_device


def load_checkpoint(checkpoint_path: str, vocab_path: str, device: str = "auto") -> tuple[NTPTransformer, dict]:
    """(model, checkpoint) of an NTPTransformer checkpoint, rebuilt from its recorded
    model_config, in eval mode on device ("auto": CUDA when available). Refuses a
    checkpoint without a model_config, one whose weights do not fit it, and one trained on
    another vocabulary than vocab_path."""
    device = torch.device(resolve_device(device))

    # weights_only=False: our own checkpoints embed the training config, which
    # contains numpy scalars that the weights-only unpickler rejects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Reconstruct from the saved architecture spec (single source of truth)
    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint has no 'model_config' (pre-NTPTransformerConfig format). Retrain.")
    config = NTPTransformerConfig.from_dict(checkpoint["model_config"])
    model = NTPTransformer.from_config(config).to(device)

    # The objective needs nothing restored beyond the architecture: its settings are module
    # constants in its own file, so loss_mode (a config field) rebuilds it exactly, and any
    # parameters it owns arrive with the state_dict below.
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(f"state_dict does not match model_config architecture — retrain. ({e})") from e
    model.eval()

    # Cross-check the checkpoint against the vocabulary it will be used with
    stoi, itos, stats = load_vocab(vocab_path)
    if config.vocab_size != len(itos):
        raise ValueError(f"Checkpoint vocab_size={config.vocab_size} != vocab file ({len(itos)}).")
    # train_model.py records the fingerprint of the ids the embeddings were learned on (tokens,
    # sex tokens, and types when embedded, as type_col is not None there exactly when
    # use_type is true).
    check_vocab_fingerprint(checkpoint, vocabulary_fingerprint(stoi, stats, use_type=bool(config.use_type)),
                            checkpoint_path, vocab_path)

    # Runtime provenance only; no new model_config fields or state_dict entries.
    model.checkpoint_path = str(Path(checkpoint_path).resolve())
    return model, checkpoint


def check_vocab_fingerprint(checkpoint: dict, expected: str, checkpoint_path, vocab_path) -> None:
    """Refuse a checkpoint trained on another vocabulary than vocab_path, whose
    dataset.vocabulary.vocabulary_fingerprint is `expected`; warn when it records none.

    Same size is not same vocabulary: another one of equal size would encode the inputs and
    decode the logits as other tokens without any error.
    """
    saved = checkpoint.get("vocab_fingerprint")
    if saved is None:
        print(f"[checkpoint] WARNING: {checkpoint_path} has no vocab_fingerprint (saved before "
              f"checkpoints recorded one), so only the vocabulary size can be checked, not that "
              f"{vocab_path} is the vocabulary the model was trained on.")
    elif saved != expected:
        trained_with = ((checkpoint.get("config") or {}).get("data") or {}).get("vocab_path")
        raise ValueError(
            f"{vocab_path} is not the vocabulary {checkpoint_path} was trained on"
            + (f" ({trained_with})" if trained_with else "") + ": its token or type ids, or "
            "its front block, birth token or time origin, differ, so the model would misread "
            "every window. Use the training vocabulary.")


def json_safe(obj):
    """Recursively replace non-finite floats with None so the result is valid JSON.

    A metric that could not be computed - AUROC on a split with one class, a C-index with
    no comparable pairs - is genuinely undefined, and NaN is the honest value to carry
    through numpy: it propagates, so an aggregate built from a partly-undefined set comes
    out visibly undefined instead of quietly averaging the parts that worked.

    On disk it has to become null. json.dump writes bare NaN, which is not JSON and which
    strict parsers (and every other language) refuse. null says the same thing and reloads
    as None. Substituting 0.5 or 0.0 instead would be worse than either: it reports a
    measurement that was never made.

    Inert on finite input - the output is byte-identical to json.dump's.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):          # numpy scalars json cannot serialise
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def save_metrics(metrics: dict[str, float], output_path: str) -> None:
    """Write metrics as JSON (non-finite values as null, see json_safe), creating the folder."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(json_safe(metrics), f, indent=2)

    print(f"Metrics saved to {output_path}")
