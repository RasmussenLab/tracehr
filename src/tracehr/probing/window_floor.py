# src/tracehr/probing/window_floor.py
"""The window floor of the embedding pipelines (04, 05): the training min_seq_len read back
from the checkpoint, the refusal of a min_seq_len config key, and the head-window report."""



def training_min_seq_len(checkpoint: dict, default: int = 1) -> int:
    """The window floor the model was TRAINED with, read off the checkpoint's config.

    Every inference path that re-windows a history has to use the training floor, or it
    asks the model to embed windows narrower than any it ever saw. A right-aligned plan
    leaves a head window of width (n_tokens mod stride), so at a floor of 1 a patient can
    contribute a 1-token window that then enters the combined patient embedding at full
    weight. Pipeline 02 already reads the value back this way (see evaluate_model.py); this
    is the same lookup, shared so pipelines 04 and 05 cannot drift from it again.

    train_model.py stores the whole training config under "config", so nothing new has to
    be persisted. A checkpoint predating that key falls back to `default`.
    """
    train_cfg = checkpoint.get("config") or {}
    seq_cfg = train_cfg.get("sequences") or {}
    return int(seq_cfg.get("min_seq_len", default))


def reject_min_seq_len_key(block: dict, block_name: str, current: int | None = None) -> None:
    """Refuse a min_seq_len key in an embedding pipeline's config block (04 extraction, 05
    outcome): the window floor is the training value (training_min_seq_len), so a key here
    could only ever disagree with it. current, when known, is quoted in the message."""
    if "min_seq_len" in block:
        now = f" (currently {current})" if current is not None else ""
        raise ValueError(
            f"{block_name}.min_seq_len is not a config knob (found {block['min_seq_len']!r}). "
            f"The window floor must match training, so it is read from the checkpoint's "
            f"sequences.min_seq_len{now}. Remove the key; to change the floor, retrain with a "
            f"different sequences.min_seq_len.")


def window_floor_from_checkpoint(checkpoint: dict) -> int:
    """The window floor of an embedding pipeline (04, 05): the training min_seq_len read off
    the checkpoint, or 1 (no floor) when it records none. The choice is printed."""
    min_seq_len = training_min_seq_len(checkpoint, default=0)
    if min_seq_len:
        print(f"Window floor: min_seq_len={min_seq_len}, from the checkpoint's training config")
        return min_seq_len
    print("Window floor: min_seq_len=1 - the checkpoint records no sequences.min_seq_len, "
          "so no floor is applied (windows as narrow as 1 token may be embedded).")
    return 1


def report_head_window(start_from_end, add_head_window: bool) -> None:
    """Print what happens to a right-aligned head span narrower than the floor (04, 05): off,
    those oldest tokens land in no window; on, an extra full-width window covers them. It
    changes the window count per patient, so the run says which one ran."""
    if start_from_end:
        print(f"Head window: add_head_window={add_head_window} - a head span narrower than "
              f"min_seq_len is "
              + ("covered by an extra full-width [0:K) window that re-reads tokens the next "
                 "window already holds." if add_head_window else
                 "DROPPED; those oldest tokens reach bag-of-words but no embedding."))
