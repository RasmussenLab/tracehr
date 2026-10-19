# src/tracehr/training/resume.py
"""What may change when pipeline 01 resumes from a checkpoint, and how the run adapts.

A resume continues the SAME run: every setting that shapes training must match the yaml the
checkpoint was trained with, and so must the vocabulary (by content, not path - its ids are
what the embeddings learned). The dataset may change. A changed epoch length (new training
data) starts a new LR phase, because the old schedule length in optimizer steps would no
longer fit. Raising num_epochs extends the run with a new LR phase. The best model is
re-scored on the current validation data at every resume (in the pipeline), so a changed
validation set is compared fairly and best_model.pt stays the best of the whole run.
"""

from pathlib import Path

from ..ntp_model.config import NTPTransformerConfig

# Yaml keys that may differ from the checkpoint's (a key also covers everything under it).
# The list names what may change, not what is checked, so a setting added later is checked
# by default.
RESUME_MAY_CHANGE = (
    "hardware",                      # where the run executes, not what it learns
    "checkpoint",                    # how often it saves
    "data.output_dir",
    "data.model_checkpoint_path",    # differs on every resume by construction
    "training.num_epochs",           # raising it extends the run (resume_lr_schedule)
    # The dataset: a changed epoch length starts a new LR phase, and the best model is
    # re-scored on the current validation data. The vocabulary file may move, but its
    # content is checked (vocabulary_fingerprint).
    "data.parquet_path", "data.train_ids", "data.val_ids", "data.vocab_path",
    "data.train_npz", "data.val_npz",
    "columns",                       # the data file's column names
    "sequences.unk_token",           # the data file's rare-token marker
)
# Allowed changes reported elsewhere (or expected), so not listed again
_NOT_REPORTED = ("data.model_checkpoint_path", "training.num_epochs")

# Checkpoint fields a resume reads; a checkpoint without them predates these checks
RESUME_FIELDS = ("vocab_fingerprint", "steps_per_epoch", "lr_schedule_start_epoch",
                 "data_changes")

_ABSENT = object()


def _covered(key: str, prefixes) -> bool:
    """Whether a dotted yaml key is one of prefixes or lies under one of them."""
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def resolve_resume_checkpoint(setting, output_dir, key: str) -> Path | None:
    """The checkpoint a run resumes from, or None to train from scratch (pipelines 01 and 06).

    `setting` is the yaml `key`: any checkpoint file, in any folder. It comes first when its
    file exists; otherwise <output_dir>/latest.pt is resumed when that exists. Every case that
    is not simply "resume the file the setting names" is reported. A folder is refused: the
    setting names the checkpoint file itself.
    """
    if isinstance(setting, str) and setting.strip().lower() in {"", "none", "null"}:
        setting = None
    requested = Path(setting) if setting is not None else None
    if requested is not None and requested.is_dir():
        raise ValueError(f"{key} {str(requested)!r} is a folder. Point it at the checkpoint "
                         f"file to resume from (e.g. <output_dir>/latest.pt), or set it to null.")
    latest = Path(output_dir) / "latest.pt"

    if requested is not None and requested.is_file():
        if latest.is_file() and latest.resolve() != requested.resolve():
            print(f"[resume] WARNING: resuming from {key} {requested}; {latest} is another "
                  f"checkpoint and will be overwritten.")
        return requested
    if requested is not None:
        print(f"[resume] WARNING: {key} {str(requested)!r} does not exist.")
    if latest.is_file():
        print(f"[resume] Found {latest}: resuming from it.")
        return latest
    print(f"[resume] No checkpoint to resume from (no {latest}): training from scratch.")
    return None


def config_differences(saved: dict, current: dict,
                       may_change=RESUME_MAY_CHANGE) -> tuple[list[str], list[str]]:
    """(forbidden, allowed) yaml differences as "dotted.key: old -> new" lines.

    The yaml is compared as written: adding a key at its default value counts as a change.
    `may_change` lists the keys allowed to differ (default: pipeline 01's; pipeline 06
    passes its own).
    """
    forbidden, allowed = [], []

    def walk(old, new, prefix):
        """Compare two yaml (sub)dicts key by key, descending into nested dicts."""
        for key in sorted(set(old) | set(new), key=str):
            path = f"{prefix}{key}"
            a, b = old.get(key, _ABSENT), new.get(key, _ABSENT)
            if isinstance(a, dict) and isinstance(b, dict):
                walk(a, b, path + ".")
            elif a != b:
                line = (f"{path}: {'<absent>' if a is _ABSENT else repr(a)} -> "
                        f"{'<absent>' if b is _ABSENT else repr(b)}")
                if not _covered(path, may_change):
                    forbidden.append(line)
                elif not _covered(path, _NOT_REPORTED):
                    allowed.append(line)

    walk(saved, current, "")
    return forbidden, allowed


def check_training_resume(ckpt: dict, resume_path, *, model_config: NTPTransformerConfig,
                          selection_metric: str, config: dict, vocab_fp: str,
                          vocab_path: str) -> None:
    """Refuse a checkpoint pipeline 01 cannot continue, then report the setting changes that
    do not affect how training continues.

    Resuming continues the SAME model, so every architecture setting must match the model
    this run built (model_config). load_state_dict cannot see all of them: the Age-RoPE
    ladder end is not a weight, and the Time2Vec periods are a buffer it would silently
    overwrite. dropout too: it shapes how training continues.
    """
    if ckpt.get("model_config") is None:
        raise ValueError(f"{resume_path} has no 'model_config' to check this run against; "
                         f"retrain from scratch (model_checkpoint_path: null).")
    saved = NTPTransformerConfig.from_dict(ckpt["model_config"]).to_dict()
    differs = {k: (saved[k], v) for k, v in model_config.to_dict().items() if saved[k] != v}
    if differs:
        raise ValueError(
            f"Cannot resume from {resume_path}: it was trained with a different model "
            f"configuration (" + ", ".join(f"{k}: {old!r} -> {new!r}"
                                           for k, (old, new) in differs.items()) + "). "
            "Retrain from scratch (model_checkpoint_path: null) or resume with the "
            "original config.")
    # The best so far is only comparable under the same criterion.
    if ckpt.get('selection_metric') != selection_metric:
        raise ValueError(
            f"{resume_path} selected best_model.pt on {ckpt.get('selection_metric')!r} "
            f"(None = saved before selection_metric existed), but this run selects on "
            f"{selection_metric!r}; retrain from scratch.")
    missing = [k for k in RESUME_FIELDS if k not in ckpt]
    if missing:
        raise ValueError(f"{resume_path} was saved before resumes were checked (no {missing}); "
                         f"retrain from scratch (model_checkpoint_path: null).")
    # Every other setting that shapes training must match the yaml the checkpoint was
    # trained with, and the vocabulary its embeddings were learned on. All are reported.
    forbidden, allowed = config_differences(ckpt["config"], config)
    if ckpt["vocab_fingerprint"] != vocab_fp:
        forbidden.append(f"vocabulary ({vocab_path}): its token or type ids, or its "
                         f"front block, birth token or time origin, differ from the "
                         f"checkpoint's, so the embeddings would be read as other tokens")
    if forbidden:
        raise ValueError(
            f"Cannot resume from {resume_path}: these differ from the run that produced it, "
            f"and each changes how training continues:\n  " + "\n  ".join(forbidden)
            + "\nResume with the settings and vocabulary the run was trained with, or retrain "
              "from scratch (model_checkpoint_path: null). Raising training.num_epochs, the "
              "dataset, and the hardware, output and checkpoint settings may change.")
    if allowed:
        print("[resume] changed, not affecting how training continues: " + "; ".join(allowed))


def resume_lr_schedule(*, lr_kind: str, start_epoch: int, num_epochs: int,
                       planned_epochs: int, saved_start: int,
                       saved_steps: int, steps_per_epoch: int) -> tuple[int, str | None]:
    """(epoch the LR schedule starts at, reason for a new phase or None) for a resume.

    A new phase warms up again and decays over the remaining epochs, Adam state kept: the
    best of the strategies measured for extending a finished run (it matched a run trained
    to the new length from scratch). Otherwise the checkpoint's schedule continues exactly,
    which needs the same length in optimizer steps - so a changed epoch length (new training
    data, or a streaming epoch estimated from another hardware.num_workers) starts one too.
    """
    if lr_kind not in ("linear", "cosine"):
        return saved_start, None                     # a constant LR has no length to keep
    reasons = []
    if num_epochs > planned_epochs:
        reasons.append(f"num_epochs {planned_epochs} -> {num_epochs}")
    if steps_per_epoch != saved_steps:
        reasons.append(f"an epoch is now {steps_per_epoch} optimizer steps, not {saved_steps}")
    if reasons:
        return start_epoch, " and ".join(reasons)
    if num_epochs < planned_epochs:
        raise ValueError(
            f"num_epochs was lowered ({planned_epochs} -> {num_epochs}) part-way through a "
            f"{lr_kind} schedule planned to decay at epoch {planned_epochs}: the run would stop "
            f"before the learning rate has decayed. Keep num_epochs: {planned_epochs}, or raise "
            f"it to extend the run.")
    return saved_start, None


def describe_lr_schedule(lr_kind: str, learning_rate: float, warmup_steps: int,
                         total_steps: int, first_epoch: int, last_epoch: int) -> str:
    """One line describing the learning-rate schedule of this run (or phase)."""
    if lr_kind not in ("linear", "cosine"):
        return f"constant {learning_rate:g}"
    return (f"{lr_kind}: warmup {warmup_steps} steps to {learning_rate:g}, then down to "
            f"{0.1 * learning_rate:g} at epoch {last_epoch} ({total_steps:,} optimizer steps "
            f"over epochs {first_epoch + 1}-{last_epoch})")
