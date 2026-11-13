# src/tracehr/finetuning/finetune_plots.py
"""Figures of a finetuning run (pipeline 06): the training curves, the style every pipeline-06
figure is drawn in (each saved as a 400-dpi PNG and an SVG), and the 5-patient rule every
reported figure and table follows. The figures it shares with pretraining are in
training/plots.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# The 5-patient rule: every number pipeline 06 reports (its figures and their tables,
# test_metrics.json, explainability.json) pools at least MIN_PATIENTS distinct patients, and
# no smaller count can be worked out from the numbers shown. A fixed value, not a setting.
MIN_PATIENTS = 5
HIDDEN = f"<{MIN_PATIENTS}"   # what a reported table or json writes in place of a hidden number

FIG_WIDTH = 7.0       # inches at print size: two journal columns (3.5 for one)
PNG_DPI = 400
# Print-size fonts, and SVG text kept as editable text. Applied with plt.rc_context, so the
# global rcParams never change.
STYLE = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
         "axes.spines.right": False, "svg.fonttype": "none"}


def save_figure(fig, stem) -> None:
    """stem.png (PNG_DPI) and stem.svg, then close fig."""
    stem = Path(stem)
    with plt.rc_context(STYLE):
        for suffix, kwargs in ((".png", {"dpi": PNG_DPI}), (".svg", {})):
            fig.savefig(stem.parent / f"{stem.name}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def hidden_groups(patients: list[set], min_patients: int = MIN_PATIENTS) -> list[bool]:
    """Which of the groups whose numbers add up to a shown total must be hidden; patients holds
    each group's distinct patients. Every group of 1 to min_patients - 1 patients is hidden,
    then the smallest non-empty others too, until at least two are hidden and they hold
    min_patients patients together: a lone hidden number could be worked out by subtracting the
    others from the total. An empty group reveals no patient and stays shown (as 0), unless no
    non-empty group is left to hide with; then every group is hidden. None is hidden when no
    group is small."""
    hidden = [0 < len(p) < min_patients for p in patients]

    def enough() -> bool:
        """True when the hidden groups protect each other (or none is hidden)."""
        together = set().union(*(p for p, h in zip(patients, hidden) if h))
        return not any(hidden) or (sum(hidden) >= 2 and len(together) >= min_patients)

    for i in sorted(range(len(patients)), key=lambda i: len(patients[i])):
        if enough():
            break
        if patients[i]:
            hidden[i] = True
    return hidden if enough() else [True] * len(patients)


def plot_training_curves(history: dict, monitor_key: str, unfreeze_after_epochs: int | None,
                         out_stem, evaluated_epoch: int | None = None) -> None:
    """The training curves of a finetuning run, saved as out_stem.png/.svg: (a) the train and val
    loss per epoch, (b) the val monitored metric (monitor_key). A dashed line marks where
    two_phase unfreezes the backbone; evaluated_epoch, when given, marks the epoch the test
    results come from."""
    with plt.rc_context(STYLE):
        fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.6), constrained_layout=True)
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax_loss.plot(epochs, history["train_loss"], marker="o", markersize=3, label="Train")
        ax_loss.plot(epochs[:len(history["val_loss"])], history["val_loss"], marker="s",
                     markersize=3, label="Validation")
        ax_score.plot(epochs[:len(history["val_score"])], history["val_score"], marker="s",
                      markersize=3, color="C1", label="Validation")
        for ax in (ax_loss, ax_score):
            # two_phase unfreezes at the start of 1-based epoch unfreeze_after_epochs + 1
            if unfreeze_after_epochs is not None and unfreeze_after_epochs < len(epochs):
                ax.axvline(unfreeze_after_epochs + 0.5, color="0.5", linestyle="--", linewidth=0.8,
                           label="Backbone unfrozen")
            if evaluated_epoch is not None:
                ax.axvline(evaluated_epoch, color="0.2", linestyle=":", linewidth=1.0,
                           label=f"Evaluated: best val {monitor_key}")
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False)
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("a  Loss", loc="left", fontweight="bold")
        ax_score.set_ylabel(f"Val {monitor_key}")
        ax_score.set_title(f"b  Validation {monitor_key}", loc="left", fontweight="bold")
    save_figure(fig, out_stem)
