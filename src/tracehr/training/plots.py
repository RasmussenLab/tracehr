# src/tracehr/training/plots.py
"""Training-curve figures of pretraining (pipeline 01). plot_epoch_metric and
plot_all_batch_losses are shared with finetuning (pipeline 06), whose own figures are in
finetuning/finetune_plots.py.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_losses(train_losses, val_losses, out_png: str) -> None:
    """Training and validation loss per epoch (pipeline 01)."""
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Across Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_epoch_metric(values, out_png: str, *, label: str, ylabel: str, title: str, color: str,
                      ylim=None, saved_name=None) -> None:
    """One validation metric per epoch. ylim fixes the y range; saved_name, when given, prints
    "<saved_name> plot saved to <out_png>"."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(values)+1), values, label=label, color=color, marker='s', markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    if saved_name:
        print(f"{saved_name} plot saved to {out_png}")


def plot_perplexity(val_perplexities, out_png: str) -> None:
    """Validation perplexity per epoch (pipeline 01)."""
    plot_epoch_metric(val_perplexities, out_png, label="Validation", ylabel="Perplexity",
                      title="Validation Perplexity Across Epochs", color="coral",
                      saved_name="Perplexity")


def plot_accuracy(val_accuracies, out_png: str) -> None:
    """Validation token accuracy per epoch (pipeline 01)."""
    plot_epoch_metric(val_accuracies, out_png, label="Validation", ylabel="Accuracy",
                      title="Validation Accuracy Across Epochs", color="seagreen", ylim=[0, 1],
                      saved_name="Accuracy")


def plot_all_batch_losses(all_batch_losses: list[list[float]], out_png: str, *,
                          title: str = "Training Loss per Batch (All Epochs)",
                          epoch_labels: str = "text", announce: bool = True) -> None:
    """Every batch loss of the run (one list per epoch), with a moving average and the epoch
    boundaries marked. epoch_labels "text" writes E2, E3, ... at each boundary (pipeline
    01); "legend" names the boundary lines once in the legend (pipeline 06). announce
    prints where the figure was saved."""
    plt.figure(figsize=(14, 6))

    # Flatten all batch losses with epoch boundaries
    flat_losses = []
    epoch_boundaries = [0]
    for epoch_losses in all_batch_losses:
        flat_losses.extend(epoch_losses)
        epoch_boundaries.append(len(flat_losses))

    batches = range(1, len(flat_losses) + 1)
    plt.plot(batches, flat_losses, 'b-', alpha=0.4, linewidth=0.8, label='Batch Loss')

    # Add smoothed line
    window_size = min(50, len(flat_losses) // 20)
    if window_size > 1:
        smoothed = np.convolve(flat_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size, len(flat_losses) + 1), smoothed,
                 'r-', linewidth=2, label=f'Smoothed (window={window_size})')

    # Mark the epoch boundaries
    for i, boundary in enumerate(epoch_boundaries[1:-1], 1):
        if epoch_labels == "text":
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            plt.text(boundary, plt.ylim()[1] * 0.95, f'E{i+1}',
                     ha='center', va='top', fontsize=9, color='gray')
        else:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5,
                        label='Epoch boundary' if i == 1 else None)

    plt.xlabel('Global Batch Number', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    if announce:
        print(f"All-epoch batch loss plot saved to {out_png}")
