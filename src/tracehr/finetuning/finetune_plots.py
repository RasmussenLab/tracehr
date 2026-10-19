# src/tracehr/finetuning/finetune_plots.py
"""Training-curve figures of a finetuning run (pipeline 06): the loss per epoch and every
batch loss. The figures it shares with pretraining are in training/plots.py."""

import matplotlib.pyplot as plt
import numpy as np


def plot_finetune_losses(train_losses, val_losses, out_png: str) -> None:
    """Training and validation loss per epoch of a finetuning run (pipeline 06)."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train", marker='o', markersize=4)
    plt.plot(epochs[:len(val_losses)], val_losses, label="Val", marker='s', markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Fine-tuning Loss Across Epochs", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_finetune_batch_losses(batch_losses: list, epoch: int, out_png: str) -> None:
    """Every batch loss of one finetuning epoch, with a moving average (pipeline 06)."""
    plt.figure(figsize=(10, 6))
    batches = range(1, len(batch_losses) + 1)
    plt.plot(batches, batch_losses, 'b-', alpha=0.7, linewidth=1, label='Batch Loss')

    window_size = min(20, len(batch_losses) // 10)
    if window_size > 1:
        smoothed = np.convolve(batch_losses, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size, len(batch_losses) + 1), smoothed,
                 'r-', linewidth=2, label=f'Smoothed (window={window_size})')

    plt.xlabel('Batch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Fine-tuning Loss per Batch - Epoch {epoch}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
