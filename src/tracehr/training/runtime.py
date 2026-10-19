# src/tracehr/training/runtime.py
"""Helpers every training and evaluation loop shares: seeding, parameter counts, devices,
batches, checkpoint pruning and the progress log."""

import functools
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

LOG_EVERY_N_BATCHES = 50  # progress line frequency of every training/validation/test loop


def set_seed(seed: int | None) -> None:
    """Seed Python's, numpy's and torch's (CUDA included) random generators; None does nothing."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    """The number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_device(device):
    """The device setting with "auto" resolved: "cuda" when torch can use a GPU, else "cpu". Any
    other value (a device name or a torch.device) is returned unchanged. A CUDA device is
    checked first (check_cuda_device), and "auto" falling back to the CPU on a machine that
    has a GPU prints why."""
    if device == "auto":
        if not torch.cuda.is_available():
            if _gpu_listed():
                print(f"WARNING: device 'auto' is using the CPU although nvidia-smi lists a GPU: "
                      f"{_torch_build()} cannot use it. A CPU-only build, or a CUDA build newer "
                      f"than the driver supports (nvidia-smi's 'CUDA Version'), is the usual "
                      f"cause; the README's install section shows which torch to install.")
            return "cpu"
        device = "cuda"
    if torch.device(device).type == "cuda":
        check_cuda_device(str(device))
    return device


@functools.cache
def check_cuda_device(device: str) -> None:
    """Fail at startup, with the fix, when this torch build cannot run on the GPU: CUDA not
    usable at all, or a GPU the build has no kernels for (a V100 with a CUDA 13 build), which
    would otherwise stop the run at its first GPU operation with "no kernel image is available
    for execution on the device". Runs one tiny operation, once per process and device."""
    if not torch.cuda.is_available():
        raise RuntimeError(f"hardware.device is '{device}', but {_torch_build()} cannot use a GPU "
                           f"here. Use device 'auto' or 'cpu', or install the torch build for "
                           f"this GPU and driver (README, install section).")
    try:
        torch.ones(1, device=device).add_(1).item()
    except RuntimeError as exc:
        try:
            major, minor = torch.cuda.get_device_capability(device)
            gpu = f"{torch.cuda.get_device_name(device)} (compute capability {major}.{minor})"
        except (RuntimeError, AssertionError, ValueError):     # e.g. an invalid device index
            gpu = f"the GPU '{device}'"
        raise RuntimeError(
            f"{_torch_build()} cannot run on {gpu}; the build has kernels for "
            f"{', '.join(torch.cuda.get_arch_list())}. Install the torch build for this GPU "
            f"(README, install section): GPUs older than compute capability 7.5, such as the "
            f"V100, need the CUDA 12.6 build, torch < 2.15. torch's error: {exc}") from exc


def _torch_build() -> str:
    """This torch's version and CUDA build, for messages."""
    build = f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU-only build"
    return f"torch {torch.__version__} ({build})"


def _gpu_listed() -> bool:
    """Whether nvidia-smi lists a GPU this process can see (under SLURM, only the job's GPUs)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        listing = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "GPU " in listing


def batch_to_device(batch: dict, device, non_blocking: bool = False) -> dict:
    """A batch dict with every tensor moved to device; other values (ids, names) unchanged."""
    return {k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def prune_epoch_checkpoints(directory, keep_last_n: int) -> None:
    """Delete all but the keep_last_n highest-numbered epoch_N.pt in directory (0 keeps all)."""
    if keep_last_n > 0:
        epochs = sorted(Path(directory).glob("epoch_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        for old in epochs[:-keep_last_n]:
            old.unlink()


class ProgressLog:
    """Drop-in for the tqdm bars of the batch loops: one plain line every
    LOG_EVERY_N_BATCHES batches instead of a redrawn bar, so log files stay readable.
    Iterate over it like the loader; set_postfix(dict) sets the values the next line shows."""

    def __init__(self, iterable, desc: str, disable: bool = False):
        """desc starts every line; disable turns the lines off."""
        self.iterable, self.desc, self.disable = iterable, desc, disable
        self.postfix = {}
        try:
            self.total = len(iterable)
        except TypeError:  # an IterableDataset loader without a length
            self.total = None

    def set_postfix(self, postfix: dict) -> None:
        """The values the next progress line shows."""
        self.postfix = postfix

    def __iter__(self):
        """Yield the loader's batches, printing a line every LOG_EVERY_N_BATCHES of them."""
        start = time.perf_counter()
        for step, item in enumerate(self.iterable, start=1):
            yield item  # the caller's loop body (and its set_postfix) runs here
            if not self.disable and step % LOG_EVERY_N_BATCHES == 0:
                elapsed = time.perf_counter() - start
                parts = [self.desc, f"step {step}/{self.total or '?'}",
                         *(f"{k} {v}" for k, v in self.postfix.items()),
                         f"{step / elapsed:.2f} batch/s", f"elapsed {elapsed / 60:.1f} min"]
                print("  ".join(parts), flush=True)
