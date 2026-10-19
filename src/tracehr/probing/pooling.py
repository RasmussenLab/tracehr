# src/tracehr/probing/pooling.py
"""Pooling a window's final hidden states into one vector. Shared: embedding extraction
(pipelines 04, 05) and the finetuning head (pipeline 06)."""

import torch


def pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor, method: str,
                       attention_scores: torch.Tensor | None = None) -> torch.Tensor:
    """[B, T, D] hidden states -> one vector per window, from its real (unpadded) positions.

    mean / max over the real positions; last: the state at the last real position; combined:
    [last, mean, max] concatenated ([B, 3D]); attention: softmax weights over the real
    positions, scored as hidden_states @ attention_scores (the finetuning head's learned
    vector). A padded state is exactly 0 in this model, so even a window with no real
    position pools to a defined value. Shared by embedding extraction (pipelines 04, 05) and
    the finetuning head (pipeline 06).
    """
    if method == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    if method == "max":
        return hidden_states.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf")).max(dim=1)[0]
    if method == "last":
        last = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)
        return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last]
    if method == "combined":
        return torch.cat([pool_hidden_states(hidden_states, attention_mask, m)
                          for m in ("last", "mean", "max")], dim=-1)
    if method == "attention" and attention_scores is not None:
        # A finite fill (not -inf) keeps a window with no real position NaN-free.
        scores = hidden_states @ attention_scores                          # [B, T]
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1).unsqueeze(-1)                       # [B, T, 1]
        return (hidden_states * weights).sum(dim=1)
    raise ValueError(f"Unknown pooling method: {method}")
