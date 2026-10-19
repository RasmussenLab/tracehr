# src/tracehr/generation/attribution.py
"""The two signed attribution methods: integrated_gradients (path from an all-<PAD>
context) and occlusion (one event replaced by <PAD>). For a generated token both share one
unit, the change in log p(token). Also the sign colours of the attribution figures.
Shared: pipeline 03 attributes generated tokens with both methods, and pipeline 06
attributes its task head's output with integrated_gradients.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Sign colours of the plots, shared with the finetuning explainability (raises / lowers p)
POSITIVE_COLOR, NEGATIVE_COLOR = "#d62728", "#1f77b4"


def _target_log_prob(model, rows: dict[str, torch.Tensor], embeds: torch.Tensor,
                     target_id: int) -> torch.Tensor:
    """log p(target) at the last position of every row ([N,L] windows, [N,L,D] embeds) -> [N].
    model.head is the scorer, whatever the objective, so attribution runs on the function
    the loss actually trained."""
    logits = model(rows, input_embeds=embeds, last_only=True)["logits"][:, -1]
    return F.log_softmax(logits.float(), dim=-1)[:, target_id]


def _repeat(sample: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    """One [1,L] window as n identical rows (views, no copies)."""
    return {k: v.expand(n, -1) for k, v in sample.items()}


def integrated_gradients(model, sample: dict[str, torch.Tensor], target_id: int,
                         n_steps: int, batch_size: int, score=_target_log_prob) -> np.ndarray:
    """Signed Integrated Gradients of log p(target) for each event of one window -> [L].

    The path runs from an all-<PAD> token baseline to the real token embeddings (time and
    type inputs unchanged) by the midpoint rule, so the scores sum to log p(target | context)
    - log p(target | all-<PAD> context) up to the discretisation error. The n_steps path
    points run batch_size at a time; gradients are taken w.r.t. the path embeddings only,
    so the model's parameter gradients are never touched.

    score(model, rows, embeds, target_id) -> [N] is the attributed output, log p(target) by
    default; the finetuning explainability passes its task head's output instead.
    """
    ids = sample["input_ids"]
    with torch.no_grad():
        embeds = model.tok(ids)                                             # [1,L,D]
        baseline = model.tok(torch.full_like(ids, model.pad_id))
    delta = embeds - baseline
    alphas = (torch.arange(n_steps, device=ids.device, dtype=embeds.dtype) + 0.5) / n_steps
    grad_sum = torch.zeros_like(embeds[0])                                  # [L,D]
    with torch.enable_grad():
        for chunk in alphas.split(batch_size):
            path = (baseline + chunk.view(-1, 1, 1) * delta).requires_grad_(True)   # [n,L,D]
            total = score(model, _repeat(sample, len(chunk)), path, target_id).sum()
            grad_sum += torch.autograd.grad(total, path)[0].sum(dim=0)
    return (grad_sum / n_steps * delta[0]).sum(dim=-1).cpu().numpy()


@torch.no_grad()
def occlusion(model, sample: dict[str, torch.Tensor], target_id: int, batch_size: int) -> np.ndarray:
    """Signed occlusion of log p(target) for each event of one window -> [L].

    Score of event i = log p(target | context) - log p(target | context with event i's
    token replaced by <PAD>, its time and type inputs unchanged). Row 0 of the batched
    input is the untouched window and row i+1 blanks event i; the L+1 rows run batch_size
    at a time.
    """
    ids = sample["input_ids"]
    L = ids.size(1)
    variants = ids.repeat(L + 1, 1)                                         # [L+1,L]
    events = torch.arange(L, device=ids.device)
    variants[events + 1, events] = model.pad_id
    log_p = []
    for part in variants.split(batch_size):
        rows = {**_repeat(sample, len(part)), "input_ids": part}
        log_p.append(_target_log_prob(model, rows, model.tok(part), target_id))
    log_p = torch.cat(log_p)
    return (log_p[0] - log_p[1:]).cpu().numpy()
