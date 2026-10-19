# src/tracehr/finetuning/outcome_model.py
"""Task-generic outcome finetuning: wraps a pretrained NTPTransformer backbone,
pools its final hidden states, and replaces the LM head with a task head:
  regression -> Linear(D, 1), Huber loss, prediction = value
  binary     -> Linear(D, 1), BCEWithLogits (+pos_weight), prediction = probability
  multiclass -> Linear(D, K), CrossEntropy (+class weights), prediction = class probs
  survival   -> Linear(D, 1), Cox partial likelihood, prediction = log relative risk
Explainability attributes the head's scalar (value / logit / predicted-class logit /
risk).

The survival head takes a TWO-column label, (duration, event) - the contract
survival_core owns and the stage-05 comparison arms read, which is what makes the two
stages' C-indices comparable.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ntp_model.ntp_transformer import NTPTransformer
from ..probing.pooling import pool_hidden_states
from .survival_core import cox_ph_loss

TASKS = ("regression", "binary", "multiclass", "survival")
POOLINGS = ("last", "mean", "max", "attention")


class OutcomeHead(nn.Module):
    """Small MLP task head on the pooled states: Linear -> GELU -> Dropout -> Linear(out_dim)."""

    def __init__(self, d_model: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.1,
                 attention_pooling: bool = False):
        """hidden_dim defaults to d_model // 2. attention_pooling adds pool_score, the learned
        position-score vector of pooling="attention"."""
        super().__init__()
        hidden_dim = hidden_dim or d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        # pooling="attention" only: the per-position score vector read by
        # NTPOutcomeModel._pool_hidden_states. It lives in the head so the optimizer
        # trains it at the head lr in every strategy. Zero init = uniform weights, so
        # training starts from exact mean pooling.
        if attention_pooling:
            self.pool_score = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw head outputs [B, out_dim] of the pooled states x [B, D]."""
        return self.net(x)


class NTPOutcomeModel(nn.Module):
    """Pretrained NTPTransformer backbone + pooling + task head (see module docstring)."""

    def __init__(
        self,
        backbone: NTPTransformer,
        task: Literal["regression", "binary", "multiclass", "survival"],
        num_classes: int | None = None,      # multiclass only (K >= 2)
        pooling: Literal["last", "mean", "max", "attention"] = "attention",
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        pos_weight: torch.Tensor | None = None,      # binary imbalance weight
        class_weights: torch.Tensor | None = None,   # multiclass CE weights [K]
    ):
        """Refuses an unknown task, and a multiclass task with fewer than two classes.
        freeze_backbone starts with the backbone's gradients off (unfreeze_backbone turns
        them back on)."""
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}.")
        if task == "multiclass":
            if num_classes is None or num_classes < 2:
                raise ValueError(f"multiclass requires num_classes >= 2, got {num_classes}.")
        self.task = task
        self.num_classes = num_classes if task == "multiclass" else None
        self.backbone = backbone
        self.pooling = pooling
        self.freeze_backbone = freeze_backbone

        d_model = backbone.tok.emb.embedding_dim
        out_dim = self.num_classes if task == "multiclass" else 1
        self.head = OutcomeHead(d_model, out_dim, hidden_dim, dropout,
                                attention_pooling=(pooling == "attention"))

        # Imbalance weights: derived from the training data and re-supplied at
        # construction; non-persistent so checkpoints load regardless
        self.register_buffer("pos_weight", None, persistent=False)
        self.register_buffer("class_weights", None, persistent=False)
        if pos_weight is not None:
            self.pos_weight = pos_weight.float().reshape(1)
        if class_weights is not None:
            self.class_weights = class_weights.float()

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Turn the backbone's gradients off; forward then runs it without autograd."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.freeze_backbone = True

    def unfreeze_backbone(self):
        """Turn the backbone's gradients back on (strategy two_phase, after its frozen phase)."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.freeze_backbone = False

    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """[B, D] pooled states under self.pooling (probing.pooling.pool_hidden_states);
        "attention" scores the positions with the head's learned pool_score vector."""
        if self.pooling not in POOLINGS:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        scores = self.head.pool_score if self.pooling == "attention" else None
        return pool_hidden_states(hidden_states, attention_mask, self.pooling, attention_scores=scores)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Task outputs of a batch: "pooled" [B, D] and "predictions" (binary: probability;
        multiclass: class probabilities; survival: log relative risk; regression: value),
        plus "logits" and "probs" (binary, multiclass; multiclass also "pred_class"), "risk"
        (survival) and, when the batch has labels, "loss". The labels never reach the
        backbone. A frozen backbone runs without autograd."""
        backbone_batch = {k: v for k, v in batch.items() if k != "labels"}
        backbone_batch["labels"] = None  # No token loss

        with torch.set_grad_enabled(not self.freeze_backbone and torch.is_grad_enabled()):
            backbone_out = self.backbone(backbone_batch, hidden_only=True)

        pooled = self._pool_hidden_states(backbone_out["hidden_states"], batch["attention_mask"])
        raw = self.head(pooled)                                     # [B, out_dim]

        out: dict[str, torch.Tensor] = {"pooled": pooled}
        if self.task == "binary":
            out["logits"] = raw.squeeze(-1)                         # [B]
            out["probs"] = torch.sigmoid(out["logits"])
            out["predictions"] = out["probs"]
        elif self.task == "multiclass":
            out["logits"] = raw                                     # [B, K]
            out["probs"] = torch.softmax(raw, dim=-1)
            out["predictions"] = out["probs"]
            out["pred_class"] = raw.argmax(dim=-1)
        elif self.task == "survival":
            # Log relative risk: unbounded, higher = shorter survival. No probability
            # exists to report - a Cox model leaves the baseline hazard unspecified - so
            # "probs" is deliberately absent and every consumer reads "predictions".
            out["risk"] = raw.squeeze(-1)                           # [B]
            out["predictions"] = out["risk"]
        else:  # regression
            out["predictions"] = raw.squeeze(-1)                    # [B]

        labels = batch.get("labels")
        if labels is not None:
            if self.task == "binary":
                out["loss"] = F.binary_cross_entropy_with_logits(
                    out["logits"], labels.float(), pos_weight=self.pos_weight)
            elif self.task == "multiclass":
                out["loss"] = F.cross_entropy(out["logits"], labels.long(), weight=self.class_weights)
            elif self.task == "survival":
                # labels: [B, 2] = (duration, event). The risk set is this batch only -
                # see cox_ph_loss for what that costs and why batch_size matters here.
                labels = labels.float()
                out["loss"] = cox_ph_loss(out["risk"], labels[:, 0], labels[:, 1])
            else:
                out["loss"] = F.huber_loss(out["predictions"], labels.float())
        return out
