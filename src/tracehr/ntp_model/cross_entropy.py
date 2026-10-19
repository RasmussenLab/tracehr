# src/tracehr/ntp_model/cross_entropy.py
"""Plain token cross-entropy, and the reference implementation of the TOKEN OBJECTIVE
CONTRACT (documented on the class below). NTPTransformer selects an objective by name
through its TOKEN_LOSSES registry, so replacing the loss means writing one module like
this one, registering it there, and setting loss_mode in the training yaml. The model,
the training loop and every downstream pipeline are untouched.

An objective owns two things: the SCORER that turns hidden states into logits
(build_head) and the objective computed from them (forward). Keeping both here is what
lets an objective that scores by distance to a learned table rather than by a linear
projection drop in without editing the model.

Objective settings are module constants, never configuration: loss_mode alone selects the
objective, so nothing per-run has to be persisted or restored beyond that one string.
Plain cross-entropy has no settings, so this module declares no constants at all. A
replacement puts its own at the top of its own file, where the constraints between them
can be written down - and keeps them PRIVATE: nothing outside the module reads them, and
summary() is the only way they reach a log. That one rule is what keeps the next swap
cheap; an objective whose constants leak into pipelines cannot be removed by deleting a
file. Diagnostics follow the same route: forward() names them and they surface in the
train/validation metrics under "loss_diag", so no pipeline needs to know them by name.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyTokenLoss(nn.Module):
    """Mean next-token cross-entropy over the non-ignored label positions.

    Registered on the model as a submodule, so a replacement objective WITH parameters
    (a learned temperature, a centroid table) is picked up by model.parameters() and
    reaches the optimizer, the state_dict and count_params with no pipeline change. Plain
    cross-entropy owns none, so it adds no state_dict entries and the checkpoints of an
    existing run keep loading.

    THE CONTRACT. A replacement module provides all of the following:

        loss_name             registry key, and the value recorded in config.loss_mode
        is_cross_entropy      whether exp(mean token_loss) is a perplexity
        __init__()            takes (d_model, vocab_size)
        build_head()          the scorer: hidden_states -> logits
        init_head()           the objective's last word on the scorer's initialization
        bind_embeddings()     make input and output token representations share parameters
        adjust_coupled_log_rate() objective-specific common log-rate term
        forward()             (scalar loss, diagnostics)
        token_cross_entropy() sum-reduced CE, the perplexity path for non-CE objectives
        abstain_prob()        mass on a non-token abstain class, or None if there is none
        no_decay_params()     parameters AdamW must not weight-decay
        summary()             one-line startup report of the resolved objective
    """

    loss_name = "cross_entropy"

    # exp(mean token_loss) IS a perplexity here, so validate()/evaluate_model() report one
    # straight from the loss at no extra cost. An objective for which that does not hold
    # sets this False and those callers switch to token_cross_entropy() instead.
    is_cross_entropy = True

    def __init__(self, d_model: int, vocab_size: int):
        """Every objective is built from (d_model, vocab_size); plain CE only records them."""
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

    def build_head(self) -> nn.Module:
        """The scorer: hidden_states [..., D] -> logits [..., V].

        Supplied by the objective, because the scorer and the objective are one decision,
        but registered BY THE MODEL as `self.head`. That keeps the state_dict key at
        `head.weight` and keeps every existing consumer of model.head working.

        An instance method, not a static one: an objective whose scorer and loss SHARE
        parameters (a centroid table read by both) needs `self` to hand the scorer a view
        onto them.
        """
        return nn.Linear(self.d_model, self.vocab_size, bias=False)

    def init_head(self, head: nn.Module) -> None:
        """The objective's last word on its scorer's initialization.

        The model applies a blanket GPT-2 init (N(0, 0.02) over every nn.Linear and
        nn.Embedding) to the whole module tree, scorer included, and then calls this. A
        no-op here: std 0.02 is exactly what plain cross-entropy wants.

        An objective whose scorer is CALIBRATED re-initializes it here and so cannot be
        silently overwritten - the failure mode this exists to prevent, since a flattened
        scorer shows up only as slightly worse perplexity. The model calls this BEFORE
        binding, so an in-place re-init can never write through shared storage into
        the token embedding table.
        """
        return None

    def bind_embeddings(self, head: nn.Module, token_embedding: nn.Module) -> None:
        """Tie the conventional LM scorer to the input token table."""
        head.weight = token_embedding.emb.weight

    def adjust_coupled_log_rate(
        self, log_rate: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """CE logits already are the per-event log intensities (up to the rate offset)."""
        return log_rate

    def forward(
        self,
        logits: torch.Tensor,                            # [B, T, V] from the model's head
        labels: torch.Tensor,                            # [B, T], -100 = not scored
        hidden_states: torch.Tensor | None = None,       # [B, T, D]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(scalar mean loss, diagnostics).

        `hidden_states` is unused here and that is the point: it is the hook an objective
        scoring states directly needs, so the signature carries it unconditionally.

        Diagnostics are detached 0-dim tensors which train_one_epoch/validate token-weight,
        average and surface in their metrics dicts without knowing what they mean. Empty
        here - plain cross-entropy is fully described by the loss it returns.
        """
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return loss, {}

    def token_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sum-reduced token cross-entropy in nats over the non-ignored positions.

        The perplexity path for objectives that are not themselves a cross-entropy: those
        set is_cross_entropy = False, and validate()/evaluate_model() call this on the
        predictive distribution so perplexity stays comparable across objectives. Unused
        while is_cross_entropy is True, since exp(mean token_loss) is already the answer.

        Takes either [B, T, V] or an already-flattened [N, V].
        """
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    def abstain_prob(self, logits: torch.Tensor) -> torch.Tensor | None:
        """Probability on an abstain class outside the vocabulary, per position [...].

        An objective with such a class scores V+1 outcomes, so softmax(logits) over the V
        tokens is conditional on NOT abstaining and overstates every token's probability by
        1 / (1 - abstain_prob). predict.py reports this next to `probs` so the full
        distribution stays recoverable as probs * (1 - abstain_prob). None here: plain
        cross-entropy has no abstain class, and callers then report nothing extra.
        """
        return None

    def no_decay_params(self) -> Iterable[nn.Parameter]:
        """Parameters AdamW must not weight-decay, beyond the usual biases and 1-d params.

        An objective whose parameters are geometry rather than a weight matrix (centroid
        tables, temperatures) returns them here and train_model.py keeps decay off them.
        Plain cross-entropy owns no parameters at all.
        """
        return ()

    def summary(self) -> str:
        """One-line report of the resolved objective, printed once per run at startup."""
        return (f"[loss] {self.loss_name}: mean next-token cross-entropy over "
                f"V={self.vocab_size} tokens (ignore_index=-100), scorer tied to the "
                f"token embedding")
