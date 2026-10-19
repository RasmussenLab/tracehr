# src/tracehr/ntp_model/halo.py
"""HALO (Hyperspherical Alignment & Latent Optimization) token objective: distance-to-centroid
logits, a virtual abstain class bolted to the origin, and a radial ("soap bubble") prior on
each state's distance to ITS OWN class centroid, plus teacher-free self-distillation.

Transcribed from the reference:
  https://github.com/4rtemi5/halo   (src/losses/halo.py)
  https://pisoni.ai/posts/halo/

The reference is two pieces. HALOModel owns nn.Parameter(randn(K, D)) centroids and
mean-centers them on every forward; HALOLoss owns gamma and the derived abstain bias. Both
live here, in one module, because the scorer and the objective are one decision: this class
owns the centroids AND gamma, and build_head() hands the model a parameter-free view that
scores against them (see _CentroidScorer).

Implements the token-objective contract documented on CrossEntropyTokenLoss. Selected with
loss_mode: "halo" in the training yaml; registered in ntp_transformer.TOKEN_LOSSES.

Adaptations for next-token prediction, and nothing else:
  * forward() drops the ignored (-100) label positions FIRST, so the abstain column, the
    soft targets and the [N, V+1] softmax cover real tokens only - the reference takes
    exactly this [N, D] / [N] pair.
  * the reference's `centroid_targets` argument is gone: for an LM head, centroid row i IS
    token id i, so it is always arange(V) and the reference's where(mask, 0.0, margin)
    reduces to a scatter that never materializes the [N, V] mask. Identical result; the
    test suite pins it against a verbatim transcription of the reference.
  * forward() scores from the logits the MODEL already computed. Recomputing them here
    would run the largest matmul in the model a second time at a clinical vocabulary size.
  * one clamp guarding log(0) in the radial term (LOG_EPS below): a state landing exactly
    on its own centroid would otherwise take log(0) and produce NaN gradients.

Three deliberate departures from the reference, all measured (synthetic EHR, V=103,
d_model 64, lr 5e-4, grad_clip 1.0, 15 epochs, same data/seed/schedule as a CE baseline):
  * gamma is stored on a log scale, gamma = exp(GAMMA_LOG_SCALE * log_gamma). The
    reference's softplus parameter moves by about lr per Adam step - ~0.003% of gamma at
    lr 5e-4 - so gamma stayed at its init (19.39 -> 19.28 over 420 steps) and a frozen
    gamma matched a "learned" one. On the log scale it settles near 6 and, with the
    other two changes, test perplexity is 8.5-8.7 (2 seeds) vs 10.3 for the reference
    and 9.4-9.6 for untied CE; the reference init is too sharp.
  * the abstain bias scales with gamma (b = b0 * gamma / gamma0). A fixed bias in logit
    units lets the abstention threshold, measured in distance, drift as 1/gamma, so once
    gamma learns the origin sink shrinks to ~0.5% and stops carrying an uncertainty
    signal. Scaled, the threshold stays |b0| / gamma0 in distance units at every gamma
    (abstain ~6% of val tokens).
  * log_gamma's gradient is scaled by GAMMA_GRAD_SCALE (value unchanged). The log scale
    makes that gradient large, and the training loop clips the GLOBAL norm: unscaled it
    was ~99% of the norm for the first ~150 steps, so the clip shrank every other
    parameter's gradient up to ~140x and, through Adam's slow second moment, left the
    model taking oversized steps afterwards - an accidental learning-rate boost worth
    ~0.4 perplexity that would not transfer across data or learning rates.
The reference is still pinned by the test suite for everything else: given the same gamma
and bias, forward() is bit-identical to a verbatim transcription of it.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================ #
# HALO objective settings
# ============================================================================ #
# The reference's own defaults. These are calibrated against one another, not per-dataset
# knobs, which is why they live here rather than in the training yaml: loss_mode alone
# selects the objective, so nothing per-run has to be persisted or restored. What they
# RESOLVE TO does depend on d_model and the vocabulary size, and summary() prints that.

# Learn the temperature gamma rather than pinning it at its init (reference: True). The init
# is calibrated for states at GAMMA_INIT_R_SQ, which only holds at the start of training, so
# a frozen gamma would leave the CE term mis-scaled for the rest of the run. False keeps
# gamma at its init.
LEARN_GAMMA = True

# gamma = exp(GAMMA_LOG_SCALE * log_gamma). Adam moves a parameter by roughly lr per step
# whatever its gradient's size, so this scale sets how fast gamma can move: about
# lr * GAMMA_LOG_SCALE in log-gamma per step, i.e. ~1% per step at lr 5e-4. At 1 (a plain
# log) gamma could change by at most ~20% over a 420-step run; the reference's softplus
# parameter moved ~0.5%. 20 is the value the measurements in the header were taken with.
GAMMA_LOG_SCALE = 20.0

# Multiplies log_gamma's GRADIENT (the forward value is untouched). Adam's update does not
# depend on a constant gradient scale, so gamma learns exactly as fast as GAMMA_LOG_SCALE
# sets; what changes is gamma's weight in the training loop's global gradient-norm clip,
# which it would otherwise dominate (see the header). At 1e-4 its share of that norm is
# ~0.1%. Larger values hand the clip back to gamma; 0 would stop gamma learning.
GAMMA_GRAD_SCALE = 1e-4

# Teacher-free self-distillation: soft targets built from the model's OWN distances
# (reference: True). Off here because it measured worse: at DISTILL_TEMPERATURE 0.1 the
# teacher underflows to exactly one-hot (offtarget_mass 0), i.e. hard-target HALO, and even
# a temperature soft enough to be active (5.0, ~6.5% off-target) trailed uniform label
# smoothing on test perplexity (10.80 vs 10.34, 15 epochs). False = LABEL_SMOOTHING targets.
DISTILL = False

# Reference: 0.1. Two things at once:
#   1. the smoothing mass (only reachable with DISTILL = False),
#   2. the abstain bias, derived below from the margin between max_prob and min_prob.
# (2) applies in every configuration, so changing this also moves the origin sink.
LABEL_SMOOTHING = 0.1

# Self-distillation teacher temperature, in logit units: the true class is pinned at margin 0
# and every rival sits at -gamma*r_sq/DISTILL_TEMPERATURE. The reference reuses
# LABEL_SMOOTHING here (0.1, so gamma/T ~ 200), which underflows the teacher to exactly
# one-hot; 0.1 therefore reproduces the reference, i.e. hard-target HALO, bit for bit.
DISTILL_TEMPERATURE = 0.1

# gamma's init solves for a true-class logit of GAMMA_INIT_LOGIT at radius GAMMA_INIT_R_SQ,
# i.e. gamma = GAMMA_INIT_LOGIT / (GAMMA_INIT_R_SQ - r_sq_target) with the reference's 20.0
# and 2.0. GAMMA_INIT_R_SQ = 2.0 is an untrained state's expected distance to a unit-scale
# centroid, so both ends of that expression assume UNIT-SCALE centroids.
GAMMA_INIT_LOGIT = 20.0
GAMMA_INIT_R_SQ = 2.0

# Numerical guard on log(r_sq) in the radial term (never reached in practice; a state landing
# exactly on its centroid would otherwise produce NaN gradients).
LOG_EPS = 1e-12

# HALO centroids live at unit coordinate scale, whereas this transformer's input token
# embeddings are initialised at std 0.02 (GPT-2 init). Tying therefore shares the centroid parameter
# but presents this fixed scaled view to the transformer. It is part of the objective's
# geometry, not a user setting.
INPUT_EMBEDDING_SCALE = 0.02

# Reduction is deliberately NOT a setting: the contract requires forward() to return a
# scalar the training loop can average per token, so "mean" is the only valid choice and
# exposing it would only permit a value that breaks the loop.


class _ScaleGrad(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by `scale` in the backward.

    A Function rather than a tensor hook: a hook would hang a closure on the parameter,
    which survives state_dict loads but breaks pickling/deep-copying the module.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Return x unchanged, remembering the gradient scale."""
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        """The incoming gradient times scale (and no gradient for scale)."""
        return grad * ctx.scale, None


class _CentroidScorer(nn.Module):
    """The model's output head under HALO: shifted distance-to-centroid logits.

    Parameter-free by design. It holds the objective's BOUND scoring method rather than the
    objective itself, so the centroids stay registered in exactly one place
    (token_loss.centroids) and appear once in the state_dict. A bound method is not a
    Module/Parameter/Tensor, so nn.Module.__setattr__ leaves it as a plain attribute.
    """

    def __init__(self, score_fn):
        """score_fn is HALOTokenLoss.logits, bound to the objective."""
        super().__init__()
        self._score = score_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[..., D] hidden states -> [..., V] shifted distance-to-centroid logits."""
        return self._score(hidden_states)


class _HALOTiedEmbedding(nn.Module):
    """Parameter-free input view of HALO's unit-scale centroid table.

    The raw table remains registered exactly once on HALOTokenLoss. Input lookup uses the
    same mean-centered geometry as the scorer, scaled to the transformer's established
    std-0.02 input magnitude. Keeping this as a view avoids a second V x D parameter.
    """

    def __init__(self, weight_fn, vocab_size: int, d_model: int, pad_id: int):
        """weight_fn returns the raw centroid table; the sizes mirror nn.Embedding's attributes."""
        super().__init__()
        self._weight = weight_fn
        self.num_embeddings = vocab_size
        self.embedding_dim = d_model
        self.padding_idx = pad_id

    @property
    def weight(self) -> torch.Tensor:
        """The full input table: the mean-centered centroids at the input scale."""
        raw = self._weight()
        return INPUT_EMBEDDING_SCALE * (raw - raw.mean(dim=0, keepdim=True))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """[B,T] token ids -> [B,T,D] scaled, mean-centered centroid rows."""
        raw = self._weight()
        rows = F.embedding(ids, raw, padding_idx=self.padding_idx)
        return INPUT_EMBEDDING_SCALE * (rows - raw.mean(dim=0))


class HALOTokenLoss(nn.Module):
    """HALO: soft-target CE over V+1 classes (the origin sink) plus the radial NLL.

    Owns the class centroids and the learned temperature gamma, so the scorer it builds and
    the loss it computes can never disagree about the geometry. Every reader of gamma and of
    the abstain bias goes through gamma() / abstain_bias(), for the same reason. Both gamma's
    init and the
    -||c||^2 penalty in the logits assume centroids at UNIT scale, which is how
    nn.Parameter(torch.randn(V, D)) starts them; initializing at GPT-2's std 0.02 instead
    puts mean(c**2) ~4e-4 and switches off both mechanisms HALO consists of. The model's
    blanket init cannot reach them because a bare nn.Parameter is neither an nn.Linear nor
    an nn.Embedding, which is why init_head() has nothing to do.
    """

    loss_name = "halo"

    # token_loss is a distance/abstain/radial objective, not a cross-entropy, so
    # exp(token_loss) is not a perplexity. validate()/evaluate_model() therefore measure the
    # genuine token CE from the predictive distribution via token_cross_entropy().
    is_cross_entropy = False

    def __init__(self, d_model: int, vocab_size: int):
        """Check the module settings, derive gamma's init and the abstain bias, and create the
        unit-scale centroids and log_gamma.
        """
        super().__init__()
        if d_model <= 1:
            raise ValueError(f"HALO needs d_model > 1, got {d_model}.")
        if DISTILL and DISTILL_TEMPERATURE <= 0.0:
            raise ValueError(
                "DISTILL=True divides the true distances by DISTILL_TEMPERATURE, so it needs "
                "DISTILL_TEMPERATURE > 0 (the reference default is 0.1). Set DISTILL=False "
                "in halo.py to use plain smoothed targets instead.")
        if not (0.0 <= LABEL_SMOOTHING < 1.0):
            raise ValueError(f"LABEL_SMOOTHING must be in [0, 1), got {LABEL_SMOOTHING}.")

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.D = float(d_model)
        self.K = float(vocab_size)
        self.r_sq_target = 1.0 - (2.0 / self.D)

        init_gamma = GAMMA_INIT_LOGIT / (GAMMA_INIT_R_SQ - self.r_sq_target)

        # The abstain bias is a DERIVED CONSTANT rather than a parameter, which is the
        # reference's point: the origin sink costs nothing to add.
        if LABEL_SMOOTHING > 0:
            # Exact expected target probabilities strictly over K classes
            max_prob = 1.0 - LABEL_SMOOTHING + (LABEL_SMOOTHING / self.K)
            min_prob = LABEL_SMOOTHING / self.K
        else:
            max_prob = 0.99
            min_prob = 0.01 / self.K
        margin_ce = math.log(max_prob / min_prob)
        t_ideal = init_gamma * (1.0 - self.r_sq_target)
        # The reference's bias, derived at the INIT gamma. abstain_bias() scales it with the
        # current gamma; both floats are rebuilt from the constants, never stored.
        self.abstain_bias_init = t_ideal - margin_ce
        self.gamma_init = init_gamma

        # gamma is stored on a log scale (see GAMMA_LOG_SCALE). Named log_gamma, not gamma, so
        # a checkpoint holding the reference's softplus parameter fails to load (missing key)
        # instead of being read as exp(GAMMA_LOG_SCALE * 19.4) = inf.
        self.log_gamma = nn.Parameter(
            torch.tensor([math.log(init_gamma) / GAMMA_LOG_SCALE], dtype=torch.float32),
            requires_grad=LEARN_GAMMA)

        # Unit scale, NOT the model's std 0.02 - see the class docstring.
        self.centroids = nn.Parameter(torch.randn(vocab_size, d_model, dtype=torch.float32))

    # ── Temperature and origin sink ───────────────────────────────────────────

    def gamma(self) -> torch.Tensor:
        """The temperature, shape [1]. The single definition every reader goes through.
        log_gamma's gradient passes through _ScaleGrad (see GAMMA_GRAD_SCALE)."""
        return torch.exp(GAMMA_LOG_SCALE * _ScaleGrad.apply(self.log_gamma, GAMMA_GRAD_SCALE))

    def abstain_bias(self, gamma: torch.Tensor | None = None) -> torch.Tensor:
        """The abstain class's shifted logit at the current gamma, 0-dim.

        b = b0 * gamma / gamma0. Every token logit is gamma times a distance, so a fixed b
        would put the abstention threshold at |b| / gamma in distance units - moving as gamma
        learns. Scaling b with gamma pins it at |b0| / gamma0: the abstain class outscores a
        state's nearest token exactly when the origin is closer to the state than that
        centroid by more than this margin, at every gamma.
        Differentiable in gamma (the variant the header's numbers were measured with).
        """
        gamma = self.gamma() if gamma is None else gamma
        return (self.abstain_bias_init * gamma / self.gamma_init).reshape(())

    # ── Scorer ────────────────────────────────────────────────────────────────

    def centered_centroids(self) -> torch.Tensor:
        """Class centroids [V, D], mean-centered, as HALOModel.forward does every step.

        Centering is not an optimization: the -||c||^2 penalty in the logits is not
        shift-invariant in c, so a shared offset across the table would be charged to every
        class alike and read as "everything is far from everything". Centering lives here
        alone, so the head and the loss can never score against differently-centered tables.
        """
        return self.centroids - self.centroids.mean(dim=0, keepdim=True)

    def centroid_table(self) -> torch.Tensor:
        """Raw shared table for the parameter-free tied input embedding view."""
        return self.centroids

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The reference's logits_k_shifted, for [..., D] states.

        Softmax is shift-invariant, so the -||x||^2 of the true squared distance factors out
        of every row. What is left is an ordinary dot product with an L2 penalty on the
        centroids - which is why this is a drop-in for a linear head and not a slow pairwise
        distance.
        """
        pos = hidden_states.to(torch.float32)
        cen = self.centered_centroids()
        y_sq = cen.pow(2).mean(dim=-1, keepdim=True)
        dot_product = (pos @ cen.T) / self.D
        return self.gamma() * (2.0 * dot_product - y_sq.T)

    def build_head(self) -> nn.Module:
        """A parameter-free view onto this objective's centroid table (see _CentroidScorer)."""
        return _CentroidScorer(self.logits)

    def init_head(self, head: nn.Module) -> None:
        """Nothing to do: the centroids are bare nn.Parameters created at unit scale in
        __init__, which the model's blanket nn.Linear/nn.Embedding init cannot reach."""
        return None

    def bind_embeddings(self, head: nn.Module, token_embedding: nn.Module) -> None:
        """Replace the ordinary input table with a scaled view of these centroids."""
        pad_id = token_embedding.emb.padding_idx
        token_embedding.emb = _HALOTiedEmbedding(
            self.centroid_table, self.vocab_size, self.d_model, pad_id)

    def adjust_coupled_log_rate(
        self, log_rate: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Restore HALO's softmax-invariant state-norm term for absolute intensities.

        The scorer returns gamma * (2 x.c - ||c||^2), with every squared norm divided by
        D. Subtracting gamma * ||x||^2 / D turns each implicit event intensity into
        -gamma * ||x-c||^2 / D. Applied after logsumexp, this avoids another [B,T,V]
        allocation because the omitted term is common to every event class.
        """
        x_sq = hidden_states.float().pow(2).mean(dim=-1)
        return log_rate - self.gamma() * x_sq

    # ── Objective ─────────────────────────────────────────────────────────────

    def forward(
        self,
        logits: torch.Tensor,                            # [B, T, V] from the scorer
        labels: torch.Tensor,                            # [B, T], -100 = not scored
        hidden_states: torch.Tensor | None = None,       # [B, T, D], required here
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(scalar loss, diagnostics) for the scorer's logits: soft-target CE over the V real
        classes plus the abstain class, plus the radial NLL of each state around its own
        centroid. hidden_states is required.
        """
        if hidden_states is None:
            raise ValueError("HALO scores states against centroids and needs hidden_states.")

        # Drop the ignored positions first: the reference takes the [N, D] / [N] pair of
        # scored rows, so the abstain column is appended to N real rows, not to padded ones.
        flat_labels = labels.reshape(-1)
        keep = flat_labels != -100
        target = flat_labels[keep]                                          # [N]
        pos = hidden_states.reshape(-1, self.d_model)[keep].to(torch.float32)  # [N, D]
        logits_k_shifted = logits.reshape(-1, self.vocab_size)[keep].to(torch.float32)

        cen = self.centered_centroids()
        gamma = self.gamma()
        x_sq = pos.pow(2).mean(dim=-1, keepdim=True)

        # The abstain class acts as an origin sink at the theoretically ideal equilibrium,
        # its logit scaled with gamma (see abstain_bias).
        logit_abstain_shifted = self.abstain_bias(gamma).to(pos.dtype).expand(pos.size(0), 1)
        logits_k_plus_1 = torch.cat([logits_k_shifted, logit_abstain_shifted], dim=-1)

        # True absolute distances, strictly for distillation and the diagnostics.
        # (Clamping against float errors.)
        logits_k_true = torch.clamp(logits_k_shifted - (gamma * x_sq), max=0.0)

        with torch.no_grad():
            if DISTILL:
                # The true class is pinned to margin 0 and every rival sits at its own
                # (negative) distance divided by the temperature. scatter() rather than the
                # reference's where(mask, ...) because centroid row i IS token id i here, so
                # no [N, V] mask has to be materialized.
                margin = logits_k_true / DISTILL_TEMPERATURE
                target_probs_k = F.softmax(
                    margin.scatter(1, target.unsqueeze(1), 0.0), dim=-1)
                target_probs = torch.cat(
                    [target_probs_k, torch.zeros_like(logit_abstain_shifted)], dim=-1)
            else:
                # Label smoothing over the K REAL classes, abstain held at a strict 0.0.
                target_probs = torch.full_like(logits_k_plus_1, LABEL_SMOOTHING / self.K)
                target_probs.scatter_(
                    1, target.unsqueeze(1), 1.0 - LABEL_SMOOTHING + (LABEL_SMOOTHING / self.K))
                target_probs[:, -1] = 0.0

        # Uses the shifted, un-clamped logits for smooth gradients.
        loss_ce = F.cross_entropy(logits_k_plus_1, target_probs, reduction="mean")

        # Geometric regularizer on the distance from each state to its OWN target centroid -
        # a per-class compactness prior, not a norm constraint on the state. The log term
        # repels (a D-dim Gaussian's mass sits on a shell, not at a point) and the quadratic
        # attracts, so the minimum sits at r_sq_target = 1 - 2/D.
        r_sq = (pos - cen[target]).pow(2).mean(dim=-1).clamp_min(LOG_EPS)
        volume_term = (0.5 - 1.0 / self.D) * torch.log(r_sq)
        gaussian_term = -0.5 * r_sq
        radial_nll = -(volume_term + gaussian_term).mean()

        total_loss = loss_ce + radial_nll
        return total_loss, self._diagnostics(
            loss_ce, radial_nll, r_sq, cen, gamma, logits_k_shifted, logits_k_true, target)

    @torch.no_grad()
    def _diagnostics(self, loss_ce, radial_nll, r_sq, cen, gamma,
                     logits_k_shifted, logits_k_true, target) -> dict[str, torch.Tensor]:
        """Detached 0-dim geometry readouts for the training loop to token-weight and average.

        Without these there is no telling whether the geometry is doing anything. The three
        that matter: r_sq (each state's distance to its OWN centroid, which the radial prior
        pulls toward 1 - 2/D), centroid_r_sq (which the -||c||^2 penalty pulls down, and
        which must stay ~1 for the objective to be calibrated), and offtarget_mass (the
        soft-target probability NOT on the true class - when it reads 0 the teacher has
        underflowed to one-hot and self-distillation is inert). Computed from tensors
        forward() already produced, so nothing is recomputed.
        """
        centroid_r_sq = cen.pow(2).mean(dim=-1)
        diag = {
            "ce": loss_ce.detach(),
            "radial": radial_nll.detach(),
            "r_sq_mean": r_sq.mean(),
            "r_sq_std": r_sq.std() if r_sq.numel() > 1 else torch.zeros_like(r_sq.sum()),
            "gamma": gamma.detach().reshape(()),
            "abstain_bias": self.abstain_bias(gamma.detach()),
            "abstain_prob": self.abstain_prob(logits_k_shifted).mean(),
            "centroid_r_sq_mean": centroid_r_sq.mean(),
            "centroid_r_sq_min": centroid_r_sq.min(),
        }
        if DISTILL:
            # Same identity: the true class is pinned to margin 0, so the mass off it is
            # sigmoid(logsumexp over the other classes). Reads 0 when the targets have
            # collapsed to one-hot, i.e. when distillation is doing nothing.
            m = (logits_k_true / DISTILL_TEMPERATURE).scatter(
                1, target.unsqueeze(1), float("-inf"))
            diag["offtarget_mass"] = torch.sigmoid(torch.logsumexp(m, dim=-1)).mean()
        return diag

    def token_cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sum-reduced token CE over the V real classes, for a CE-comparable perplexity.

        The predictive distribution is the softmax of the scorer's logits - the same scores
        accuracy and the rank metrics use - so perplexity stays comparable to plain-CE runs.
        The abstain class is deliberately absent: it is not a vocabulary token.
        """
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    def abstain_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """p(abstain) per position [...], from the scorer's [..., V] logits.

        p(abstain) = exp(b) / (exp(b) + sum_k exp(l_k)) = sigmoid(b - logsumexp(l)), which is
        exact and avoids materializing a second [..., V+1] softmax. The softmax over the V
        tokens alone is conditional on not abstaining, so it overstates every token by
        1 / (1 - p(abstain)); the full distribution is probs * (1 - abstain_prob). Pass the
        UNRESTRICTED logits: masking tokens to -inf would read as extra abstention. b is the
        bias at the current gamma (abstain_bias()).
        """
        return torch.sigmoid(self.abstain_bias() - torch.logsumexp(logits.float(), dim=-1))

    def no_decay_params(self) -> Iterable[nn.Parameter]:
        """The centroids are geometry, not a weight matrix. The -||c||^2 term in the logits
        is ALREADY the intended shrinkage, and AdamW decay on top of it pulls the whole table
        toward the origin (~x0.74 in r_sq over 50k steps at lr 3e-4 / wd 0.01), quietly
        decalibrating the objective as the run gets longer. (log_gamma is 1-d, so the
        optimizer's usual rule already exempts it.)"""
        return (self.centroids,)

    def summary(self) -> str:
        """What the fixed settings RESOLVE TO for this run's d_model and vocabulary."""
        gamma = self.gamma().item()
        bias = self.abstain_bias().item()
        centroid_r_sq = self.centered_centroids().detach().float().pow(2).mean(-1).mean().item()
        lines = [
            f"[loss] {self.loss_name}: distance-to-centroid objective "
            f"(port of github.com/4rtemi5/halo)",
            f"[loss]   gamma {gamma:.4g} (init {self.gamma_init:.4g}; "
            + (f"learned on a log scale x{GAMMA_LOG_SCALE:g}, gradient x{GAMMA_GRAD_SCALE:g} "
               f"so it cannot dominate the grad-norm clip)" if LEARN_GAMMA else "frozen)")
            + f", r_sq_target {self.r_sq_target:.4f}",
            f"[loss]   abstain bias {bias:+.4g} (derived {self.abstain_bias_init:+.4g} at init, "
            f"scales with gamma): abstain outscores the nearest token when the origin is "
            f"closer than its centroid by {abs(self.abstain_bias_init) / self.gamma_init:.3g} "
            f"(mean sq. distance), at every gamma",
            f"[loss]   DISTILL={DISTILL}, LABEL_SMOOTHING={LABEL_SMOOTHING:.3g}, "
            f"DISTILL_TEMPERATURE={DISTILL_TEMPERATURE:.3g} (fixed in ntp_model/halo.py); "
            f"one shared centroid table at mean r_sq {centroid_r_sq:.3g}, input view x"
            f"{INPUT_EMBEDDING_SCALE:g}",
            "[loss]   token_loss is not a cross-entropy; compare runs on perplexity and "
            "accuracy, which are computed from the predictive distribution.",
        ]
        if DISTILL:
            # gamma/DISTILL_TEMPERATURE sets the teacher's sharpness. The true class is pinned
            # to 0.0 (the max of a tensor that is everywhere <= 0) while every other class
            # sits at -gamma*r_sq/T, so the soft targets collapse to one-hot unless that
            # ratio is O(1/r_sq). At the reference's settings it is ~200.
            ratio = gamma / DISTILL_TEMPERATURE
            lines.append(
                f"[loss]   NOTE: gamma/DISTILL_TEMPERATURE = {ratio:.0f}, so the off-target "
                f"teacher weights are exp(-{ratio:.0f}*r_sq) and the soft targets can "
                f"approach or underflow to one-hot. Watch the per-epoch offtarget_mass: a "
                f"reading of 0 means self-distillation is currently inert, not broken.")
        return "\n".join(lines)
