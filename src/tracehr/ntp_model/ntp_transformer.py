# src/tracehr/ntp_model/ntp_transformer.py
"""NTPTransformer: a causal decoder over patient event sequences, with its token objective,
its rotary positions and its time-to-event (T2E) likelihood.

The model embeds tokens, event types and times (Time2Vec), runs a stack of pre-norm
transformer blocks with RoPE (index positions, Age positions, or both), and scores the next
token with the scorer of its token objective (TOKEN_LOSSES: plain cross-entropy or HALO).
With t2e_weight > 0 the same logits also define a coupled competing-risks event rate, and the
gap to the next event is scored by its interval under a piecewise-constant hazard
(t2e_interval_nll). All times are float64 days; the module constants below fix the time
resolution, the T2E intervals and the Age-RoPE ladder.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import TransformerBlock
from .config import NTPTransformerConfig
from .cross_entropy import CrossEntropyTokenLoss
from .embed_layers import DualTime2Vec, Time2Vec, TokenEmbedding, TypeEmbedding
from .halo import HALOTokenLoss
from .rope import index_rope_cos_sin, rope_cos_sin_from_positions

# TOKEN OBJECTIVES selectable via loss_mode. The class owns its own name, so this mapping
# cannot drift from it. To add an objective: write its module (see cross_entropy.py for the
# contract), import it above, and add it to this tuple - nothing else in the repo changes.
# Every registered objective stays loadable, so checkpoints trained under one can still be
# evaluated after another is added.
TOKEN_LOSSES = {cls.loss_name: cls for cls in (CrossEntropyTokenLoss, HALOTokenLoss)}

# DAYS CONTRACT: all time inputs (time_age, time_abs, t2e time_deltas) are in DAYS.
# Internally times are normalized to years for Time2Vec/T2E numerics.
DAYS_PER_YEAR = 365.25

# TIME RESOLUTION: every model resolves time to the minute, for both kinds of time input.
# The times stay float64 from the reader into the model, so a minute is exact at any age.
# It fixes three things:
#   * The T2E's first interval. Recorded times are rounded to the minute, so a gap of "0"
#     means "within the same minute", not "instantly": the T2E scores it in its own first
#     interval, [0, 1 minute) (see T2E_INTERVAL_EDGES_DAYS).
#   * The fastest Age-RoPE rotation: 1 rad per minute (see the RoPE note below).
#   * Time2Vec's daily cycle: its first pair has a 1-day period, the time of day.
TIME_RESOLUTION_DAYS = 1.0 / 1440
_LN2 = math.log(2.0)

# TIME-ROPE LADDER. The fast end is TIME_RESOLUTION_DAYS: Age-driven RoPE positions are
# Age / TIME_RESOLUTION_DAYS, so the fastest rotation pair has wavelength 2*pi minutes and
# a 1-minute gap turns it by 1.0 rad. ROPE_TIME_MAX_DAYS is the slowest pair's wavelength,
# set to 2x the ~36,500-day (100-year) data span so the slowest rotation sweeps only pi
# across a full lifetime and never wraps - long gaps stay monotonically decodable. The
# pairs therefore span 6.28 min -> 200 y. The slow end is the default of a setting that is
# recorded per model in NTPTransformerConfig (rope_time_max_days), so editing it can never
# silently reinterpret a checkpoint; read the ends off the model as self.rope_time_*. The
# per-block frequency base is solved for from the two ends and the block's pair count, so
# widening head_dim densifies the ladder without changing its reach.
#
# Precision note: RoPE is relative, so attention depends only on Age DIFFERENCES, and the
# angles are built in float64 from the float64 time_age (see rope_cos_sin_from_positions),
# so a 1-minute difference is exact at any age.
ROPE_TIME_MAX_DAYS = 73000.0    # 200 years


def _human_days(days: float) -> str:
    """A duration in days, written in the unit that reads best (startup reports)."""
    if days < 1.0 / 24:
        return f"{days * 1440:.3g} min"
    if days < 1.0:
        return f"{days * 24:.3g} h"
    if days < 60.0:
        return f"{days:.3g} d"
    if days < 730.0:
        return f"{days / 30.44:.3g} mo"
    return f"{days / DAYS_PER_YEAR:.3g} y"


def _log1mexp_from_log(r: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(-exp(r))), stable for every float32 r, with finite gradients.

    Three regimes, each fed a harmless value where it is not selected (torch.where
    backpropagates through BOTH branches, so an inf/NaN in the unused one would still
    poison the gradient):
      r < -10      log(1 - e^-a) = log(a) - a/2 + O(a^2)   (a underflows to 0 below r ~ -104)
      a < ln 2     log(-expm1(-a))
      a >= ln 2    log1p(-exp(-a))
    """
    a = torch.exp(r)
    tiny = r < -10.0
    mid = (~tiny) & (a < _LN2)
    r_tiny = torch.where(tiny, r, torch.full_like(r, -20.0))
    a_mid = torch.where(mid, a, torch.full_like(a, 0.5))
    a_big = torch.where(tiny | mid, torch.ones_like(a), a)
    out_tiny = r_tiny - 0.5 * torch.exp(r_tiny)
    out_mid = torch.log(-torch.expm1(-a_mid))
    out_big = torch.log1p(-torch.exp(-a_big))
    return torch.where(tiny, out_tiny, torch.where(mid, out_mid, out_big))


# T2E INTERVALS. The gap from an event to the next is scored by the interval it lands in,
# under a piecewise-constant hazard lambda * exp(c_j): lambda is the coupled event rate
# (_t2e_log_rate) and c_0..c_6 the model's t2e_interval_offsets, shared by all positions. The
# bounds, in days: the same minute (TIME_RESOLUTION_DAYS), 1 hour, 1 day, 1 week, 1 month,
# 6 months and 1 year; the last interval, a year or more, needs no offset of its own.
T2E_INTERVAL_EDGES_DAYS = (TIME_RESOLUTION_DAYS, 1.0 / 24, 1.0, 7.0, DAYS_PER_YEAR / 12,
                           DAYS_PER_YEAR / 2, DAYS_PER_YEAR)
T2E_INTERVAL_NAMES = ("same minute", "1 min - 1 h", "1 h - 1 d", "1 d - 1 wk", "1 wk - 1 mo",
                      "1 - 6 mo", "6 mo - 1 y", "1 y or more")
_T2E_INTERVAL_STARTS_DAYS = (0.0,) + T2E_INTERVAL_EDGES_DAYS[:-1]
_T2E_INTERVAL_WIDTHS_DAYS = tuple(hi - lo for lo, hi in zip(_T2E_INTERVAL_STARTS_DAYS,
                                                            T2E_INTERVAL_EDGES_DAYS))
# log(width in years) of each bounded interval: the rate lambda is per year
_T2E_INTERVAL_LOG_WIDTHS = tuple(math.log(w / DAYS_PER_YEAR) for w in _T2E_INTERVAL_WIDTHS_DAYS)


def t2e_interval_index(time_deltas_days: torch.Tensor) -> torch.Tensor:
    """The interval of each gap, 0-7, as int64 [..]. A gap on a bound belongs to the later
    interval. Gaps are differences of float64 Ages, so a gap a rounding error below a bound
    counts as on it (slack: one float64 eps, relative, plus 1e-9 days, ~0.1 ms). clamp puts
    the -1 sentinel in interval 0."""
    gaps = time_deltas_days.double().clamp(min=0.0)
    edges = torch.tensor(T2E_INTERVAL_EDGES_DAYS, dtype=torch.float64, device=gaps.device)
    return torch.bucketize(gaps * (1.0 + torch.finfo(torch.float64).eps) + 1e-9, edges, right=True)


def t2e_log_hazards(log_rate: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """log H_j [.., 7], the log cumulative hazard over each bounded interval:
    log(lambda) + c_j + log(width_j in years), in float32.

    Divergence guard: the value is capped at 75 while the gradient passes straight through,
    so an overflowing rate cannot turn the gradient into inf * 0 = NaN and destroy every
    weight in one step; it still points back toward sane rates. Nothing a trained model
    emits comes near it.
    """
    log_widths = torch.tensor(_T2E_INTERVAL_LOG_WIDTHS, dtype=torch.float64)
    r = log_rate.float().unsqueeze(-1) + offsets.float() + log_widths.to(log_rate.device, torch.float32)
    return r - (r - r.clamp(max=75.0)).detach()


def t2e_interval_nll(log_rate: torch.Tensor, offsets: torch.Tensor,
                     time_deltas_days: torch.Tensor) -> torch.Tensor:
    """Per-position NLL of the observed gap's interval j [..] in float32:
    -log P(j) = H_0 + ... + H_(j-1) - log(1 - exp(-H_j)), as P(j) = S(start of j) - S(end of j)
    with S = exp(-cumulative hazard); the open last interval has P = S(1 year), so its NLL is
    H_0 + ... + H_6.

    log_rate:          log(lambda), lambda per YEAR (the model's internal unit)   [..]
    offsets:           c_0..c_6, the model's t2e_interval_offsets                 [7]
    time_deltas_days:  observed gaps in DAYS, >= 0 where scored                    [..]
    Positions the caller will mask may hold anything finite (e.g. the -1 sentinel).
    """
    r = t2e_log_hazards(log_rate, offsets)                                  # [.., 7]
    before = F.pad(torch.cumsum(torch.exp(r), dim=-1), (1, 0))              # [.., 8] H_0 + ... + H_(j-1)
    ending = F.pad(-_log1mexp_from_log(r), (0, 1))                          # [.., 8] -log(1 - e^-H_j), 0 for the last
    j = t2e_interval_index(time_deltas_days)
    return (before + ending).gather(-1, j.unsqueeze(-1)).squeeze(-1)


def t2e_median_gap(log_rate: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """The median gap in DAYS [..] float64, the T2E prediction: in the first interval whose
    cumulative hazard reaches log 2, at the point its constant hazard reaches it; past 1 year,
    where the 6-month - 1-year hazard continues, when no bounded interval does. Always > 0."""
    hazard = torch.exp(t2e_log_hazards(log_rate, offsets).double())        # [.., 7]
    after = torch.cumsum(hazard, dim=-1)
    reached = after >= _LN2
    j = torch.where(reached.any(dim=-1), reached.int().argmax(dim=-1),
                    torch.full_like(reached[..., 0], len(T2E_INTERVAL_EDGES_DAYS) - 1,
                                    dtype=torch.int64))
    starts = torch.tensor(_T2E_INTERVAL_STARTS_DAYS, dtype=torch.float64, device=hazard.device)
    widths = torch.tensor(_T2E_INTERVAL_WIDTHS_DAYS, dtype=torch.float64, device=hazard.device)

    def at(x):
        return x.gather(-1, j.unsqueeze(-1)).squeeze(-1)

    per_day = (at(hazard) / widths[j]).clamp(min=1e-300)
    return starts[j] + (_LN2 - at(after - hazard)) / per_day


def t2e_calibrated_offsets(log_rate: torch.Tensor, intervals: torch.Tensor) -> torch.Tensor:
    """The offsets c_0..c_6 [7] float64 that, at the constant rate exp(mean log(lambda)), give
    each interval exactly its observed share (the maximum-likelihood fit for one constant
    rate): interval j's cumulative hazard is -log(1 - d_j / n_j), with d_j gaps in interval j
    and n_j gaps reaching it. The shares are clamped to [1e-4, 1 - 1e-4], so an empty interval
    stays finite."""
    counts = torch.bincount(intervals.cpu(), minlength=len(T2E_INTERVAL_NAMES)).double()
    reaching = counts.flip(0).cumsum(0).flip(0)[:-1]
    share = (counts[:-1] / reaching.clamp(min=1.0)).clamp(1e-4, 1.0 - 1e-4)
    log_widths = torch.tensor(_T2E_INTERVAL_LOG_WIDTHS, dtype=torch.float64)
    return torch.log(-torch.log1p(-share)) - log_widths - log_rate.double().cpu().mean()


class NTPTransformer(nn.Module):
    """Causal decoder transformer for next-event prediction on patient histories.

    Every architecture-defining argument is recorded in self.config (NTPTransformerConfig),
    which checkpoints store and from_config rebuilds; use_sdpa and store_attn are runtime
    switches that do not change the weights. forward documents the batch it takes.
    """

    def __init__(
        self,
        vocab_size: int,                    # size of the token vocabulary
        d_model: int = 256,                 # hidden size (embedding / attention / MLP width)
        n_layers: int = 8,                  # number of transformer blocks
        n_heads: int = 8,                   # attention heads per block
        max_seq_len: int = 512,             # context window: the longest supported sequence
        use_time_age: bool = True,          # add Time2Vec features of time_age
        use_time_abs: bool = True,          # add Time2Vec features of time_abs
        time_fourier_dims: int = 16,        # sin/cos frequency pairs per Time2Vec
        use_type: bool = True,              # add event-type embeddings
        num_types: int | None = None,       # number of type ids (with the reserved 0); None disables types
        dropout: float = 0.1,               # dropout on the inputs, attention and MLP
        rope_base: float = 10000.0,         # index-block RoPE base frequency; unused when rope_mode="age"
        rope_mode: str = "index",           # "index" = token order | "age" = elapsed time | "hybrid" = head dims split between both
        rope_hybrid_split: float = 0.5,     # hybrid only: fraction of the head-dim rotation PAIRS given to the index block
        rope_time_max_days: float = ROPE_TIME_MAX_DAYS,      # Age-ladder slow end: the slowest wavelength in days
        pad_id: int = 0,                    # padding token id
        use_sdpa: bool = True,              # use the fused SDPA fast path (False = manual attention)
        store_attn: bool = False,           # capture per-layer attention weights (forces the manual path; explainability)
        mlp_inner_dim: int | None = None,   # SwiGLU inner width; None = derived from d_model by the (8/3) formula
        loss_mode: str = "cross_entropy",   # token objective, one of TOKEN_LOSSES (the registry above)
        # An objective takes no options here: its settings are module constants in its own
        # file, so loss_mode alone is everything a reload has to rebuild.
        # The only T2E setting: zero disables it; a positive value enables the coupled
        # competing-risks objective and is its multiplier in the combined loss.
        t2e_weight: float = 0.0,
        t2e_exclude_ids: Sequence[int] | None = None,  # token ids (PAD/SOS/CON/...) excluded from the T2E event rate
    ):
        """Build the embeddings, the block stack, the objective's scorer and the RoPE
        ladder, initialize the weights GPT-2-style and record the resolved config."""
        super().__init__()
        # Core settings. DAYS CONTRACT: time_age/time_abs inputs and t2e deltas are DAYS;
        # the fixed internal DAYS_PER_YEAR scale normalizes them to years for the numerics.
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_time_age = use_time_age
        self.use_time_abs = use_time_abs
        self.use_type = use_type
        self.pad_id = pad_id
        self.time_scale_age = DAYS_PER_YEAR
        self.time_scale_abs = DAYS_PER_YEAR
        # Token objective. Resolved here and used below to build both the scorer and the
        # loss, so the two can never come from different objectives.
        if loss_mode not in TOKEN_LOSSES:
            raise ValueError(
                f"loss_mode must be one of {sorted(TOKEN_LOSSES)}, got {loss_mode!r}. An "
                f"objective is added by registering it in TOKEN_LOSSES (ntp_transformer.py)."
            )
        loss_cls = TOKEN_LOSSES[loss_mode]
        self.loss_mode = loss_mode
        # One persisted source of truth for T2E architecture, data requirements and loss
        # weighting. Reject bool explicitly: bool is an int subclass, but `true` is not a
        # meaningful numeric loss weight in a configuration that promises one scalar knob.
        if isinstance(t2e_weight, bool) or not isinstance(t2e_weight, (int, float)):
            raise TypeError(f"t2e_weight must be a finite number >= 0, got {t2e_weight!r}.")
        self.t2e_weight = float(t2e_weight)
        if not math.isfinite(self.t2e_weight) or self.t2e_weight < 0.0:
            raise ValueError(f"t2e_weight must be finite and >= 0, got {t2e_weight!r}.")
        self.t2e_pred = self.t2e_weight > 0.0
        # Vocabulary mask of the non-event tokens (PAD/SOS/CON/...) kept out of the event
        # rate; non-persistent: rebuilt from the config, never loaded from a checkpoint
        self.t2e_exclude_ids = sorted(int(i) for i in t2e_exclude_ids) if t2e_exclude_ids else []
        excluded = torch.zeros(vocab_size if self.t2e_exclude_ids else 0, dtype=torch.bool)
        excluded[self.t2e_exclude_ids] = True
        self.register_buffer("_t2e_excluded", excluded, persistent=False)

        # Embeddings
        self.tok = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
        self.typ = TypeEmbedding(num_types, d_model, pad_id=0) if (use_type and num_types is not None) else None
        # Time embeddings: dual (summed) when both signals are enabled, single otherwise.
        # Time is resolved to the minute, so the first Time2Vec pair is the time of day.
        self._dual_time = use_time_age and use_time_abs
        self.time_daily_cycle = True
        if self._dual_time:
            self.time_emb = DualTime2Vec(
                d_model, time_fourier_dims,
                time_scale_age=self.time_scale_age, time_scale_abs=self.time_scale_abs,
                daily_cycle=self.time_daily_cycle,
            )
        elif use_time_age or use_time_abs:
            single_scale = self.time_scale_age if use_time_age else self.time_scale_abs
            self.time_emb = Time2Vec(d_model, time_fourier_dims, time_scale=single_scale,
                                     daily_cycle=self.time_daily_cycle)
        else:
            self.time_emb = None

        # Stack
        self.store_attn = store_attn
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout, use_sdpa=use_sdpa,
                             store_attn=store_attn, mlp_inner_dim=mlp_inner_dim)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        # The objective is CONSTRUCTED first so it can supply the scorer, but the attributes
        # are ASSIGNED in their original order: registration order is what
        # self.apply(_init_weights) walks, and nn.Linear draws RNG at construction, so
        # reordering would change the weights of every existing checkpoint.
        token_loss = loss_cls(d_model, vocab_size)
        # The objective supplies the scorer but the MODEL registers it, so the state_dict
        # key stays "head.weight" and every consumer of model.head keeps working.
        self.head = token_loss.build_head()
        # T2E interval offsets c_0..c_6: the hazard in interval j is lambda * exp(c_j), with
        # log(lambda) = logsumexp(event logits) (see T2E_INTERVAL_EDGES_DAYS). Shared by all
        # positions, they set the shape of the gap distribution and its overall level, which
        # otherwise only shifting EVERY token logit could set (a gradient that dominates
        # early training and slows token learning). Invisible to the softmax, and the
        # per-type hazards exp(logit_k + c_j) stay a consistent competing-risks model. Bare
        # parameters with a fixed init: no RNG draw, untouched by _init_weights, 1-d so the
        # optimizer's rule keeps weight decay off them. Set from the data before training by
        # calibrate_t2e_interval_offsets. With T2E off there are none.
        self.t2e_interval_offsets = (nn.Parameter(torch.zeros(len(T2E_INTERVAL_EDGES_DAYS)))
                                     if self.t2e_pred else None)
        # The objective itself. A submodule, so an objective with parameters reaches the
        # optimizer, the state_dict and count_params with no pipeline change; plain CE owns
        # none and therefore adds no state_dict entries.
        self.token_loss = token_loss

        # GPT-2-style init: N(0, 0.02) with scaled-down residual projections
        self.apply(self._init_weights)
        resid_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.mlp.w_out.weight, mean=0.0, std=resid_std)

        # The objective's last word on its own scorer: the blanket init above reaches every
        # nn.Linear/nn.Embedding in the tree, the scorer included, and a CALIBRATED scorer
        # overwritten at std 0.02 fails silently (worse perplexity, no error). A no-op for
        # plain CE. BEFORE tying, so an in-place re-init can never write through shared
        # storage into the token embedding table.
        self.token_loss.init_head(self.head)

        # Every objective is tied. Plain CE aliases the scorer to the ordinary input table;
        # HALO installs a scaled, parameter-free input view of its unit-scale centroids.
        self.token_loss.bind_embeddings(self.head, self.tok)

        # RoPE settings. cos/sin are built fresh for every forward by rope_cos_sin - there is
        # deliberately no cache; see _index_rope.
        self.rope_base = rope_base
        self.rope_mode = rope_mode
        self.rope_hybrid_split = float(rope_hybrid_split)
        # Stashed BEFORE _resolve_rope_blocks, which solves the Age base from them; the slow
        # end is recorded in self.config below, the fast end is the fixed 1-minute
        # resolution. Read them as bare self.rope_time_* everywhere - never getattr(...,
        # MODULE_CONSTANT): if the ordering here is ever broken the AttributeError is the point.
        self.rope_time_scale_days = TIME_RESOLUTION_DAYS
        self.rope_time_max_days = float(rope_time_max_days)
        self._resolve_rope_blocks()

        # Resolved architecture spec (single source of truth for checkpoints)
        self.config = NTPTransformerConfig(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            max_seq_len=max_seq_len, use_time_age=use_time_age, use_time_abs=use_time_abs,
            time_fourier_dims=time_fourier_dims, use_type=use_type, num_types=num_types,
            dropout=dropout, rope_base=rope_base, pad_id=pad_id,
            mlp_inner_dim=self.blocks[0].mlp_inner_dim,   # concrete, even when derived
            t2e_weight=self.t2e_weight, loss_mode=loss_mode,
            t2e_exclude_ids=self.t2e_exclude_ids or None,
            rope_mode=rope_mode, rope_hybrid_split=self.rope_hybrid_split,
            rope_time_max_days=self.rope_time_max_days,
        )

    def _resolve_rope_blocks(self) -> None:
        """Split head_dim between the index- and Age-driven rotary blocks.

        Splitting at PAIR granularity keeps the dim boundary even, so a rotation pair
        (dims 2j, 2j+1) is never cut in half. Each block then gets its OWN full
        geometric frequency spread over its own width - not a slice of one shared
        spread - so each keeps a complete multi-scale ladder in its own subspace.
        """
        if self.rope_mode not in ("index", "age", "hybrid"):
            raise ValueError(f"rope_mode must be 'index', 'age' or 'hybrid', got {self.rope_mode!r}.")

        # Validate the slow ladder end even in index mode, so a bad config is rejected at
        # construction rather than at the first age/hybrid run. It fails SILENTLY
        # otherwise: max_days <= 2*pi*scale gives base<=1, which INVERTS the ladder so every
        # pair rotates on a minutes scale and long gaps stop being decodable - the only
        # symptom being worse perplexity.
        lam_min = 2.0 * math.pi * self.rope_time_scale_days
        if not (math.isfinite(self.rope_time_max_days) and self.rope_time_max_days > lam_min):
            raise ValueError(
                f"rope_time_max_days must be finite and exceed the fastest wavelength "
                f"2*pi*rope_time_scale_days = {lam_min:.4g} days, got "
                f"{self.rope_time_max_days!r}. Otherwise the Age ladder inverts or collapses "
                f"and long gaps stop being decodable."
            )

        hd = self.blocks[0].attn.head_dim
        self._rope_head_dim = hd
        if self.rope_mode == "index":
            self._rope_index_dims, self._rope_time_dims = hd, 0
        elif self.rope_mode == "age":
            self._rope_index_dims, self._rope_time_dims = 0, hd
        else:
            if not (0.0 < self.rope_hybrid_split < 1.0):
                raise ValueError(
                    f"rope_hybrid_split must be strictly between 0 and 1 (both blocks "
                    f"non-empty), got {self.rope_hybrid_split}."
                )
            n_pairs = hd // 2
            if n_pairs < 2:
                raise ValueError(
                    f"rope_mode='hybrid' needs head_dim >= 4 to split rotation pairs, but "
                    f"d_model/n_heads = {hd}. Use rope_mode='index' or 'age'."
                )
            n_idx = min(max(int(self.rope_hybrid_split * n_pairs + 0.5), 1), n_pairs - 1)
            self._rope_index_dims, self._rope_time_dims = 2 * n_idx, hd - 2 * n_idx

        # Solve the Age block's base from the fixed ladder ends and its pair count
        self._rope_time_base = None
        if self._rope_time_dims:
            n_time = self._rope_time_dims // 2
            if n_time < 2:
                raise ValueError(
                    f"rope_mode={self.rope_mode!r} needs at least 2 Age rotation pairs to span "
                    f"the {self.rope_time_max_days:.0f}-day ladder, but d_model/n_heads = {hd} "
                    f"leaves {n_time}. Increase d_model or decrease n_heads."
                )
            self._rope_time_base = (self.rope_time_max_days / lam_min) ** (n_time / (n_time - 1))

    @classmethod
    def from_config(cls, config: NTPTransformerConfig) -> "NTPTransformer":
        """Rebuild a model from a saved architecture spec (as checkpoints record it)."""
        return cls(**config.to_dict())

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2-style init of one module: N(0, 0.02) weights, zero biases, a zero pad row."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def enable_attention_capture(self, flag: bool = True) -> None:
        """Turn per-layer attention-weight capture on or off at runtime (no rebuild needed).
        When on, attention runs the manual path and keeps its weights for
        get_attention_maps()."""
        self.store_attn = flag
        for blk in self.blocks:
            blk.attn.store_attn = flag

    def get_attention_maps(self) -> list:
        """Per-layer [B,H,T,T] attention weights of the last forward (None where not captured)."""
        return [blk.attn.last_attn for blk in self.blocks]

    def _index_rope(self, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Index-position cos/sin, exactly [T, self._rope_index_dims]. Stateless by design.

        Building costs ~0.05 ms (0.01-0.02% of a forward), so it is not cached: a cache
        would have to be keyed on the block WIDTH - in hybrid mode the index block is
        narrower than head_dim and carries its own frequency spread, so a cache built at full
        head_dim would slice cleanly and yield the WRONG FREQUENCIES with no shape error. No
        cache, no such bug - and no module state mutated mid-forward.

        The build is at max(T, max_seq_len), not T, which keeps the tensor's numel constant
        across every forward. That matters because torch's vectorised cos/sin is
        numel-sensitive: chunk boundaries under at::parallel_for move with numel, so
        building at a varying T would make cos/sin - and hence logits - depend on both T
        and OMP_NUM_THREADS at awkward sizes (e.g. T=257, head_dim=16).
        """
        cos, sin = index_rope_cos_sin(max(T, self.max_seq_len), self._rope_index_dims,
                                    base=self.rope_base, device=device)
        return cos[:T], sin[:T]

    def _rope_time_positions(self, batch: dict[str, torch.Tensor], T: int, device: torch.device) -> torch.Tensor:
        """Age-driven RoPE positions in minutes -> [B,T] float64 (see
        rope_cos_sin_from_positions for why float64: the fastest pair's angle reaches ~5e7
        rad over a lifetime)."""
        t = batch.get("time_age", None)
        if t is None:
            raise ValueError(
                f"rope_mode={self.rope_mode!r} derives RoPE positions from patient Age, but the "
                "batch has no 'time_age'. Every dataset in this repo emits it (computed from the "
                "timestamps); a caller gating it on model.use_time_age must gate on "
                "model.needs_time_age instead."
            )
        if t.dim() != 2 or t.shape[1] != T:
            raise ValueError(f"'time_age' must be [B,{T}] to match input_ids, got {tuple(t.shape)}.")
        if t.dtype != torch.float64:
            raise TypeError(f"'time_age' must be float64 (minute precision at any age), got "
                            f"{t.dtype}: the data path cast the times to a lower precision.")
        # self.rope_time_scale_days: the base was solved from this same value, and dividing
        # by a different one silently rescales the whole ladder (lam_max becomes
        # MAX * S_here/S_base, so the slowest pair wraps mid-lifetime).
        return t.to(device=device) / self.rope_time_scale_days

    def rope_cos_sin(
        self,
        batch: dict[str, torch.Tensor],
        T: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE cos/sin for this batch, already sized to T. The single resolution point.

        index  -> [T, Dh]      shared across the batch
        age    -> [B, T, Dh]   positions = time_age in DAYS
        hybrid -> [B, T, Dh]   dims [0,d_idx) index-driven, [d_idx,Dh) Age-driven
        """
        device = device if device is not None else batch["input_ids"].device
        if self.rope_mode == "index":
            return self._index_rope(T, device)

        # Time inputs never require grad, so no graph is built either way; explicit for clarity
        with torch.no_grad():
            pos = self._rope_time_positions(batch, T, device)                  # [B,T] float64
            cos_t, sin_t = rope_cos_sin_from_positions(
                pos, self._rope_time_dims, self._rope_time_base, compute_dtype=torch.float64)
            if self.rope_mode == "age":
                return cos_t, sin_t                                            # [B,T,Dh]
            cos_i, sin_i = self._index_rope(T, device)                         # [T,d_idx]
            B = pos.size(0)
            cos_i = cos_i.unsqueeze(0).expand(B, -1, -1)
            sin_i = sin_i.unsqueeze(0).expand(B, -1, -1)
            return torch.cat([cos_i, cos_t], dim=-1), torch.cat([sin_i, sin_t], dim=-1)

    @property
    def needs_time_age(self) -> bool:
        """Whether the batch must carry time_age: for Time2Vec, or for Age-driven RoPE positions."""
        return self.use_time_age or self.rope_mode in ("age", "hybrid")

    @property
    def needs_time_abs(self) -> bool:
        """Whether the batch must carry time_abs (for Time2Vec only). absTime is deliberately
        not a RoPE source: within a patient its differences equal Age's, so a relative
        encoding built on it would be mathematically identical."""
        return self.use_time_abs

    def loss_summary(self) -> str:
        """Human-readable report of the resolved token objective (data-dependent decisions)."""
        return self.token_loss.summary()

    def rope_summary(self) -> str:
        """Human-readable report of the resolved rotary ladder (data-dependent decision)."""
        human = _human_days

        def ladder(n_dims: int, base: float, unit: float) -> list:
            """The wavelengths of one rotary block, in the given unit."""
            n = n_dims // 2
            return [2.0 * math.pi * (base ** (j / n)) * unit for j in range(n)]

        hd, lines = self._rope_head_dim, []
        lines.append(f"[rope] mode={self.rope_mode} head_dim={hd} ({hd // 2} pairs)")
        if self._rope_index_dims:
            waves = ladder(self._rope_index_dims, self.rope_base, 1.0)
            lines.append(
                f"[rope]   index dims 0-{self._rope_index_dims - 1} "
                f"({self._rope_index_dims // 2} pairs, base={self.rope_base:.4g}): "
                + ", ".join(f"{w:.3g}" for w in waves) + " tokens"
            )
        if self._rope_time_dims:
            waves = ladder(self._rope_time_dims, self._rope_time_base, self.rope_time_scale_days)
            lines.append(
                f"[rope]   age   dims {self._rope_index_dims}-{hd - 1} "
                f"({self._rope_time_dims // 2} pairs, base={self._rope_time_base:.4g}): "
                + ", ".join(human(w) for w in waves)
            )
            # Derived, not hardcoded: the ladder reaches 2x this span before wrapping, and
            # both numbers move if the config's ends are ever changed.
            lines.append(
                f"[rope]   Age ladder reaches {self.rope_time_max_days:.0f} days "
                f"({self.rope_time_max_days / DAYS_PER_YEAR:.0f} y) - spans up to "
                f"{self.rope_time_max_days / 2 / DAYS_PER_YEAR:.0f} y of history without wrap"
            )
        if self.rope_mode == "age":
            lines.append(
                "[rope]   NOTE: same-Age events receive identical RoPE positions; their "
                "ordering is carried only by the causal mask"
            )
        return "\n".join(lines)

    def time2vec_summary(self) -> str:
        """Human-readable report of the Time2Vec periods (the first is the time of day)."""
        if self.time_emb is None:
            return "[time2vec] off (use_time_age and use_time_abs are both false)"
        t2v = self.time_emb.time_age if self._dual_time else self.time_emb
        periods = (2 * math.pi * t2v.time_scale / t2v.W).tolist()           # DAYS
        return (f"[time2vec] {len(periods)} periods: " + ", ".join(_human_days(p) for p in periods)
                + (" (first = time of day)" if self.time_daily_cycle else ""))

    def _t2e_log_rate(self, logits: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Coupled competing-risks log(lambda) [B, T] in float32.

        Token logits are per-event log intensities, so total rate is their masked
        logsumexp. The objective may restore a common term omitted from its token-softmax
        scores (HALO's -gamma*||state||^2/D); CE leaves the value untouched.
        Float32 Throughout: Under fp16 Autocast The Rate Terms In The NLL Overflow Or
        Underflow Well Inside The Range Real Gaps Drive log_rate Into.
        """
        log_rate = torch.logsumexp(self._mask_non_event_logits(logits.float()), dim=-1)
        return self.token_loss.adjust_coupled_log_rate(log_rate, hidden_states)

    @torch.no_grad()
    def calibrate_t2e_interval_offsets(self, batch: dict[str, torch.Tensor]) -> str:
        """Fit the interval offsets to this batch's interval shares (t2e_calibrated_offsets).

        All network outputs stay fixed. Only the offsets c_j are set, in closed form, so token
        logits and probabilities cannot move.
        """
        if not self.t2e_pred:
            raise RuntimeError("calibrate_t2e_interval_offsets needs t2e_weight > 0 (T2E is off).")
        offsets = self.t2e_interval_offsets
        was_training = self.training
        try:
            self.eval()
            offsets.zero_()
            out = self({k: v for k, v in batch.items() if k != "labels"})
            mask = (batch["labels"] != -100) & (batch["time_deltas"] >= 0)
            if not bool(mask.any()):
                return "[t2e] interval offsets left at 0: the calibration batch has no scored gap"
            log_rate = self._t2e_log_rate(out["logits"], out["hidden_states"])[mask]
            deltas = batch["time_deltas"][mask]
            if not (torch.isfinite(log_rate).all() and torch.isfinite(deltas).all()):
                raise ValueError("T2E calibration received non-finite rates or gaps.")
            intervals = t2e_interval_index(deltas)
            before = float(t2e_interval_nll(log_rate, offsets, deltas).mean())
            offsets.copy_(t2e_calibrated_offsets(log_rate, intervals).to(offsets))
            after = float(t2e_interval_nll(log_rate, offsets, deltas).mean())
            counts = torch.bincount(intervals.cpu(), minlength=len(T2E_INTERVAL_NAMES)).tolist()
            shares = ", ".join(f"{name} {n / intervals.numel():.3f}"
                               for name, n in zip(T2E_INTERVAL_NAMES, counts))
            return (f"[t2e] interval offsets calibrated on {intervals.numel():,} gaps (shares "
                    f"{shares}): interval NLL {before:.4f} -> {after:.4f} (token logits untouched)")
        finally:
            self.train(was_training)

    def _compute_t2e_loss(
        self,
        log_rate: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean interval NLL over the valid positions (see t2e_interval_nll).

        Each position scores -log P(the gap's interval) under the piecewise-constant hazard
        lambda * exp(c_j). The coupling of lambda to the token logits is Delphi's; the
        intervals replace its single exponential, whose one rate per position cannot fit
        same-minute gaps and gaps of months at once.

        Args:
            log_rate: log(lambda) [B, T] float32, per year (from _t2e_log_rate)
            time_deltas: Actual time to next event in days [B, T]
            mask: Valid positions [B, T] (excludes padding and -1 sentinel deltas)

        Returns:
            Scalar loss (mean NLL over valid positions)
        """
        nll = t2e_interval_nll(log_rate, self.t2e_interval_offsets, time_deltas)
        # where, not nll * mask: a masked position holding inf would make inf * 0 = NaN
        nll = torch.where(mask, nll, torch.zeros_like(nll))
        return nll.sum() / mask.float().sum().clamp(min=1)

    def _mask_non_event_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Logits with the non-event tokens (PAD/SOS/CON/...) set to -inf, so the event rate
        lambda sums real events only."""
        if self._t2e_excluded.numel():
            logits = logits.masked_fill(self._t2e_excluded, float("-inf"))
        return logits

    def _extract_t2e_predictions(self, log_rate: torch.Tensor) -> torch.Tensor:
        """Predicted gap to the next event in days -> [B, T] float64: the median of the
        distribution the loss scores (see t2e_median_gap)."""
        return t2e_median_gap(log_rate, self.t2e_interval_offsets)

    def forward(self, batch: dict[str, torch.Tensor], hidden_only: bool = False,
                input_embeds: torch.Tensor | None = None,
                last_only: bool = False) -> dict[str, torch.Tensor]:
        """Run the model on a batch of windows.

        batch holds input_ids and attention_mask [B,T], the time and type inputs the model is
        built for, and, for a loss, labels (plus time_deltas when T2E is on). Returns
        hidden_states [B,T,D] and logits [B,T,V]; predicted_time_delta with T2E; and with
        labels, loss, token_loss (and t2e_loss, t2e_count) plus the objective's loss_diag.

        hidden_only: return only hidden_states, skipping the LM head (finetuning, embeddings).
        input_embeds: [B,T,D] token embeddings used in place of the token lookup, so an
            attribution can differentiate with respect to them (explainability); every other
            input is read from the batch as usual.
        last_only: score only the last position (logits [B,1,V]), for label-free scoring of
            the next token (explainability).
        """
        # Unpack the batch
        ids: torch.Tensor = batch["input_ids"]                      # [B,T] int64
        attn: torch.Tensor = batch["attention_mask"]                # [B,T] int8/uint8 (1=token, 0=pad)
        time_age = batch.get("time_age", None)                      # [B,T] float64 age-based time
        time_abs = batch.get("time_abs", None)                      # [B,T] float64 absolute time
        typ = batch.get("type", None)                               # [B,T] int64 or None
        labels = batch.get("labels", None)                          # [B,T] int64 or None
        time_deltas = batch.get("time_deltas", None)                # [B,T] float64 or None (for T2E prediction)

        # Shapes
        B, T = ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds model max_seq_len {self.max_seq_len}.")

        # ── Embeddings ────────────────────────────────────────────────────────
        x = self.tok(ids) if input_embeds is None else input_embeds  # [B,T,D]

        # Feature mask for the pads (keeps the pad states exactly zero)
        feat_mask = (attn != 0).unsqueeze(-1).to(dtype=x.dtype)     # [B,T,1]

        # Configured features are required: a missing key means a config/data mismatch,
        # and silently training without the signal would be worse than failing loudly
        if self.typ is not None:
            if typ is None:
                raise ValueError("Model has type embeddings (use_type=True) but the batch has no 'type'.")
            x = x + self.typ(typ) * feat_mask                       # Row 0 is zero; mask is extra safety

        if self.time_emb is not None:
            if self._dual_time:
                if time_age is None or time_abs is None:
                    raise ValueError(
                        "Model uses both time signals (use_time_age and use_time_abs) but the "
                        "batch is missing 'time_age' and/or 'time_abs'."
                    )
                x = x + self.time_emb(time_age, time_abs) * feat_mask
            else:
                t = time_age if self.use_time_age else time_abs
                if t is None:
                    needed = "time_age" if self.use_time_age else "time_abs"
                    raise ValueError(f"Model uses {needed} embeddings but the batch has no '{needed}'.")
                x = x + self.time_emb(t) * feat_mask

        # ── Transformer ───────────────────────────────────────────────────────
        # Input dropout, then zero the pads
        x = self.dropout(x) * feat_mask

        # RoPE for this step ([T,Dh] for index positions, [B,T,Dh] once Age drives them)
        rope_cos, rope_sin = self.rope_cos_sin(batch, T, device=x.device)

        # Attention mask as [B,T] bool (1=token, 0=pad)
        attn_bool = (attn != 0)

        # Transformer stack (re-zero the pads after each residual)
        for blk in self.blocks:
            x = blk(x, attn_bool, rope_cos, rope_sin)
            x = x * feat_mask

        # ── Output ────────────────────────────────────────────────────────────
        hidden_states = self.ln_f(x) * feat_mask
        # hidden_only: for callers that read only the states (finetuning, embeddings). Skips the
        # LM head, whose [B,T,V] logits are the largest tensor of the forward, and all that is
        # built on them (T2E predictions, token loss).
        if hidden_only:
            return {"hidden_states": hidden_states}
        scored = hidden_states[:, -1:] if last_only else hidden_states  # the positions scored
        logits = self.head(scored)                                  # [B,T,V] ([B,1,V] last_only)
        out = {"logits": logits, "hidden_states": hidden_states}

        # ── T2E predictions (when enabled) ────────────────────────────────────
        if self.t2e_pred:
            log_rate = self._t2e_log_rate(logits, scored)                         # [B,T] float32
            out["predicted_time_delta"] = self._extract_t2e_predictions(log_rate)  # [B,T]

        # ── Token loss ────────────────────────────────────────────────────────
        if labels is not None:
            if last_only:
                raise ValueError("last_only scores the last position only, so it takes no labels.")
            # The objective owns the computation; hidden_states is passed so an objective
            # scoring states directly needs no change here. Diagnostics are surfaced only
            # when there are any - train_one_epoch/validate average whatever they contain.
            token_loss, loss_diag = self.token_loss(logits, labels, hidden_states=hidden_states)
            if loss_diag:
                out["loss_diag"] = loss_diag
            out["token_loss"] = token_loss
            out["loss"] = token_loss

            # ── Coupled competing-risks T2E loss ──────────────────────────────
            if self.t2e_pred:
                if time_deltas is None:
                    raise ValueError(
                        f"t2e_weight={self.t2e_weight:g} enables T2E but the batch has no "
                        "'time_deltas'; enable compute_time_deltas in the dataset."
                    )
                # Valid = has a next token AND a real waiting time (-1 marks the synthetic
                # preamble deltas, which can sit on label-valid positions)
                mask = (labels != -100) & (time_deltas >= 0)        # [B,T]
                t2e_loss = self._compute_t2e_loss(log_rate, time_deltas, mask)
                out["t2e_loss"] = t2e_loss
                out["t2e_count"] = mask.sum()

                out["loss"] = token_loss + self.t2e_weight * t2e_loss

        return out
