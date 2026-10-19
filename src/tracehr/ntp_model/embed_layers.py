# src/tracehr/ntp_model/embed_layers.py
"""The input embeddings: tokens, event types and times (Time2Vec).

NTPTransformer sums them per position: token + type + time, the time part a Time2Vec of
time_age, of time_abs, or of both (DualTime2Vec). Times are float64 days.
"""

import math

import torch
import torch.nn as nn

# Time2Vec frequency ladder (days): K sin/cos pairs with geometrically spaced periods.
# Fixed, not learned: d(phase)/dW = t, so at ages of decades a learned frequency either never
# leaves its init (Adam moves it ~lr per step) or scrambles every late-life phase. Stored as a
# buffer, so each checkpoint carries the exact ladder it was trained with.
T2V_MIN_PERIOD_DAYS = 7.0       # 1 week: exactly the weekday on absTime's fixed calendar
T2V_MAX_PERIOD_DAYS = 182625.0  # 500 years: near-linear over 0-120 years, so the slowest
                                # pairs carry the long-term trend (no unbounded linear term)
T2V_DAILY_PERIOD_DAYS = 1.0     # daily cycle: exactly the time of day on absTime, 1 minute =
                                # 0.0044 rad. On whole days it is constant.


class TokenEmbedding(nn.Module):
    """Token ids -> d_model vectors; the <PAD> row is zero and receives no gradient."""

    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0):
        """The table is the attribute `emb`, so it is never confused with a tensor .weight."""
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """[B,T] token ids -> [B,T,D]."""
        return self.emb(ids)


class TypeEmbedding(nn.Module):
    """Event-type ids -> d_model vectors; type 0 (pad / missing / unknown) is a zero row that
    receives no gradient."""

    def __init__(self, num_types: int, d_model: int, pad_id: int = 0):
        """num_types counts the reserved type 0."""
        super().__init__()
        self.emb = nn.Embedding(num_types, d_model, padding_idx=pad_id)

    def forward(self, types: torch.Tensor) -> torch.Tensor:
        """[B,T] int64 type ids -> [B,T,D] float32."""
        return self.emb(types)


class Time2Vec(nn.Module):
    """Time2Vec: t -> [sin(W_k t), cos(W_k t)]_k / sqrt(K) -> Linear to d_model.

    The features have unit norm, so the embedding has the same scale at every t, and a
    constant offset in t (absTime's reference date) is a fixed rotation per pair that the
    projection absorbs. time_scale is days per model time unit (365.25: t enters in years).
    With daily_cycle the first pair gets the 1-day period and the other K-1 keep the
    1 week -> 500 year ladder.
    """

    def __init__(self, out_dim: int, fourier_dims: int, time_scale: float = 1.0,
                 daily_cycle: bool = False):
        """Build the fixed frequency buffer W (float64) and the projection to out_dim."""
        super().__init__()
        k = max(1, int(fourier_dims))
        self.time_scale = float(time_scale)
        n_ladder = k - 1 if daily_cycle else k
        periods = torch.logspace(math.log10(T2V_MIN_PERIOD_DAYS), math.log10(T2V_MAX_PERIOD_DAYS),
                                 n_ladder, dtype=torch.float64)                       # [K] days
        if daily_cycle:
            periods = torch.cat([torch.tensor([T2V_DAILY_PERIOD_DAYS], dtype=torch.float64), periods])
        # Float64: a float32 frequency is off by up to 6e-8, which over absTime's ~46,000
        # days drifts the daily phase by minutes
        self.register_buffer("W", 2 * math.pi * self.time_scale / periods)            # [K] rad per unit
        self.proj = nn.Linear(2 * k, out_dim, bias=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """[B,T] float64 days -> [B,T,D].

        The angles reach ~3e5 rad (absTime's daily pair), so they are built in float64 -
        float32 holds them only to minutes - and only the unit-norm features are cast to
        the module's dtype (also under AMP).
        """
        if t.dtype != torch.float64:
            raise TypeError(f"Time2Vec takes float64 times (minute precision at any age), got "
                            f"{t.dtype}: the data path cast the times to a lower precision.")
        ang = (t / self.time_scale).unsqueeze(-1) * self.W                       # [B,T,K] float64
        x = torch.cat([ang.sin(), ang.cos()], dim=-1) / math.sqrt(ang.size(-1))  # [B,T,2K] unit norm
        return self.proj(x.to(self.proj.weight.dtype))                           # [B,T,D]


class DualTime2Vec(nn.Module):
    """Dual Time2Vec for Age and absTime: one Time2Vec per signal, summed and scaled by 1/sqrt(2).

    Age places an event in the patient's life, absTime in the calendar (era, birth cohort,
    weekday). Summed rather than gated: a learned convex gate on the token embedding never
    left 0.5 in training (its output is a product of three near-zero init factors), and the
    transformer blocks already weigh age against calendar per token. The 1/sqrt(2) keeps the
    combined embedding on the scale of a single Time2Vec, so single- and dual-time models
    start with the same time-signal strength.
    """
    def __init__(
        self,
        d_model: int,
        fourier_dims: int = 16,
        time_scale_age: float = 1.0,   # days per model time unit for Age
        time_scale_abs: float = 1.0,   # days per model time unit for absTime
        daily_cycle: bool = False,     # add the 1-day period to both ladders (NTPTransformer: always)
    ):
        """One Time2Vec for Age and one for absTime."""
        super().__init__()
        self.time_age = Time2Vec(d_model, fourier_dims, time_scale=time_scale_age,
                                 daily_cycle=daily_cycle)                          # Age to D-dim
        self.time_abs = Time2Vec(d_model, fourier_dims, time_scale=time_scale_abs,
                                 daily_cycle=daily_cycle)                          # absTime to D-dim

    def forward(self, t_age: torch.Tensor, t_abs: torch.Tensor) -> torch.Tensor:
        """t_age, t_abs: [B,T] days -> [B,T,D]."""
        return (self.time_age(t_age) + self.time_abs(t_abs)) / math.sqrt(2.0)
