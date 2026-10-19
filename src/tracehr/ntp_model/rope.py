# src/tracehr/ntp_model/rope.py
"""Rotary position embeddings (RoPE) for the attention queries and keys.

NTPTransformer.rope_cos_sin builds the cos/sin tables for every forward - from the token
positions (index RoPE), from time_age in minutes (Age RoPE), or from both on separate head
dimensions (hybrid) - and every attention layer applies them with apply_rope.

In Age mode each pair of head dimensions is a clock hand with its own turn rate:
  fastest hand: one full turn per 2*pi minutes (1 rad per minute), so minute-scale gaps
                stay distinguishable;
  slowest hand: one full turn per rope_time_max_days (200 years by default), twice the
                100-year data span, so across a whole lifetime it sweeps only half a turn and
                never wraps - long gaps stay unambiguous.
The hands in between are spaced geometrically (hours, days, months, years, decades), so
reading them together pins down any gap from a minute to a lifetime. The ladder is fixed by
the 1-minute time resolution and the model config; the [rope] line printed at startup shows
it for the run's head_dim.
"""

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate every (even, odd) feature pair by 90 degrees: (x1, x2) -> (-x2, x1), the
    imaginary part of multiplying by i."""
    x1, x2 = x[..., ::2], x[..., 1::2]                       # [*, D/2] each
    return torch.stack((-x2, x1), dim=-1).flatten(-2)        # [*, D]


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k ([B,H,T,Dh]) by the RoPE angles whose cos/sin are given.

    cos/sin are [T,Dh] (index positions, shared across the batch) or [B,T,Dh] (Age-driven
    positions, one table per sequence), already sized to T: NTPTransformer.rope_cos_sin owns
    the sizing, so a mismatch is a bug, not a slice to absorb silently.
    """
    if cos.dim() not in (2, 3):
        raise ValueError(f"RoPE cos/sin must be rank 2 ([T,Dh]) or 3 ([B,T,Dh]), got {tuple(cos.shape)}.")
    if cos.shape[-2] != q.shape[-2]:
        raise ValueError(
            f"RoPE cos/sin cover {cos.shape[-2]} positions but q/k have {q.shape[-2]}; "
            "pass the tensors returned by NTPTransformer.rope_cos_sin(batch, T)."
        )
    # cos/sin arrive in float32; cast them to q/k's dtype (float16 under AMP)
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    if cos.dim() == 2:
        cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)  # [1,1,T,Dh]
    else:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)                            # [B,1,T,Dh]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def rope_freqs(n_dims: int, base: float, *, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The geometric frequency spread of one n_dims-wide rotary block -> [n_dims/2].

    Wavelengths run from 2*pi to 2*pi*base^((n-1)/n) position units, n = n_dims/2.
    """
    # The expression below is load-bearing for bit-identity with the original index RoPE:
    # do not rewrite it as base ** (-j / n_pairs), which differs in the last ULP.
    assert n_dims % 2 == 0, "RoPE block width must be even."
    idx = torch.arange(0, n_dims, 2, device=device, dtype=dtype)  # [n_dims/2]
    return 1.0 / (base ** (idx / n_dims))                         # [n_dims/2]


def rope_cos_sin_from_positions(
    positions: torch.Tensor,
    n_dims: int,
    base: float,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for arbitrary (not necessarily integer or monotone) positions.

    positions [..., T] -> (cos, sin), each [..., T, n_dims], always returned in float32. The
    angles are computed in compute_dtype: float64 for Age positions, because the fastest
    pair's angle reaches ~5e7 rad over a 100-year lifetime in minutes, where float32 cannot
    even resolve whole radians (its spacing there is 4 rad).
    """
    freqs = rope_freqs(n_dims, base, device=positions.device, dtype=compute_dtype)
    angles = positions.to(compute_dtype)[..., None] * freqs   # [...,T,n_dims/2]
    angles = torch.repeat_interleave(angles, 2, dim=-1)       # [...,T,n_dims]
    return angles.cos().float(), angles.sin().float()


def index_rope_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [seq_len, head_dim] for the token positions 0..seq_len-1, computed in
    float32 (half precision would lose angle precision). NTPTransformer builds them for every
    forward; nothing is cached."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)         # [T]
    return rope_cos_sin_from_positions(pos, head_dim, base)                 # [T, Dh] each
