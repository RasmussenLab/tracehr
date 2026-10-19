# src/tracehr/ntp_model/attention.py
"""Causal multi-head self-attention with rotary position embeddings."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention: fused q/k/v projection, RoPE on q and k, and an
    output projection.

    The fused SDPA kernel runs unless use_sdpa is off or store_attn is on; the manual path
    computes the same attention and, with store_attn, keeps each layer's weights
    (last_attn, [B,H,T,T]) for explainability.
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 attn_dropout: float = 0.0,
                 resid_dropout: float = 0.0,
                 bias: bool = False,
                 use_sdpa: bool = True,
                 store_attn: bool = False
                 ):
        """Build the projections and dropouts; d_model must be divisible by n_heads."""
        super().__init__()

        # d_model must split evenly across the heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropouts
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.use_sdpa = use_sdpa
        # Explainability: when set, run the manual path and keep the attention weights
        self.store_attn = store_attn
        self.last_attn = None

    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor | None,
                rope_cos: torch.Tensor,
                rope_sin: torch.Tensor
                ) -> torch.Tensor:
        """x [B,T,D] -> [B,T,D]. attention_mask is [B,T] (1 = token, 0 = pad); rope_cos /
        rope_sin are [T,Dh] (index positions) or [B,T,Dh] (Age-driven positions), already
        sized to T by NTPTransformer.rope_cos_sin."""
        B, T, D = x.shape
        qkv = self.qkv(x)                                                    # [B,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)                                       # [B,T,D] each

        # Split heads
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            """[B,T,D] -> [B,H,T,Dh]."""
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B,H,T,Dh]

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # RoPE on q and k (cos/sin arrive exactly T long; apply_rope checks it)
        q, k = apply_rope(q, k, rope_cos, rope_sin)                          # [B,H,T,Dh]

        if self.use_sdpa and not self.store_attn:
            # Sequences are right-padded, so the causal mask already hides the pad keys from
            # every real query; pad positions themselves are ignored by the loss. This keeps
            # the fused is_causal fast path with no extra mask tensor. (store_attn forces the
            # manual path below, since SDPA never exposes the weights.)
            p = self.attn_dropout.p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=p,
                is_causal=True
            )  # [B,H,T,Dh]
        else:
            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B,H,T,T]
            # Causal upper triangle = -inf
            causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=att.dtype), diagonal=1)
            att = att + causal
            # Additive key-padding mask (only needed in the manual path)
            if attention_mask is not None:
                pad_bias = (attention_mask == 0).view(B, 1, 1, T)
                att = att.masked_fill(pad_bias, float("-inf"))
            w = F.softmax(att, dim=-1)
            if self.store_attn:
                self.last_attn = w.detach()  # [B,H,T,T] true attention distribution (pre-dropout)
            w = self.attn_dropout(w)
            y = torch.matmul(w, v)                                            # [B,H,T,Dh]

        # Merge heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, D)                      # [B,T,D]
        return self.resid_dropout(self.proj(y))
