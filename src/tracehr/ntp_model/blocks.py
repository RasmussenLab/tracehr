# src/tracehr/ntp_model/blocks.py
"""One pre-norm transformer block: attention and a SwiGLU MLP, each with a residual."""

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import SwiGLU


class TransformerBlock(nn.Module):
    """LayerNorm -> causal self-attention (RoPE on q/k) -> residual -> LayerNorm ->
    SwiGLU MLP -> residual.

    The SwiGLU inner width defaults to (8/3) * d_model rounded to a multiple of 64, which
    gives ~8*d^2 parameters (three inner x d matrices), the size of the standard 4x GELU
    MLP; an explicit mlp_inner_dim (from a saved config) overrides the formula.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, bias: bool = False,
                 use_sdpa: bool = True, store_attn: bool = False, mlp_inner_dim: int | None = None):
        """Build the two norms, the attention (SDPA unless attention weights are captured)
        and the MLP; mlp_inner_dim exposes the resolved width for the model config."""
        super().__init__()
        # Norm layers
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)

        # Attention
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=dropout,
            resid_dropout=dropout,
            bias=bias,
            use_sdpa=use_sdpa,   # PyTorch's scaled dot-product attention (SDPA)
            store_attn=store_attn,  # capture attention weights (forces the manual path)
            )

        # MLP
        inner = mlp_inner_dim if mlp_inner_dim is not None else max(64, int(round((8.0 / 3.0) * d_model / 64.0)) * 64)
        self.mlp_inner_dim = inner  # the resolved width, recorded in the model config
        self.mlp = SwiGLU(
            d_model=d_model,
            inner_dim=inner,
            dropout=dropout,
            bias=bias
            )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        """x [B,T,D] -> [B,T,D]. rope_cos/rope_sin are [T,Dh] (index positions) or [B,T,Dh]
        (Age-driven positions); attn_mask is [B,T] (1 = token, 0 = pad)."""
        # Pre-norm + residual attention (RoPE applied to q/k inside the attention)
        x = x + self.attn(self.ln1(x), attn_mask, rope_cos, rope_sin)
        # Pre-norm + residual MLP (SwiGLU)
        x = x + self.mlp(self.ln2(x))
        return x
