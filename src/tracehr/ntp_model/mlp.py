# src/tracehr/ntp_model/mlp.py
"""The SwiGLU feed-forward block of every transformer layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: w_out(SiLU(gate) * value), with gate and value from one fused
    projection of width 2 * inner_dim, then dropout on the output (0.0 for inference).

    No bias by default, as in most modern LLM blocks. The linear layers keep PyTorch's
    fan-in initialization; NTPTransformer re-initializes them with its GPT-2-style init.
    """

    def __init__(
        self,
        d_model: int,
        inner_dim: int,                 # MLP inner dimension
        dropout: float = 0.0,           # dropout on the MLP output; 0.0 for inference
        bias: bool = False              # no bias, as in most modern LLM blocks
    ):
        """Build the fused value+gate projection, the output projection and the dropout."""
        super().__init__()
        # One fused projection for value + gate reduces memory traffic and kernel launches
        self.w_in = nn.Linear(d_model, 2 * inner_dim, bias=bias)
        self.w_out = nn.Linear(inner_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[*, d_model] -> [*, d_model]."""
        # Gate and value from one matmul, then split
        g, v = self.w_in(x).chunk(2, dim=-1)        # [*, inner] each
        y = self.w_out(F.silu(g) * v)               # SwiGLU: SiLU(gate) * value
        return self.dropout(y)                      # dropout on the output
