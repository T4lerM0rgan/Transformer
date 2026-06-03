from __future__ import annotations

from einops import einsum, rearrange

from . import attention, utils, rope
from typing import Any
from jaxtyping import Float, Int, Bool
from torch import Tensor

import torch

class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int = 4_096,
                 theta: float = 10_000,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        self.ln1 = utils.RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = utils.RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = utils.SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.attn = attention.MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, rope=True, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " batch seq_len d_model"]) -> Float[Tensor, " batch seq_len d_model"]:
        z = self.attn(self.ln1(x)) + x
        y = self.ffn(self.ln2(z)) + z
        return y