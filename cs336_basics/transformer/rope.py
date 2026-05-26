from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int

from einops import einsum, rearrange

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.theta = theta

        assert d_k % 2 == 0
        self.d_k = d_k

        self.max_seq_len = max_seq_len
        self.device = device

        self.positions: Int[Tensor, "max_seq_len"] = torch.arange(max_seq_len, device=device)
        pairs: Int[Tensor, "half_d_k"] = torch.arange(0, d_k // 2, device=device)
        freqs: Float[Tensor, "half_d_k"] = theta ** (-2 * pairs.float() / d_k)
        angles: Float[Tensor, "max_seq_len half_d_k"] = torch.outer(self.positions.float(), freqs)

        sin_cache: Float[Tensor, "max_seq_len half_d_k"] = torch.sin(angles)
        cos_cache: Float[Tensor, "max_seq_len half_d_k"] = torch.cos(angles)

        self.register_buffer("sin_cache", sin_cache, persistent=False)
        self.register_buffer("cos_cache", cos_cache, persistent=False)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"] = None) -> Float[Tensor, "... seq_len d_k"]:
        if token_positions == None:
            token_positions = self.positions
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]
        while sin.ndim < x_even.ndim:
            sin = sin.unsqueeze(-3)
            cos = cos.unsqueeze(-3)
        x_hat_even = x_even*cos - x_odd*sin
        x_hat_odd = x_even*sin + x_odd*cos
        out = torch.empty_like(x)
        out[..., 0::2] = x_hat_even
        out[..., 1::2] = x_hat_odd
        return out
