from __future__ import annotations

import torch
from typing import Any
from torch import Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum, rearrange

from . import utils
from .rope import RotaryPositionalEmbedding

def sdpa(Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
        d_k = Q.size(-1)
        score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        score = score / (d_k ** 0.5)
        if mask is not None:
            score = score.masked_fill(~mask, -float("inf"))
        attention_weights = utils.softmax(x=score, dim=-1)
        output = einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
        return output

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: bool = False, theta: float = 10_000, max_seq_len: int = 4_098, device: torch.device = None, dtype: torch.dtype = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        assert d_model % num_heads == 0
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.d_k: int = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        self.q_proj_weight = utils.Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj_weight = utils.Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.o_proj_weight = utils.Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj_weight = utils.Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        self.isrope = rope
        if rope:
            self.rope = RotaryPositionalEmbedding(theta = self.theta, d_k = self.d_k, max_seq_len = self.max_seq_len, device = self.device)

    def forward(self, x: Float[Tensor, "... seq_count d_model"]):
        q = rearrange(self.q_proj_weight(x), "... seq_count (num_heads d_k) -> ... num_heads seq_count d_k", num_heads = self.num_heads, d_k = self.d_k)
        k = rearrange(self.k_proj_weight(x), "... seq_count (num_heads d_k) -> ... num_heads seq_count d_k", num_heads = self.num_heads, d_k = self.d_k)
        v = rearrange(self.v_proj_weight(x), "... seq_count (num_heads d_k) -> ... num_heads seq_count d_k", num_heads = self.num_heads, d_k = self.d_k)
        seq_count = q.shape[-2]
        if self.isrope:
            token_positions = torch.arange(seq_count, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = torch.ones(seq_count, seq_count, device=x.device, dtype=torch.bool)
        causal_mask = torch.tril(mask)
        output = sdpa(Q=q, K=k, V=v, mask=causal_mask)
        output = rearrange(output, "... num_heads seq_count d_k -> ... seq_count (num_heads d_k)", num_heads = self.num_heads, d_k = self.d_k)
        return self.o_proj_weight(output)