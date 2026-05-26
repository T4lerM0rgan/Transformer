from __future__ import annotations

from typing import Any

import numpy.typing as npt
import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int

from einops import einsum, rearrange


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None, *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        weight: Tensor = torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype)
        mean: float = 0
        var: float = 2 / (self.out_features + self.in_features)
        std: float = var ** 0.5
        weight = torch.nn.init.trunc_normal_(tensor=weight, mean=mean, std=std, a=-3 * std, b=3 * std)
        self.weight: Float[Tensor, "out_features in_features"] = torch.nn.Parameter(weight)

    def forward(self, x: Float[Tensor, "batch ... in_features"]) -> Float[Tensor, "batch ... out_features"]:
        res: Float[Tensor, "batch ... out_features"] = einsum(
            x,
            self.weight,
            "batch ... in_features, out_features in_features -> batch ... out_features"
        )
        return res

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim:int, device: torch.device = None, dtype: torch.dtype = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        weight: Tensor = torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype)
        mean: float = 0
        var: float = 1
        std: float = var ** 0.5
        weight: Tensor = torch.nn.init.trunc_normal_(tensor=weight, mean=mean, std=std, a=-3 * std, b=3 * std)
        self.weight: Float[Tensor, "num_embeddings, embedding_dim"] = torch.nn.Parameter(weight)

    def forward(self, token_ids: Float[Tensor, "... num_embeddings"]) -> Float[Tensor, "... embedding_dim"]:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        inverse_RMS = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True)+self.eps)

        result = x*inverse_RMS*self.weight

        return result.to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, device: torch.device = None, dtype: torch.dtype = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.d_ff = d_model * 8 // 3
        self.device = device
        self.dtype = dtype
        self.W1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.W3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def silu(self, x: Float[Tensor, "... d_ff"]) -> Float[Tensor, "... d_ff"]:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.W2((self.silu(self.W1(x)) * self.W3(x)))

def softmax(x: Float[Tensor, "... d_m"], dim:int) -> Float[Tensor, "... d_m"]:
    m = torch.amax(input=x, dim=dim, keepdim=True)
    shifted = x - m
    exp_shifted = torch.exp(shifted)
    den = torch.sum(exp_shifted, dim=dim, keepdim=True)
    return exp_shifted/den
