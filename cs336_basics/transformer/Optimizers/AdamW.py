from __future__ import annotations

import torch
import math

from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional, Any

class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params: Iterable[Tensor] | Iterable[dict[str, Any]],
                 lr: float,
                 betas: tuple[float, float],
                 eps: float,
                 weight_decay: float,
                 *args, **kwargs):
        defaults = {
            "lr" : lr,
            "betas" : betas,
            "eps" : eps,
            "weight_decay" : weight_decay
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
