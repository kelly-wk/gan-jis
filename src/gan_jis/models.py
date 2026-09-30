"""Compact conditional WGAN-GP models used by the reproduction runner."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int, *, sigmoid: bool) -> nn.Sequential:
    modules: list[nn.Module] = []
    width = in_dim
    for _ in range(layers):
        modules.extend([nn.Linear(width, hidden), nn.LeakyReLU(0.2)])
        width = hidden
    modules.append(nn.Linear(width, out_dim))
    if sigmoid:
        modules.append(nn.Sigmoid())
    return nn.Sequential(*modules)


class Generator(nn.Module):
    def __init__(self, z_dim: int, c_dim: int, x_dim: int = 96, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.net = _mlp(z_dim + c_dim, x_dim, hidden, layers, sigmoid=True)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, condition], dim=1))


class Critic(nn.Module):
    def __init__(self, c_dim: int, x_dim: int = 96, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.net = _mlp(x_dim + c_dim, 1, hidden, layers, sigmoid=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, condition], dim=1)).view(-1)


def gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.rand(real.size(0), 1, device=real.device)
    mixed = epsilon * real + (1.0 - epsilon) * fake
    mixed.requires_grad_(True)
    score = critic(mixed, condition)
    gradient = torch.autograd.grad(
        score,
        mixed,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((gradient.norm(2, dim=1) - 1.0) ** 2).mean()
