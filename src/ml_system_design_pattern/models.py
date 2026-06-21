"""Sample models used by the starter framework."""

from __future__ import annotations

from torch import nn


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def build_linear(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """Build a linear model with a factory-compatible signature."""

    return nn.Linear(input_dim, output_dim)
