"""Decorator pattern for wrapping model behavior."""

from __future__ import annotations

import time

from torch import Tensor, nn


class ModuleDecorator(nn.Module):
    """Base decorator that delegates calls to a wrapped module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs: Tensor) -> Tensor:  # pragma: no cover - simple delegation
        return self.module(inputs)


class ValidationDecorator(ModuleDecorator):
    """Ensures input tensors have at least 2 dimensions."""

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() < 2:
            raise ValueError("Expected inputs with shape [batch, features, ...]")
        return self.module(inputs)


class TimingDecorator(ModuleDecorator):
    """Measures duration of each forward pass."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__(module)
        self.last_duration_sec: float | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        start = time.perf_counter()
        outputs = self.module(inputs)
        self.last_duration_sec = time.perf_counter() - start
        return outputs
