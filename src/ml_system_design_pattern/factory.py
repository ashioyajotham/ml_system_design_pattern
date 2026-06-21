"""Factory pattern for model creation."""

from __future__ import annotations

from typing import Callable

from torch import nn


class ModelFactory:
    """Registry-backed factory for PyTorch model builders."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str, builder: Callable[..., nn.Module]) -> None:
        self._builders[name] = builder

    def create(self, name: str, **kwargs) -> nn.Module:
        if name not in self._builders:
            available = ", ".join(sorted(self._builders)) or "<empty>"
            raise ValueError(f"Unknown model '{name}'. Available models: {available}")
        return self._builders[name](**kwargs)
