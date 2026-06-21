"""Composite pattern for model ensembles."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


class ModelEnsemble(nn.Module):
    """Runs child models and reduces their predictions."""

    def __init__(self, models: Iterable[nn.Module], reduction: str = "mean") -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        if not self.models:
            raise ValueError("ModelEnsemble requires at least one model")
        if reduction != "mean":
            raise ValueError(f"Only 'mean' reduction is currently supported. Got: '{reduction}'")
        self.reduction = reduction

    def forward(self, inputs: Tensor) -> Tensor:
        predictions = [model(inputs) for model in self.models]
        stacked = torch.stack(predictions, dim=0)
        return stacked.mean(dim=0)
