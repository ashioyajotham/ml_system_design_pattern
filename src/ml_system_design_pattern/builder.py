"""Builder pattern for assembling an ML experiment."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn, optim

from .composite import ModelEnsemble
from .decorators import TimingDecorator, ValidationDecorator
from .factory import ModelFactory


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for assembling an experiment."""

    model_name: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    lr: float = 1e-3
    use_validation: bool = True
    use_timing: bool = True


class ExperimentBuilder:
    """Incrementally configures and assembles trainable components."""

    def __init__(self, factory: ModelFactory) -> None:
        self.factory = factory

    def build_model(self, config: ExperimentConfig) -> nn.Module:
        model = self.factory.create(
            config.model_name,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )
        if config.use_validation:
            model = ValidationDecorator(model)
        if config.use_timing:
            model = TimingDecorator(model)
        return model

    def build_ensemble(self, *models: nn.Module) -> ModelEnsemble:
        return ModelEnsemble(models)

    def build_optimizer(self, model: nn.Module, config: ExperimentConfig) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=config.lr)

    def build_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()
