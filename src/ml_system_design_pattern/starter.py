"""Starter setup helper wiring all patterns together."""

from __future__ import annotations

from .builder import ExperimentBuilder, ExperimentConfig
from .factory import ModelFactory
from .models import build_linear, build_mlp


def create_starter_experiment(config: ExperimentConfig) -> ExperimentBuilder:
    factory = ModelFactory()
    factory.register("mlp", build_mlp)
    factory.register("linear", build_linear)
    builder = ExperimentBuilder(factory)
    builder.build_model(config)
    return builder
