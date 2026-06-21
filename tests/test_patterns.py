import unittest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

if torch is not None:
    from ml_system_design_pattern.builder import ExperimentBuilder, ExperimentConfig
    from ml_system_design_pattern.composite import ModelEnsemble
    from ml_system_design_pattern.factory import ModelFactory
    from ml_system_design_pattern.models import build_linear, build_mlp
    from ml_system_design_pattern.starter import create_starter_experiment


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PatternTests(unittest.TestCase):
    def test_factory_builder_and_decorators(self):
        factory = ModelFactory()
        factory.register("mlp", build_mlp)
        builder = ExperimentBuilder(factory)
        cfg = ExperimentConfig(model_name="mlp", input_dim=4, hidden_dim=8, output_dim=3)

        model = builder.build_model(cfg)
        out = model(torch.randn(2, 4))

        self.assertEqual(tuple(out.shape), (2, 3))

    def test_composite_ensemble(self):
        m1 = build_linear(4, 8, 2)
        m2 = build_linear(4, 8, 2)
        ensemble = ModelEnsemble([m1, m2])

        out = ensemble(torch.randn(5, 4))

        self.assertEqual(tuple(out.shape), (5, 2))

    def test_create_starter_experiment(self):
        cfg = ExperimentConfig(model_name="linear", input_dim=4, hidden_dim=8, output_dim=2)

        builder, model = create_starter_experiment(cfg)
        out = model(torch.randn(3, 4))

        self.assertIsInstance(builder, ExperimentBuilder)
        self.assertEqual(tuple(out.shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
