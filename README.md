# ml_system_design_pattern

Starter repository that demonstrates four architecture patterns for production-grade PyTorch systems:

- **Factory**: registry-backed model creation (`ModelFactory`)
- **Decorator**: model wrappers for validation and timing (`ValidationDecorator`, `TimingDecorator`)
- **Builder**: experiment assembly from config (`ExperimentBuilder`)
- **Composite**: model ensembling (`ModelEnsemble`)

## Layout

- `/src/ml_system_design_pattern`: reusable pattern implementations
- `/tests`: focused unit tests for pattern behavior

## Quick start

```bash
pip install torch
export PYTHONPATH=src
python -m unittest discover -s tests -v
```
