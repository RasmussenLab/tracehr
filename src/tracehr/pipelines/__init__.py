# src/tracehr/pipelines/__init__.py
"""The end-to-end stages, one config-driven module per pipeline, each wiring the other tracehr
packages into one run.
Pipelines: all (01-06), one module each.

The stages are importable so they can be driven programmatically after
`pip install tracehr`. Each stage module exposes `main(config_path)` (the stage
logic) and `cli()` (argparse wrapper). Example:

    from tracehr.pipelines import train_model
    train_model.main("configs/01_train_model.yaml")

Each stage is also a console command (see [project.scripts] in pyproject.toml):
    tracehr-train / tracehr-evaluate / tracehr-generate / tracehr-embed /
    tracehr-outcome / tracehr-finetune
    (all take --config PATH)

and remains runnable as a module: `python -m tracehr.pipelines.train_model --config ...`.
The six stages above are imported here.
"""

from . import (
    train_model,
    evaluate_model,
    generate_tokens,
    extract_embeddings,
    outcome_probe_prediction,
    finetune_model,
)

__all__ = [
    "train_model",
    "evaluate_model",
    "generate_tokens",
    "extract_embeddings",
    "outcome_probe_prediction",
    "finetune_model",
]
