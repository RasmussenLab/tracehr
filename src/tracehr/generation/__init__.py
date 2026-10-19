# src/tracehr/generation/__init__.py
"""Using the trained model as a next-token predictor: test metrics and token analysis
(pipeline 02), and generation from anchor-truncated histories with its explanations
(pipeline 03). model_io and attribution are shared with the later pipelines.
Pipelines: 02 and 03; model_io in 02-06, attribution in 03 and 06.

Modules:
    model_io         checkpoint loading and checks, metrics files (shared)
    attribution      the signed attribution methods and their sign colours (shared)
    evaluation       next-token test metrics (pipeline 02)
    token_analysis   what the model learned about each vocabulary token (pipeline 02)
    predict          autoregressive generation from anchor-truncated histories (pipeline 03)
    explainability   attributing generated tokens to their context (pipeline 03)
"""
