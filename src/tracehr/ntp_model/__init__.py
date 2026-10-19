# src/tracehr/ntp_model/__init__.py
"""The model: the NTPTransformer (a causal decoder over patient event sequences, with its
embeddings, attention, RoPE and token objectives), defined once so training builds and every
later pipeline rebuilds it identically from a checkpoint.
Pipelines: 01-06.

Modules:
    ntp_transformer   the model, its RoPE ladder and the time-to-event likelihood
    config            NTPTransformerConfig, the architecture recorded in every checkpoint
    embed_layers      token, type and Time2Vec embeddings
    attention         causal self-attention
    mlp               the SwiGLU feed-forward block
    blocks            one pre-norm transformer block
    rope              rotary position embeddings
    cross_entropy     plain next-token cross-entropy and the token-objective contract
    halo              the HALO token objective
"""
