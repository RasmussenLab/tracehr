# src/tracehr/probing/__init__.py
"""Using the trained model frozen, as a feature extractor: patient embeddings at anchors
(pipeline 04), and outcomes predicted from them by six learners compared under
cross-validation, with SHAP (pipeline 05). pooling is shared with finetuning.
Pipelines: 04 and 05; pooling also in 06.

Modules:
    pooling                        pooling a window's hidden states into one vector (shared)
    embeddings                     window embeddings, window combination and the anchor-grid
                                   extraction of pipeline 04
    window_floor                   the training window floor that 04 and 05 reuse
    embedding_outcome_prediction   the stage of pipeline 05: loop over the outcomes files,
                                   extract, compare, summarize
    anchor_windows                 the anchor-truncated windows: the shared producer and the
                                   reusable on-disk window store
    anchor_embeddings              the embedding of each (patient, anchor) history
    downstream_comparison          the six models, their metrics and figures
    ml_classification              binary learners, metrics and cross-validation
    ml_regression                  regression learners, metrics and cross-validation
    ml_survival                    survival learners and cross-validation
    cross_validation               the shared cross-validation loops
    shap_explain                   saving the fitted models and their SHAP explanations

This file imports nothing on purpose: finetuning imports pooling from this package, and
loading the learner modules here would put xgboost into every tracehr import (on
macOS, torch and xgboost loaded together crash the process).
"""
