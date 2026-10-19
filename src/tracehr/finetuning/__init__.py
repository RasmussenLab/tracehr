# src/tracehr/finetuning/__init__.py
"""Finetuning a pretrained NTPTransformer on one outcome (task head, loop, metrics,
explanations), apart from pretraining because it trains a task head on anchored windows.
Pipelines: 06; survival_core is shared with 05.

Modules:
    outcome_model            the backbone with pooling and a task head
    finetune                 the finetuning loop, checkpointing and resuming
    finetune_plots           the training curves of a finetuning run
    outcome_evaluation       task metrics and the test-split figure
    outcome_explain          Integrated Gradients, cohort summaries and attention maps
    survival_core            the (duration, event) label contract, C-index, Cox loss and
                             Kaplan-Meier panel (shared)
"""
