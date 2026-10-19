# src/tracehr/training/__init__.py
"""The training loop and its helpers (epochs, validation, resume rules, curves, devices),
shared so pretraining and finetuning train and resume the same way.
Pipelines: 01 (the loop), 01 and 06 (resume, plots); runtime in every model pipeline
(01-06).

Modules:
    one_epoch    one training epoch
    validation   one validation pass
    resume       what a resumed run may change and how it adapts
    plots        training curves of pipeline 01, two of them shared with 06
    runtime      seeding, devices, checkpoint pruning and the progress log of every batch loop
"""
