# src/tracehr/dataset/__init__.py
"""Turns the MEDS input into what the models read (checked events, vocabulary, anchors and
windows, streamed or stored), so every pipeline reads the data the same way.
Pipelines: all (01-06).

Modules:
    column_mapping       the `columns:` and `anchor_columns:` config blocks
    cohort               the patient-id split files, the anchors table and the cutoff rule
    meds_reader          the checked event view of the MEDS input and the readers built on it
    vocabulary           building, loading and fingerprinting the vocabulary
    windows              the front block, window planning and the on-disk window store
    ntp_batching         next-token labels and time deltas, the window-store dataset and collate_fn
    streaming_datasets   the streaming datasets (pretraining windows, anchor-truncated windows)
"""
