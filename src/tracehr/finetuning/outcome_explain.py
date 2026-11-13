# src/tracehr/finetuning/outcome_explain.py
"""Explainability for NTPOutcomeModel (regression | binary | multiclass | survival), run by
pipeline 06 on the test split: every test prediction, or a seeded random sample of them. Each
explained prediction gets one row per position of its window, holding:
  1. Integrated Gradients on the token input embeddings (stage 03's integrated_gradients
     applied to the head's raw scalar: binary event logit, multiclass predicted-class logit,
     regression value, survival log relative risk) against a zero-vector token baseline with the
     event times and types kept, so only token identity is attributed; completeness_residual =
     the prediction's attributions minus its output change (the n_steps discretisation error);
  2. with plot_attention, its readout attention per layer (readout_attention) and the uniform
     reference.
The rows go to <output_dir>/token_attributions.parquet (patient level, joined to
predictions.parquet on patient_id + anchor_age); explain_summary turns them into the cohort
tables and figures of <output_dir>/explainability/, which hold aggregates only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..generation.attribution import integrated_gradients
from ..training.runtime import ProgressLog, batch_to_device, resolve_device
from .explain_summary import SEED, _unit, summarise_explanations
from .finetune_plots import MIN_PATIENTS, HIDDEN

# Files earlier versions of pipeline 06 wrote into explainability/: per-patient figures, and
# the patient-level attributions now written one level up
OLD_INDIVIDUAL_FILES = ("sample_*_importance.png", "sample_*_attn.png",
                        "global_token_importance*.png", "global_token_importance*.json",
                        "token_attributions.parquet")
# The query weights r_t of readout_attention, per pooling (recorded in explainability.json)
_READOUT = {
    "last": "the last position",
    "mean": "every position equally",
    "attention": "the head's learned pooling weights",
    "max": "every position equally (max pooling has no single weight per position)",
}
_ATTENTION_NOTE = ("Attention shows where the information the head reads comes from, not how much "
                   "it changes the output; after layer 0 every position already mixes in the events "
                   "before it. Integrated Gradients is the attribution; attention complements it.")


def old_individual_files(expl_dir) -> list[Path]:
    """The files of OLD_INDIVIDUAL_FILES in expl_dir."""
    return sorted(p for pattern in OLD_INDIVIDUAL_FILES for p in Path(expl_dir).glob(pattern))


# ============================================================================ #
# Attribution and attention of one prediction
# ============================================================================ #
def integrated_gradients_signed(model, batch: dict[str, torch.Tensor], n_steps: int = 50,
                                target_index: int | None = None, batch_size: int = 64) -> np.ndarray:
    """Signed Integrated Gradients per input token for a single-sample batch.

    Stage 03's integrated_gradients (zero-vector token baseline, midpoint rule, path points
    run batch_size at a time, gradients w.r.t. the path embeddings only) attributing the
    head's scalar: interpolated embeddings go through the backbone (its forward with
    input_embeds), pooling, and the task head. Multiclass attributes the logit of
    `target_index`, the predicted class when not given. The window is first trimmed to its
    real tokens, exact as in stage 03: it is right-padded, attention is causal and pooling
    reads only real positions. Returns a length-`seq_len` signed attribution.
    """
    model.eval()
    device = next(model.parameters()).device
    seq_len = int(batch["attention_mask"][0].sum().item())
    sample = {k: v[:1, :seq_len].to(device) for k, v in batch.items() if k != "labels"}

    if model.task == "multiclass" and target_index is None:
        with torch.no_grad():
            target_index = int(model(sample)["logits"].argmax(dim=-1)[0].item())

    def head_output(backbone, rows, embeds, k):
        """Head output k of every row of path embeddings -> [N]; the backbone runs to its
        hidden states only."""
        hidden = backbone(rows, hidden_only=True, input_embeds=embeds)["hidden_states"]
        return model.head(model._pool_hidden_states(hidden, rows["attention_mask"]))[:, k]

    k = 0 if target_index is None else target_index        # one-output heads explain output 0
    return integrated_gradients(model.backbone, sample, k, n_steps, batch_size, score=head_output)


@torch.no_grad()
def readout_attention(model, sample: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """([L, T] readout attention per layer, [T] uniform reference) of one window trimmed to its
    T real positions (sample: [1, T] tensors).

    The readout attention on position j in layer l is sum_t r_t A_l[t, j]: A_l is the layer's
    causal attention averaged over its heads, and r_t how much query position t feeds the pooled
    vector the head reads (_READOUT). Every row of A_l sums to 1 over the keys its query sees,
    so the readout attention sums to 1. The uniform reference is the same sum with every query
    spreading its attention evenly over what it sees. The backbone's attention capture is
    restored afterwards."""
    backbone = model.backbone
    was_capturing = backbone.store_attn
    backbone.enable_attention_capture(True)
    try:
        model.eval()
        hidden = backbone(sample, hidden_only=True)["hidden_states"][0]                  # [T, D]
        attn = torch.stack([m[0].mean(dim=0) for m in backbone.get_attention_maps()])    # [L, T, T]
    finally:
        backbone.enable_attention_capture(was_capturing)
    T = hidden.size(0)
    if model.pooling == "last":
        r = torch.zeros(T, device=hidden.device)
        r[-1] = 1.0
    elif model.pooling == "attention":
        r = (hidden @ model.head.pool_score).softmax(dim=0)
    else:                                                      # mean; max has no single weight
        r = torch.full((T,), 1.0 / T, device=hidden.device)
    uniform = (torch.ones(T, T, device=hidden.device).tril()
               / torch.arange(1, T + 1, device=hidden.device).unsqueeze(1))
    return (r @ attn).float().cpu().numpy(), (r @ uniform).float().cpu().numpy()


def _explain_sample(model, batch: dict[str, torch.Tensor], idx: int, *, target: int | None,
                    pid: str, anchor: float, vocab_itos: dict[int, str], front_ids: set[int],
                    n_steps: int, batch_size: int, with_attention: bool) -> list[dict]:
    """The rows of one explained prediction, one per position of its window: the token, its
    age, its signed attribution, the prediction's completeness residual, whether the position
    is in the front block (SOS/CON, sex, static) and, with with_attention, its readout attention
    per layer and the uniform reference. target is the class IG explains (multiclass), else
    None."""
    signed = integrated_gradients_signed(model, batch, n_steps=n_steps, target_index=target,
                                         batch_size=batch_size)
    # Completeness: the attributions sum to the explained output's change from the zero-vector
    # token baseline (times and types kept), up to the n_steps discretisation error recorded here
    k = target if target is not None else 0               # the head output IG explains
    rows = {key: v for key, v in batch.items() if key != "labels"}
    with torch.no_grad():
        zero = torch.zeros_like(model.backbone.tok(rows["input_ids"]))
        hidden = model.backbone(rows, hidden_only=True, input_embeds=zero)["hidden_states"]
        baseline = model.head(model._pool_hidden_states(hidden, rows["attention_mask"]))[0, k]
        change = model.head(model(batch)["pooled"])[0, k] - baseline
    residual = float(signed.sum()) - change.item()
    seq_len = len(signed)
    ids = batch["input_ids"][0, :seq_len].cpu().numpy()
    ages = batch["time_age"][0, :seq_len].cpu().numpy() if "time_age" in batch else np.full(seq_len, np.nan)
    if with_attention:
        attention, uniform = readout_attention(
            model, {key: v[:1, :seq_len] for key, v in batch.items() if key != "labels"})

    rows = []
    for pos in range(seq_len):
        row = {
            "patient_id": pid,
            "sample_index": idx,
            "position": pos,
            "token": vocab_itos.get(int(ids[pos]), f"tok_{ids[pos]}"),
            "age": float(ages[pos]),
            "attribution": float(signed[pos]),
            "abs_attribution": float(abs(signed[pos])),
            "completeness_residual": residual,
        }
        if target is not None:
            row["target_class"] = target
        row["anchor_age"] = anchor  # then the newer columns, so every earlier one keeps its position
        row["front_block"] = int(ids[pos]) in front_ids
        if with_attention:
            for layer in range(len(attention)):
                row[f"attention_layer{layer}"] = float(attention[layer, pos])
            row["attention_uniform"] = float(uniform[pos])
        rows.append(row)
    return rows


# ============================================================================ #
# The explainability run (pipeline 06)
# ============================================================================ #
def explain_outcome_predictions(
    model,
    dataset,
    vocab_itos: dict[int, str],
    task: str,
    output_dir: str,
    collate_fn,
    *,
    vocab_stats: dict,
    num_samples: int | None = None,
    top_k_tokens: int = 25,
    n_steps: int = 50,
    batch_size: int = 64,
    device: str = "auto",
    plot_attention: bool = True,
    min_patients: int = MIN_PATIENTS,
    evaluated_model: dict | None = None,
) -> pd.DataFrame:
    """Integrated Gradients (and, with plot_attention, readout attention) for the predictions
    of dataset: all of them (num_samples None), or num_samples drawn at random (seed SEED).

    Writes the per-position rows to <output_dir>/token_attributions.parquet (patient level) and
    their cohort summaries to <output_dir>/explainability/ (summarise_explanations: aggregates
    of at least min_patients patients, top_k_tokens codes per figure, explainability.json with
    the settings and evaluated_model, the record of the weights explained). Files of earlier
    versions in explainability/ (OLD_INDIVIDUAL_FILES) are deleted once the new parquet is
    written. vocab_stats (load_vocab's) names the structural and front-block tokens. Returns the
    rows."""
    device = torch.device(resolve_device(device))
    model = model.to(device).eval()
    expl_dir = Path(output_dir) / "explainability"

    special = vocab_stats["special_tokens"]
    structural = set(special)
    front_ids = {special[t] for t in ("<SOS>", "<CON>") if t in special}
    front_ids |= set(vocab_stats.get("sex_token_ids", [])) | set(vocab_stats.get("static_token_ids", []))
    patient_ids = getattr(dataset, "patient_ids", None)
    # Each sample's own anchor (a patient may have several): with patient_id, the key of its
    # row in predictions.parquet
    anchor_ages = getattr(dataset, "anchor_ages", None)

    n = len(dataset)
    chosen = (np.arange(n) if num_samples is None or num_samples >= n
              else np.sort(np.random.default_rng(SEED).choice(n, num_samples, replace=False)))
    print(f"[explain] Integrated Gradients on {len(chosen)} of {n} test predictions (n_steps={n_steps})...")
    rows = []
    for idx in ProgressLog(chosen, "[explain]"):
        idx = int(idx)
        batch = batch_to_device(collate_fn([dataset[idx]]), device)
        target = None
        if task == "multiclass":           # each prediction is explained for its predicted class
            with torch.no_grad():
                target = int(model(batch)["logits"].argmax(dim=-1)[0].item())
        pid = str(patient_ids[idx]) if patient_ids is not None else str(idx)
        anchor = float(anchor_ages[idx]) if anchor_ages is not None else float("nan")
        rows.extend(_explain_sample(model, batch, idx, target=target, pid=pid, anchor=anchor,
                                    vocab_itos=vocab_itos, front_ids=front_ids, n_steps=n_steps,
                                    batch_size=batch_size, with_attention=plot_attention))

    attributions = pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    attributions.to_parquet(Path(output_dir) / "token_attributions.parquet", index=False)
    for old in old_individual_files(expl_dir):
        old.unlink()
        print(f"[explain] deleted {old.name}, an individual-level file of an earlier run")

    worst = float(attributions["completeness_residual"].abs().max()) if rows else 0.0
    test_patients = len(set(map(str, patient_ids))) if patient_ids is not None else n
    explained_patients = attributions["patient_id"].nunique() if rows else 0
    backbone_config = model.backbone.config
    meta = {
        "task": task,
        "explained_output": _unit(task, None) if task != "multiclass" else "each prediction's own class logit",
        "selection": ("all test predictions" if len(chosen) == n
                      else f"{len(chosen)} test predictions drawn at random, seed {SEED}"),
        # 5-patient rule: a count of predictions from fewer than min_patients patients is hidden
        "n_test_predictions": n if test_patients >= min_patients else HIDDEN,
        "n_explained_predictions": len(chosen) if explained_patients >= min_patients else HIDDEN,
        "n_explained_patients": explained_patients if explained_patients >= min_patients else HIDDEN,
        "n_steps": n_steps,
        "max_abs_completeness_residual": worst,
        "structural_tokens": sorted(structural),
        "structural_share_of_abs_attribution": (
            float(attributions.loc[attributions["token"].isin(structural), "abs_attribution"].sum()
                  / attributions["abs_attribution"].sum()) if rows else None),
        "pooling": model.pooling,
        "evaluated_model": evaluated_model,
    }
    if plot_attention:
        meta.update({"readout_attention_queries": _READOUT[model.pooling],
                     "n_layers": backbone_config.n_layers, "n_heads": backbone_config.n_heads,
                     "attention_note": _ATTENTION_NOTE})
    summarise_explanations(attributions, task=task, out_dir=expl_dir, structural=structural,
                           meta=meta, min_patients=min_patients, top_k=top_k_tokens)

    print(f"[explain] IG completeness: largest |residual| {worst:.2e} (raise n_steps if not small)")
    print(f"[explain] Wrote {Path(output_dir) / 'token_attributions.parquet'} and the summaries in {expl_dir}")
    return attributions
