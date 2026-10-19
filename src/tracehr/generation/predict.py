# src/tracehr/generation/predict.py
"""Autoregressive generation from anchor-truncated histories (pipeline 03): greedy rollouts,
their scoring against the real continuation, and the generations parquet."""

from collections import Counter

import numpy as np
import pandas as pd
import torch

from ..dataset.vocabulary import encode_type_value, normalize_tokens, type_counts_per_token
from ..dataset.windows import build_window, window_capacity
from .explainability import explain_generated


def _abstain_prob(model, logits: torch.Tensor) -> torch.Tensor | None:
    """The objective's abstain mass for [..., V] logits, or None when it has no abstain
    class (plain CE). `probs` over the V tokens is then conditional on not abstaining; the
    full distribution is probs * (1 - abstain_prob)."""
    return model.token_loss.abstain_prob(logits)


def token_modal_types(parquet_path: str, stoi: dict[str, int], unk_id: int, type_to_id: dict[str, int],
                      vocab_size: int, event_times, tok_col: str = "code",
                      type_col: str = "type") -> np.ndarray:
    """[vocab_size] modal type id of each token over the rows the readers keep (event_times),
    0 for a token never seen; generated tokens inherit their usual type. Ties go to the
    smallest type id (like .mode())."""
    tc = type_counts_per_token(parquet_path, tok_col, type_col, event_times, stoi, unk_id,
                               type_key=lambda tp: encode_type_value(tp, type_to_id))
    out = np.zeros(vocab_size, dtype=np.int64)
    for tid, counts in tc.items():
        if 0 <= tid < vocab_size:
            out[tid] = min(counts, key=lambda t: (-counts[t], t))
    return out


def _preview(items: list, k: int = 10) -> str:
    """The first k items and a count of the rest, so a long pasted list stays readable in the log."""
    more = f", ... (+{len(items) - k:,} more)" if len(items) > k else ""
    return "[" + ", ".join(repr(x) for x in items[:k]) + more + "]"


def allowed_token_ids_from_names(names, stoi: dict[str, int],
                                 front_token_ids=()) -> set[int] | None:
    """Token names (as written in the input's code column) -> the ids generation may pick.

    None or [] means no restriction (returns None). Names are matched the way data tokens
    are encoded (normalize_tokens: trimmed, lowercased), so the list survives a vocabulary
    rebuild. Names missing from the vocabulary are skipped and reported, so a long external
    code list (e.g. a whole ICD chapter) works as is; it raises only if none is left. Values
    that would be reread silently also raise: a value that is not a list, a non-string entry
    (unquoted, YAML reads 250.00 as the number 250.0 and no as False), or a sex or static
    token (front_token_ids: front-block attributes the model is never trained to predict).
    """
    if names is None:
        return None
    if not isinstance(names, (list, tuple)):
        raise TypeError(f"allowed_token_names must be a list of token names, e.g. "
                        f"['e11.9', 'death'], or null; got {names!r}.")
    if not names:
        return None
    not_str = [n for n in names if not isinstance(n, str)]
    if not_str:
        raise TypeError(f"allowed_token_names entries must be strings, got {_preview(not_str)}. "
                        f"Quote them in the yaml ('250.00', 'no'): unquoted, YAML reads such "
                        f"values as numbers or booleans.")
    # One entry per distinct name (as normalized), in list order, spelled as first written
    first_spelling = {}
    for n in names:
        first_spelling.setdefault(normalize_tokens(n), n)
    distinct = list(first_spelling.items())
    missing = [n for key, n in distinct if key not in stoi]
    if len(missing) == len(distinct):
        raise ValueError(f"None of the {len(distinct):,} allowed_token_names is in the vocabulary "
                         f"{_preview(missing)}, so there is nothing to generate. Names match the "
                         f"data tokens ignoring case and surrounding spaces.")
    if missing:
        print(f"[generate] allowed_token_names: {len(missing):,} of {len(distinct):,} names are "
              f"not in the vocabulary and are skipped (a token absent from the training "
              f"vocabulary cannot be generated): {_preview(missing)}")
    known = [(key, n) for key, n in distinct if key in stoi]
    front_ids = {int(i) for i in front_token_ids}
    front = [n for key, n in known if stoi[key] in front_ids]
    if front:
        raise ValueError(f"allowed_token_names contains sex token(s) or static token(s) {front}: "
                         f"they form the front block, which the model is never trained to "
                         f"predict. Remove them.")
    return {int(stoi[key]) for key, _ in known}


# ============================================================================ #
# Generation
# ============================================================================ #
def _time_source(model) -> str | None:
    """How the rollout advances time, reported - three cases, no fabricated signals: "t2e"
    (the model's expected waiting time), "history_median" (a model that consumes time inputs
    without T2E: each patient's median pre-anchor inter-event gap) or None (a time-free
    model: the times are NaN).

    needs_*, not use_*: an Age-driven RoPE consumes time_age even with the time embeddings
    off, and a frozen age would then park every generated token on the same RoPE position,
    collapsing the whole rollout.
    """
    uses_time = model.needs_time_age or model.needs_time_abs
    if model.t2e_pred:
        return "t2e"
    if uses_time:
        print("[generate] t2e_weight=0 but the model consumes time inputs (embeddings and/or "
              "Age-driven RoPE): advancing time by each patient's median pre-anchor inter-event "
              "gap (time_source='history_median')")
        return "history_median"
    print("[generate] time-free model: rollout is token-only; time columns will be NaN")
    return None


def _prompt_states(dataset, num_tokens: int, allowed_token_ids: set[int] | None,
                   time_source: str | None) -> list[dict]:
    """One rollout state per prompt (dataset sample), from its anchor-truncated final window:
    the window's events without the front block, the front block, the start token, the full
    prompt length and the real continuation (the first num_tokens events after the prompt,
    restricted like the selection). With time_source "history_median" each state carries
    its median inter-event gap (the cohort median when it has fewer than two events)."""
    patients = []
    all_gaps = []
    for i in range(len(dataset)):
        pi, ws, we, first = (int(v) for v in dataset.windows[i])
        # Reconstruct the patient's post-truncation arrays, without the front block, from
        # the lazy dataset (fetched + encoded on the fly), then slice this window.
        toks_full, ages_full, abss_full, typs_full, front, _ = dataset._patient_arrays(
            dataset.patient_table[pi], dataset.cutoff_table[pi])
        toks = [int(t) for t in toks_full[ws:we]]
        ages = [float(a) for a in ages_full[ws:we]]
        abss = ([float(b) for b in abss_full[ws:we]] if abss_full is not None else [0.0] * len(toks))
        typs = ([int(t) for t in typs_full[ws:we]] if typs_full is not None else [0] * len(toks))
        gaps = np.diff(ages) if len(ages) >= 2 else np.array([])
        all_gaps.extend(gaps.tolist())
        # The real continuation: the events right after the prompt in the patient's full
        # record (no cutoff; a reader-cache hit with the same encoding, order and front
        # block removal), restricted like the selection and capped at num_tokens
        rec_toks, rec_ages = dataset._patient_arrays(dataset.patient_table[pi], None)[:2]
        n_prompt = len(toks_full)
        if not (np.array_equal(rec_toks[:n_prompt], toks_full) and np.array_equal(rec_ages[:n_prompt], ages_full)):
            raise ValueError(f"Patient {dataset.patient_table[pi]}: the anchor-truncated prompt is not the start "
                             f"of the full record, so the real continuation cannot be aligned.")
        future = [(int(t), float(a)) for t, a in zip(rec_toks[n_prompt:], rec_ages[n_prompt:])
                  if allowed_token_ids is None or int(t) in allowed_token_ids][:num_tokens]
        patients.append({
            "pid": dataset.patient_table[pi],
            "anchor_age": float(dataset.cutoff_table[pi]),
            "start0": int(dataset.sos_id if first else dataset.con_id),
            "front": front,
            "toks": toks, "ages": ages, "abss": abss, "typs": typs,
            "prompt_len": 1 + len(front[0]) + len(toks_full),   # full history, not just the window
            "median_gap": float(np.median(gaps)) if len(gaps) else None,
            "future": future,
        })

    if time_source == "history_median":
        if not all_gaps:
            raise ValueError("history_median time workaround needs at least one inter-event gap in the cohort.")
        cohort_median = float(np.median(all_gaps))
        for p in patients:
            if p["median_gap"] is None:
                p["median_gap"] = cohort_median
    return patients


def _selection_mask(itos: list[str], stoi: dict[str, int], allowed_token_ids: set[int] | None,
                    device) -> torch.Tensor:
    """[V] additive mask of the tokens generation may pick (0) or not (-inf): the structural
    specials are always out, and with allowed_token_ids every token outside that set too."""
    V = len(itos)
    sel_mask = torch.zeros(V)
    structural = [stoi[t] for t in ("<PAD>", "<SOS>", "<CON>", "<MASK>") if t in stoi]
    if allowed_token_ids is not None:
        sel_mask[:] = float("-inf")
        sel_mask[[t for t in allowed_token_ids if 0 <= t < V]] = 0.0
    sel_mask[structural] = float("-inf")
    # With nothing selectable every logit is -inf and argmax would silently return <PAD>
    if not torch.isfinite(sel_mask).any():
        raise ValueError(f"allowed_token_ids {sorted(allowed_token_ids)} leaves no token to "
                         f"generate: none is a vocabulary id other than <PAD>/<SOS>/<CON>/<MASK>.")
    return sel_mask.to(device)


@torch.no_grad()
def generate_tokens(
    model,
    dataset,                                   # StreamingAnchorsDataset (final_windows_only=True)
    itos: list[str],
    stoi: dict[str, int],
    num_tokens: int = 1,
    batch_size: int = 64,
    type_lookup: np.ndarray | None = None,
    allowed_token_ids: set[int] | None = None,
    explain: dict | None = None,
) -> pd.DataFrame:
    """Greedy autoregressive rollout from each anchor-truncated prompt.

    A prompt is one dataset sample - one (patient, anchor) - so a patient with several
    anchors gets one rollout per anchor; anchor_age is the sample's own anchor (NaN = the
    full history). One row per prediction step per prompt; `probs` holds the FULL vocabulary softmax
    as {token string: prob} (selection masking never alters it). true_token / true_age
    are the real event at this step of the continuation right after the prompt (filtered
    to allowed_token_ids when set; None / NaN past the end of the record). With `explain` (the
    settings from explainability.resolve_explainability) each row also gets
    attr_<method> / top_<method>: the generated token attributed to the exact window
    that produced it (see explainability.explain_generated).
    Time handling:
    t2e active -> each step advances by the model's expected waiting time (the mean of the
    grid distribution its loss scores, t2e_expected_delta);
    time-free model -> NaN times; time inputs without t2e -> the patient's median
    pre-anchor inter-event gap (cohort median fallback) as an honest workaround.
    """
    model.eval()
    device = next(model.parameters()).device

    time_source = _time_source(model)
    patients = _prompt_states(dataset, num_tokens, allowed_token_ids, time_source)
    sel_mask = _selection_mask(itos, stoi, allowed_token_ids, device)

    K = window_capacity(dataset.max_seq_len, dataset.preamble.slots)  # with a full front block
    rows = []
    for step in range(1, num_tokens + 1):
        for lo in range(0, len(patients), batch_size):
            chunk = patients[lo:lo + batch_size]
            wins = [build_window(
                        np.asarray(p["toks"][-K:], dtype=np.int32),
                        np.asarray(p["ages"][-K:], dtype=np.float64),   # times are float64
                        np.asarray(p["abss"][-K:], dtype=np.float64),
                        np.asarray(p["typs"][-K:], dtype=np.int32),
                        p["start0"] if len(p["toks"]) <= K else dataset.con_id,  # in-distribution sliding
                        p["front"], dataset.max_seq_len, dataset.pad_id)
                    for p in chunk]
            ids = torch.from_numpy(np.stack([w[0] for w in wins])).to(torch.int64).to(device)
            attn = torch.from_numpy(np.stack([w[4] for w in wins])).to(torch.int64).to(device)
            batch = {"input_ids": ids, "attention_mask": attn}
            if model.needs_time_age:
                batch["time_age"] = torch.from_numpy(np.stack([w[1] for w in wins])).to(device)
            if model.needs_time_abs:
                batch["time_abs"] = torch.from_numpy(np.stack([w[2] for w in wins])).to(device)
            if model.use_type and model.typ is not None:
                batch["type"] = torch.from_numpy(np.stack([w[3] for w in wins])).to(torch.int64).to(device)

            out = model(batch)
            last = attn.sum(dim=1) - 1                                   # [B] last real position
            bidx = torch.arange(len(chunk), device=device)
            logits = out["logits"][bidx, last, :].float()                # [B, V]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()          # raw, saved as-is
            abstain = _abstain_prob(model, logits)                       # [B] or None
            abstain = abstain.cpu().numpy() if abstain is not None else None
            picks = (logits + sel_mask).argmax(dim=-1).cpu().numpy()     # masked selection only
            deltas = (out["predicted_time_delta"][bidx, last].cpu().numpy()
                      if time_source == "t2e" else None)
            # Optional explainability: each pick attributed to the window that produced it
            attributions = (explain_generated(model, batch, last, picks, itos, explain, batch_size)
                            if explain is not None else None)

            for j, p in enumerate(chunk):
                tok = int(picks[j])
                if time_source == "t2e":
                    dt = float(deltas[j])
                elif time_source == "history_median":
                    dt = p["median_gap"]
                else:
                    dt = None
                new_age = p["ages"][-1] + dt if dt is not None else float("nan")
                new_abs = p["abss"][-1] + dt if dt is not None else float("nan")
                win_real = int(attn[j].sum())
                true_tok, true_age = p["future"][step - 1] if step <= len(p["future"]) else (None, float("nan"))
                rows.append({
                    "patient_id": str(p["pid"]),
                    "anchor_age": p["anchor_age"],
                    "step": step,
                    "position": p["prompt_len"] + step - 1,
                    "context_tokens": [itos[int(t)] for t in ids[j, :win_real].cpu().numpy()],
                    "predicted_token_id": tok,
                    "predicted_token": itos[tok],
                    "true_token": itos[true_tok] if true_tok is not None else None,
                    "probs": dict(zip(itos, probs[j].tolist())),
                    "predicted_time_delta": dt if dt is not None else float("nan"),
                    "generated_age": new_age,
                    "true_age": true_age,
                    "time_source": time_source,
                    **({"abstain_prob": float(abstain[j])} if abstain is not None else {}),
                    **(attributions[j] if attributions is not None else {}),
                })
                p["toks"].append(tok)
                p["ages"].append(new_age if dt is not None else (p["ages"][-1] if p["ages"] else 0.0))
                p["abss"].append(new_abs if dt is not None else (p["abss"][-1] if p["abss"] else 0.0))
                p["typs"].append(int(type_lookup[tok]) if type_lookup is not None else 0)

    return pd.DataFrame(rows)


# ============================================================================ #
# Scoring and saving the generations
# ============================================================================ #
def score_generations(df: pd.DataFrame, unk_token: str = "<UNK>") -> pd.DataFrame:
    """generate_tokens' rows -> the same rows scored against the real continuation.

    correct       predicted_token == true_token; <NA> when the step has no real event or
                  the real event is <UNK> (a code outside the vocabulary cannot be judged)
    cum_accuracy  per prompt (patient, anchor_age), the share of the scored steps 1..k
                  that match in order
    cum_overlap   per prompt, the share of the scored real events of steps 1..k found
                  among the generated tokens of those steps in any order (each event
                  counted as often as it occurs), so same-day events, stored in token-id
                  order, are not penalised for their order
    Both cumulative scores carry forward past the end of the real continuation and are
    NaN until a step is scored.
    """
    correct = pd.Series(pd.NA, index=df.index, dtype="boolean")
    cum_accuracy = pd.Series(np.nan, index=df.index)
    cum_overlap = pd.Series(np.nan, index=df.index)
    # One group per prompt: two anchors of a patient are two rollouts. dropna=False keeps
    # the full-history prompt (no anchor, NaN anchor_age) as a group of its own.
    for _, g in df.sort_values("step").groupby(["patient_id", "anchor_age"], sort=False,
                                               dropna=False):
        generated, real = Counter(), Counter()
        n_correct = 0
        for idx, gen_tok, true_tok in zip(g.index, g["predicted_token"], g["true_token"]):
            if not pd.isna(true_tok):                       # within the real continuation
                generated[gen_tok] += 1
                if true_tok != unk_token:
                    real[true_tok] += 1
                    correct[idx] = gen_tok == true_tok
                    n_correct += gen_tok == true_tok
            n_scored = sum(real.values())
            if n_scored:
                cum_accuracy[idx] = n_correct / n_scored
                cum_overlap[idx] = sum((generated & real).values()) / n_scored
    return df.assign(correct=correct, cum_accuracy=cum_accuracy, cum_overlap=cum_overlap)


def write_generations_parquet(df: pd.DataFrame, path) -> None:
    """Write generate_tokens' rows to path, with probs as a true map<string, float32> column
    (a plain to_parquet would infer a struct with one field per token)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    probs = pa.array(df["probs"].tolist(), type=pa.map_(pa.string(), pa.float32()))
    table = pa.Table.from_pandas(df.drop(columns=["probs"]), preserve_index=False)
    pq.write_table(table.append_column("probs", probs), path)
