# src/tracehr/dataset/vocabulary.py
"""The vocabulary: token and type ids, the front-block settings it records, and its
fingerprints.

Pipeline 01 builds it from the training split (build_vocab) and every later run loads it
(load_vocab). Tokens and types are normalized (stripped, lowercased) the same way in every
reader, in Python (encode_tokens, encode_types) and in SQL (meds_reader.normalized_sql).
The window stores and the checkpoints record a fingerprint of it, so data built under
another vocabulary is never mixed in.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from .cohort import load_patient_id_filter
from .meds_reader import check_patients, event_source, register_ids, to_us, token_type_counts

# The front-block and time settings a vocabulary records. Window stores and checkpoints
# fingerprint them together with the token mapping, so data built under other settings is
# never mixed in.
PREAMBLE_KEYS = ("sex_token_ids", "static_token_ids", "preamble_slots", "birth_token",
                 "max_preamble_len", "earliest_timestamp")


# ============================================================================ #
# Normalizing and encoding tokens and types
# ============================================================================ #
def normalize_tokens(token: Any) -> str:
    """A token as the vocabulary stores it: str, stripped, lowercased."""
    s = str(token)
    s = s.strip()
    s = s.lower()
    return s


def normalize_type_value(value: Any) -> str | None:
    """Normalize categorical event type values for stable ID mapping."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def encode_type_value(value: Any, type_to_id: dict[str, int]) -> int:
    """Encode a raw event type value; 0 is reserved for missing/unknown/pad."""
    key = normalize_type_value(value)
    if key is None:
        return 0
    return int(type_to_id.get(key, 0))


def encode_tokens(tok_vals, stoi: dict[str, int], unk_id: int) -> np.ndarray:
    """Raw token values -> int32 ids: normalize_tokens, else (or for None) unk_id. The rule
    every Python-side reader applies; stream_encoded_patient_sequences applies it in SQL."""
    return np.asarray([stoi.get(normalize_tokens(v), unk_id) if v is not None else unk_id
                       for v in tok_vals], dtype=np.int32)


def encode_types(type_vals, type_to_id: dict[str, int]) -> np.ndarray | None:
    """Raw type values -> int32 type ids (encode_type_value); None when there are no types."""
    if type_vals is None:
        return None
    return np.asarray([encode_type_value(v, type_to_id) for v in type_vals], dtype=np.int32)


def type_id_map(stats: dict) -> dict[str, int]:
    """The vocabulary's type -> type id map ({} when it records none), keys as str, ids as int."""
    return {str(k): int(v) for k, v in stats.get("type_to_id", {}).items()}


# ============================================================================ #
# Building and loading
# ============================================================================ #
def build_vocab(
    parquet_path: str = None,
    ids_path: str | None = None,
    ids_col: str = "subject_id",
    tok_col: str = "code",
    type_col: str = "type",
    special_tokens: dict[str, int] | None = None,
    unk_token: str = "<UNK>",
    sex_tokens: list[str] | None = None,
    output_path: str = "vocab.json",
    static_tokens: list[str] | None = None,
    birth_token: str | None = None,
    max_preamble_len: int | None = None,
    earliest_timestamp: datetime | None = None,
    timestamp_col: str | None = None,
) -> tuple[dict[str, int], list[str], dict[str, Any]]:
    """Build the vocabulary of the training split and record the run's front-block settings.

    Every listed sex and static token is in the vocabulary, whether or not the training data
    has it. Codes and types come only from the rows the readers keep (meds_reader's
    check_patients and event_source): never the birth token (the clock origin), rows
    without a token, rows with a null timestamp (timestamp_col) that are not sex or static
    tokens, or patients dropped whole. S (preamble_slots) is the most front-block tokens one
    kept training patient has: one sex token plus at most max_preamble_len - 2 static
    tokens. max_preamble_len and earliest_timestamp are recorded as given, and every later
    run uses the recorded values. Raises when no training patient is kept.
    """
    if not birth_token or max_preamble_len is None or earliest_timestamp is None or not timestamp_col:
        raise ValueError("build_vocab needs birth_token, max_preamble_len, earliest_timestamp and "
                         "timestamp_col (pipeline 01 passes sequences.birth_token, the module "
                         "constants and columns.timestamp_col).")
    birth = normalize_tokens(birth_token)
    sex = {normalize_tokens(t) for t in (sex_tokens or [])}
    static = {normalize_tokens(t) for t in (static_tokens or [])} - sex
    if birth in sex | static:
        raise ValueError(f"The birth token {birth_token!r} is also listed as a sex or static token: "
                         f"it is the clock origin only, never a token.")
    max_static = int(max_preamble_len) - 2
    if max_static < 0:
        raise ValueError("max_preamble_len counts the start token and the sex token, so it is at least 2.")

    # Default special tokens when none are given
    if special_tokens is None:
        special_tokens = {"<PAD>": 0, "<SOS>": 1, "<CON>": 2, "<UNK>": 3, "<MASK>": 4}

    # Load patient ids if provided
    ids = load_patient_id_filter(ids_path)

    # Only what the readers keep counts (meds_reader defines it, once): the rows of the
    # patients that pass the checks, without birth rows, rows without a token and rows with a
    # null timestamp that are not sex or static tokens.
    event_times, patients, _ = check_patients(
        parquet_path, {"id_col": ids_col, "tok_col": tok_col, "timestamp_col": timestamp_col},
        birth, sorted(sex), sorted(static), to_us(earliest_timestamp))
    patients = patients.filter(patients["kept"])
    if ids is not None:
        patients = patients.filter(pa.array(np.isin(patients["pid"].to_numpy(zero_copy_only=False), ids)))
    if patients.num_rows == 0:
        raise ValueError(
            f"No training patient in {parquet_path} has usable data, so no vocabulary is built: "
            f"each needs one birth row ({birth_token!r}) with a valid timestamp, at or after "
            f"{earliest_timestamp:%Y-%m-%d}, and no event before it.")
    con = duckdb.connect()
    try:
        source = event_source(con, parquet_path, event_times, ids_col, tok_col, type_col)
        where = register_ids(con, ids, ids_col)
        typ = f'e."{type_col}"' if type_col else "NULL"
        pairs = con.execute(f'SELECT DISTINCT e."{tok_col}", {typ} FROM {source} AS e{where}').fetchall()
    finally:
        con.close()

    # Unique normalized tokens, and stable categorical type values (type id 0 is reserved for
    # missing/unknown/pad, so actual types start at 1)
    uniq = {normalize_tokens(tok) for tok, _ in pairs}
    type_values = ({key for key in (normalize_type_value(typ) for _, typ in pairs) if key is not None}
                   if type_col else set())

    absent = sorted((sex | static) - uniq)
    uniq |= sex | static   # every listed sex and static token, seen in training or not

    # uniq holds normalized (lowercased) data tokens, so exclude the specials in that
    # same normalized form -- otherwise a data "<UNK>"/"<PAD>" survives as its own
    # "<unk>"/"<pad>" token instead of falling through to the UNK id at encode time.
    uniq.difference_update(normalize_tokens(s) for s in special_tokens)
    uniq.discard(normalize_tokens(unk_token))   # data's rare-token marker -> UNK at encode

    # Sort remaining tokens alphabetically
    normal_sorted = sorted(uniq)

    # Build stoi with specials first (respect their ids), then assign ascending ids to the rest
    stoi: dict[str, int] = dict(special_tokens)
    start_id = (max(stoi.values()) + 1) if stoi else 0  # Start after the highest special id
    for i, tok in enumerate(normal_sorted, start=start_id):
        stoi[tok] = i

    # Build itos as a dense list indexed by id
    max_id = max(stoi.values()) if stoi else -1
    itos = [""] * (max_id + 1)
    for tok, idx in stoi.items():
        itos[idx] = tok

    # Build stable type ids. 0 is reserved for padding/missing/unknown.
    type_to_id = {value: idx for idx, value in enumerate(sorted(type_values), start=1)}
    num_types = len(type_to_id) + 1 if type_to_id else 0

    # Sex and static tokens: exact match after the same normalization applied to data tokens.
    sex_token_ids = sorted(i for t, i in stoi.items() if t in sex)
    static_token_ids = sorted(i for t, i in stoi.items() if t in static)
    # S: the most front-block tokens one kept training patient has (one sex token, at most
    # max_static static tokens)
    has_sex = patients["n_sex"].to_numpy() > 0
    n_static = patients["n_static"].to_numpy()
    preamble_slots = int(np.max(has_sex + np.minimum(n_static, max_static)))
    n_cut = int(np.sum(n_static > max_static))

    # Build minimal stats
    stats = {
        "total_tokens": len(stoi),
        "num_special": len(special_tokens),
        "num_normal": len(stoi) - len(special_tokens),
        "special_tokens": special_tokens,
        "num_types": num_types, # Includes reserved type id 0 when types exist
        "type_to_id": type_to_id,
        "sex_token_ids": sex_token_ids,  # resolved once here; all consumers read this
        "static_token_ids": static_token_ids,
        "preamble_slots": preamble_slots,
        "birth_token": birth,
        "max_preamble_len": int(max_preamble_len),
        "earliest_timestamp": earliest_timestamp.isoformat(),
    }
    print(f"[INFO]: Vocab built with {stats['total_tokens']} tokens ({stats['num_special']} special, {stats['num_normal']} normal).")
    print(f"[INFO]: Found {len(type_to_id)} type values ({num_types} ids including pad/missing).")
    print(f"[INFO]: Front block: SOS/CON + up to {preamble_slots} sex/static token(s) "
          f"(limit {int(max_preamble_len)} positions); birth token {birth!r}.")
    if absent:
        print(f"[INFO]: Listed sex/static token(s) that no training patient has, in the vocabulary "
              f"with untrained embeddings: {absent}")
    if n_cut:
        print(f"[INFO]: {n_cut} training patient(s) have more than {max_static} static tokens and "
              f"keep the first {max_static} alphabetically.")

    # Save to json (reloadable)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos, "stats": stats}, f, ensure_ascii=False)
    return stoi, itos, stats


def load_vocab(vocab_path: str):
    """(stoi, itos, stats) of a vocabulary json written by build_vocab."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["stoi"], data["itos"], data["stats"]


def resolve_t2e_exclude_ids(
    stoi,
    special_tokens=("<PAD>", "<SOS>", "<CON>", "<MASK>"),
    extra_ids=(),
):
    """Token IDs kept out of the T2E competing-risk event set.

    Structural specials are always non-events. Callers add IDs for data-specific tokens
    that can appear as inputs but never as future events (the sex and static tokens).
    UNK deliberately remains a real event unless its ID is explicitly supplied.
    """
    excluded = {stoi[t] for t in special_tokens if t in stoi}
    excluded.update(int(i) for i in extra_ids)
    return sorted(excluded)


# ============================================================================ #
# Fingerprints
# ============================================================================ #
def json_digest(obj) -> str:
    """16-hex-char SHA-1 digest of a JSON-able object, canonicalized (sort_keys) so it depends
    only on the content, not on dict or file ordering. The one recipe behind every
    fingerprint: the window stores', the checkpoints' and the anchor stores'."""
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(canonical).hexdigest()[:16]


def vocab_fingerprint(vocab_path: str) -> str:
    """Fingerprint of what a window store depends on, for a vocab json on disk: the token ->
    id and type -> id mappings (a store holds both ids) and the front-block and time settings
    it records (PREAMBLE_KEYS). Used to detect window stores built against a different
    vocabulary or other settings."""
    stoi, _, stats = load_vocab(vocab_path)
    return json_digest({"stoi": stoi, "type_to_id": stats.get("type_to_id", {}),
                        **{k: stats.get(k) for k in PREAMBLE_KEYS}})


def vocabulary_fingerprint(stoi: dict[str, int], stats: dict, use_type: bool) -> str:
    """Fingerprint of what a model's embeddings depend on, recorded in every checkpoint: the
    token -> id mapping, the front-block and time settings the vocabulary records
    (PREAMBLE_KEYS: the sex and static ids reinserted as preamble attributes, S, the birth
    token, the two recorded constants) and, when the model embeds types, type -> id.
    vocab_fingerprint above is the window stores' fingerprint, which always includes types."""
    return json_digest({
        "stoi": stoi,
        **{k: stats.get(k) for k in PREAMBLE_KEYS},
        "type_to_id": stats.get("type_to_id", {}) if use_type else None,
    })


# ============================================================================ #
# Statistics over the event view
# ============================================================================ #
def unk_token_share(parquet_path: str, ids_path: str, id_col: str, tok_col: str,
                    stoi: dict[str, int], event_times) -> tuple[int, int]:
    """(tokens that train as <UNK>, all tokens) for the patients in ids_path, over the rows
    the readers keep (the run's event_times) and mapped exactly as the data loaders map them
    (normalize_tokens). Counts distinct values in DuckDB, so the Python work scales with the
    vocabulary, not the rows."""
    ids = load_patient_id_filter(ids_path)
    con = duckdb.connect()
    try:
        source = event_source(con, parquet_path, event_times, id_col, tok_col, None)
        counts = con.execute(f'SELECT e."{tok_col}", COUNT(*) FROM {source} AS e'
                             f'{register_ids(con, ids, id_col)} GROUP BY 1').fetchall()
    finally:
        con.close()
    unknown = sum(n for value, n in counts if normalize_tokens(value) not in stoi)
    return unknown, sum(n for _, n in counts)


def type_counts_per_token(parquet_path: str, tok_col: str, type_col: str, event_times,
                          stoi: dict[str, int], unk_id: int,
                          type_key: Callable[[Any], Any]) -> dict[int, dict[Any, int]]:
    """{token id: {type: count}} over the rows the readers keep (the run's event_times).
    Tokens map as the readers encode them (normalize_tokens, else unk_id); type_key turns a
    raw type value into the type's key (its name or its id). A bounded DuckDB aggregate
    (token_type_counts), so no row is materialised."""
    agg = token_type_counts(parquet_path, tok_col, type_col, event_times)
    counts = defaultdict(lambda: defaultdict(int))
    for tok, typ, n in zip(agg["tok"].tolist(), agg["typ"].tolist(), agg["n"].tolist()):
        counts[stoi.get(normalize_tokens(tok), unk_id)][type_key(typ)] += int(n)
    return counts
