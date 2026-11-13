# src/tracehr/dataset/meds_reader.py
"""The MEDS input through DuckDB: the single event source of every reader.

The input is a MEDS-style event table (a parquet file, or a folder of parquet shards): a
patient id, a token, a timestamp[us] and an optional type column. Each run checks every
patient once (load_event_times) and every reader then selects from the same event view
(event_source): the rows and patients that pass the checks, with time_age and time_abs
computed as float64 days from the timestamps (see "Event times" below). On top of it:
aggregate queries (token_type_counts, count_rows), the ordered per-patient streams
(stream_patient_sequences returns raw tokens and types, stream_encoded_patient_sequences
encodes them in SQL) and a random-access reader (PatientEventReader). Every path sorts a
patient's events by canonical_order, so streamed windows are byte-identical to the
window-store build. Every DuckDB connection is opened by connect(): the run's memory limit,
the job's CPUs and a spill folder of its own.
"""

import glob
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from torch.utils.data import get_worker_info

# time_abs counts days from this moment; births before it are dropped, so no event has a
# negative time_abs. A new vocabulary records this value, and every later run uses the
# recorded one (with a notice when this constant differs).
EARLIEST_TIMESTAMP = datetime(1900, 1, 1)

# Names under which event_source computes the two times (time_age, time_abs) from the
# timestamps; they are never read from the input. resolve_columns returns them as age_col /
# abstime_col, the names every reader and pipeline is wired with. The package prefix keeps
# them from colliding with an input column, and resolve_columns refuses a configured column
# with either name (ignoring case, as DuckDB does).
AGE_COL, ABS_COL = "tracehr_time_age", "tracehr_time_abs"

US_PER_DAY = 86_400_000_000

_WARNED = set()


# ============================================================================ #
# Input files
# ============================================================================ #
def parquet_files(parquet_path) -> list:
    """The parquet files a path stands for: the file itself, or every .parquet file under a
    folder (a MEDS data/ folder of shards, in any folder structure), hidden files excluded.
    The one list every reader reads. A MEDS dataset root (a folder holding metadata/) is
    refused: its data are the shards under data/."""
    path = str(parquet_path)
    if not os.path.isdir(path):
        return [path]
    if os.path.isdir(os.path.join(path, "metadata")):
        raise ValueError(f"{path} holds a metadata/ folder: it is a MEDS dataset root, not its data. "
                         f"Point data.parquet_path at {os.path.join(path, 'data')}.")
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise ValueError(f"No .parquet file under {path}.")
    return files


def _duck_source(parquet_path) -> str:
    """The SQL table expression reading the input: read_parquet with safely quoted literal
    paths (robust across duckdb versions). A folder is read as one table, exactly the files
    of parquet_files; union_by_name because shards may carry different extra columns."""
    path = str(parquet_path)
    if not os.path.isdir(path):
        return "read_parquet('" + path.replace("'", "''") + "')"
    files = ", ".join("'" + f.replace("'", "''") + "'" for f in parquet_files(path))
    return f"read_parquet([{files}], union_by_name=true)"


def _duck_type_class(type_name: str):
    """Bucket a DuckDB column type into 'numeric', 'string', or None (undetermined).
    Two ids only join by exact equality if they share a class."""
    t = str(type_name).upper()
    if any(k in t for k in ("CHAR", "TEXT", "STRING", "UUID")):
        return "string"
    if any(k in t for k in ("INT", "DECIMAL", "DOUBLE", "FLOAT", "REAL", "NUMERIC")):
        return "numeric"
    return None


def parquet_id_class(parquet_path, id_col):
    """Classify a parquet id column as 'numeric', 'string', or None by reading the
    schema (DESCRIBE is metadata-only, no row scan). Lets a Python-dict join give the
    same loud type check DuckDB's own id filter already raises on."""
    con = connect()
    try:
        desc = con.execute(
            f'DESCRIBE SELECT "{id_col}" FROM {_duck_source(parquet_path)}').fetchall()
    finally:
        con.close()
    return _duck_type_class(desc[0][1]) if desc else None


def warn_precluster(name: str) -> None:
    """One-time hint that the parquet should be clustered by patient id."""
    if name in _WARNED:
        return
    _WARNED.add(name)
    print(f"[meds-reader] For fast reads, cluster {name} by patient id "
          f"(ORDER BY id, time, as MEDS shards are). Unsorted is correct but slower.")


# ============================================================================ #
# SQL helpers
# ============================================================================ #
def _sql_str(value: str) -> str:
    """value as an SQL string literal (single quotes doubled)."""
    return "'" + str(value).replace("'", "''") + "'"


def sql_in(expr: str, values) -> str:
    """`expr IN (...)` over literal strings; FALSE for an empty list (SQL has no empty IN)."""
    return f"{expr} IN ({', '.join(_sql_str(v) for v in values)})" if values else "FALSE"


def normalized_sql(col_expr: str) -> str:
    """The SQL twin of normalize_tokens / normalize_type_value (vocabulary.py): str -> strip
    -> lower. DuckDB's bare trim() only strips spaces, so the full ASCII whitespace set is
    stripped with a regex, to match Python's str.strip()."""
    return f"lower(regexp_replace(CAST({col_expr} AS VARCHAR), '^\\s+|\\s+$', '', 'g'))"


def register_ids(con, ids, id_col) -> str:
    """Register the id filter as an Arrow table; return the WHERE clause (or '')."""
    if ids is None:
        return ""
    con.register("_ids", pa.table({id_col: pa.array(list(ids))}))
    return f' WHERE "{id_col}" IN (SELECT "{id_col}" FROM _ids)'


def _register_lookup(con, name: str, mapping: dict, key_col: str, val_col: str) -> None:
    """Register a {key: int} map (the vocabulary or the type map) as an Arrow table for a
    SQL join. Keys are registered verbatim, so specials like "<PAD>" never match the
    lowercased data side - exactly as normalize_tokens("<PAD>") -> "<pad>" misses stoi in
    the Python encoder - and both paths fall through to the same default."""
    con.register(name, pa.table({
        key_col: pa.array([str(k) for k in mapping], type=pa.string()),
        val_col: pa.array([int(v) for v in mapping.values()], type=pa.int32()),
    }))


def _select_cols(id_col, tok_col, time_age_col, time_abs_col, type_col):
    """The columns a per-patient read selects: id, token, time_age, then time_abs and type
    when used."""
    cols = [id_col, tok_col, time_age_col]
    if time_abs_col is not None:
        cols.append(time_abs_col)
    if type_col is not None:
        cols.append(type_col)
    return cols


# ============================================================================ #
# Event times
# ============================================================================ #
# Each pipeline run checks every patient's rows once (load_event_times) and every reader then
# selects FROM event_source: the input without the rows and patients that failed the checks,
# with time_age and time_abs computed from the timestamps. The day formula exists once in SQL
# (_days_sql) and once in Python (days_between, for anchors), both on whole microseconds, so an
# event and an anchor at the same moment get bit-identical time_age.

def to_us(moment: datetime) -> int:
    """Whole microseconds since 1970-01-01 (the unit of a timestamp[us] column)."""
    return (moment - datetime(1970, 1, 1)) // timedelta(microseconds=1)


def days_between(us, origin_us) -> np.ndarray:
    """Days from origin_us to us (whole microseconds since 1970) as float64.

    The integer difference is exact and one float64 division rounds it once; _days_sql
    does the same two steps in DuckDB, so both give bit-identical days.
    """
    return (np.asarray(us, dtype=np.int64) - np.asarray(origin_us, dtype=np.int64)) / float(US_PER_DAY)


def _days_sql(us_expr: str, origin_us_expr: str) -> str:
    """SQL twin of days_between: the exact BIGINT difference, then one DOUBLE division."""
    return f"(CAST(({us_expr}) - ({origin_us_expr}) AS DOUBLE) / CAST({US_PER_DAY} AS DOUBLE))"


@dataclass(frozen=True, eq=False)
class EventTimes:
    """A run's checked time basis: built once by load_event_times, passed to every reader.

    Readers compute time_age (days since the patient's birth) and time_abs (days since the
    origin) from timestamp_col, and join births - one row per kept patient (id_col,
    birth_us) - which leaves out every dropped patient. The sex and static tokens take the
    birth moment; birth rows are the clock origin only and never reach a reader.
    """
    id_col: str
    timestamp_col: str
    births: pa.Table                    # one row per kept patient: id_col, birth_us
    birth_token: str                    # normalized, as the vocabulary records it
    preamble_tokens: tuple[str, ...]    # the vocabulary's sex and static tokens, normalized
    origin_us: int                      # the vocabulary's EARLIEST_TIMESTAMP, as to_us()
    file_paths: tuple[str, ...]         # parquet_files of the input; births' file_index points here

    @cached_property
    def _birth_index(self):
        """(sorted patient ids, their rows in births), built once for birth_row's search."""
        ids = self.births.column(self.id_col).to_numpy(zero_copy_only=False)
        order = np.argsort(ids, kind="stable")
        return ids[order], order

    def birth_row(self, pid) -> pa.Table:
        """One patient's births row (no row when the patient was dropped)."""
        ids, order = self._birth_index
        i = int(np.searchsorted(ids, pid))
        if i < len(ids) and ids[i] == pid:
            return self.births.slice(int(order[i]), 1)
        return self.births.slice(0, 0)

    def births_by_file(self, ids=None):
        """(file path, births of these patients in that file) for each input file holding one
        of them, in file order; ids None means every kept patient. A MEDS file holds each of its
        patients whole (check_patients makes sure), so a reader can sort one file at a time."""
        births = self.births
        if ids is not None:
            have = births.column(self.id_col).to_numpy(zero_copy_only=False)
            births = births.filter(pa.array(np.isin(have, np.asarray(list(ids)))))
        index = births.column("file_index").to_numpy()
        order = np.argsort(index, kind="stable")
        bounds = np.searchsorted(index[order], np.arange(len(self.file_paths) + 1))
        for i, path in enumerate(self.file_paths):
            if bounds[i] < bounds[i + 1]:
                yield path, births.take(pa.array(order[bounds[i]:bounds[i + 1]]))

    def file_of(self, pid) -> str:
        """The input file holding this patient (the first file for a patient not kept, whose
        read is then empty)."""
        row = self.birth_row(pid)
        return self.file_paths[row.column("file_index")[0].as_py() if row.num_rows else 0]


def check_patients(parquet_path, cols, birth, sex, static, origin_us):
    """The checks behind load_event_times, without its report: which patients the readers
    keep. birth, sex and static are the normalized birth, sex and static token names,
    origin_us the time origin (to_us). Raises when a patient has rows in more than one
    input file (_patient_files). Pipeline 01's vocabulary build calls it with the
    config's values, before any vocabulary exists; load_event_times with the vocabulary's.

    Returns (event_times, per_patient, counts). per_patient has one row per patient with an
    id and a token: pid, birth_us, n_rows, n_birth, n_untimed, n_sex, n_static, the drop
    flags r0..r4 (no birth row; birth rows without a timestamp; birth rows at different
    timestamps; a birth before the origin; a real event before the birth) and kept.
    counts = (rows, rows without an id, rows without a token) of the whole input.
    event_times joins the kept patients only. Two queries over the input, three for a folder.
    """
    id_col, tok_col, ts_col = cols["id_col"], cols["tok_col"], cols["timestamp_col"]
    source = _duck_source(parquet_path)
    tok = f'"{tok_col}"'
    is_birth = f"norm = {_sql_str(birth)}"
    is_sex, is_static = sql_in("norm", sex), sql_in("norm", static)
    real = f"NOT ({is_birth}) AND NOT ({is_sex}) AND NOT ({is_static})"
    con = connect()
    try:
        patient_files = _patient_files(con, source, id_col, parquet_path)
        counts = con.execute(
            f'SELECT COUNT(*), COUNT(*) FILTER (WHERE "{id_col}" IS NULL), '
            f'COUNT(*) FILTER (WHERE "{id_col}" IS NOT NULL AND {tok} IS NULL) FROM {source}').fetchone()
        per_patient = con.execute(
            f'SELECT pid, epoch_us(b_min) AS birth_us, n_rows, n_birth, n_untimed, n_sex, n_static, '
            f'n_birth = 0 AS r0, n_birth > 0 AND n_birth_ts = 0 AS r1, '
            f'COALESCE(b_min <> b_max, false) AS r2, COALESCE(epoch_us(b_min) < {origin_us}, false) AS r3, '
            f'COALESCE(t_min < b_min, false) AS r4 FROM ('
            f'SELECT pid, COUNT(*) AS n_rows, COUNT(*) FILTER (WHERE {is_birth}) AS n_birth, '
            f'COUNT(ts) FILTER (WHERE {is_birth}) AS n_birth_ts, '
            f'MIN(ts) FILTER (WHERE {is_birth}) AS b_min, MAX(ts) FILTER (WHERE {is_birth}) AS b_max, '
            f'MIN(ts) FILTER (WHERE {real}) AS t_min, '
            f'COUNT(*) FILTER (WHERE ts IS NULL AND {real}) AS n_untimed, '
            f'COUNT(DISTINCT norm) FILTER (WHERE {is_sex}) AS n_sex, '
            f'COUNT(DISTINCT norm) FILTER (WHERE {is_static}) AS n_static '
            f'FROM (SELECT "{id_col}" AS pid, "{ts_col}" AS ts, {normalized_sql(tok)} AS norm FROM {source} '
            f'WHERE "{id_col}" IS NOT NULL AND {tok} IS NOT NULL) GROUP BY pid)'
        ).to_arrow_table()
    finally:
        con.close()

    flags = [per_patient[f"r{i}"].to_numpy() for i in range(5)]
    is_bad = np.logical_or.reduce(flags) if per_patient.num_rows else np.zeros(0, dtype=bool)
    per_patient = per_patient.append_column("kept", pa.array(~is_bad))
    kept = per_patient.filter(pa.array(~is_bad))
    if patient_files is None:
        file_index = pa.array(np.zeros(kept.num_rows, dtype=np.int64))
    else:
        file_index = patient_files["file_index"].take(
            pc.index_in(kept["pid"], value_set=patient_files["pid"]))
    event_times = EventTimes(id_col=id_col, timestamp_col=ts_col,
                             births=pa.table({id_col: kept["pid"], "birth_us": kept["birth_us"],
                                              "file_index": file_index}),
                             birth_token=birth, preamble_tokens=tuple(sorted(set(sex) | set(static))),
                             origin_us=origin_us, file_paths=tuple(parquet_files(parquet_path)))
    return event_times, per_patient, counts


def _patient_files(con, source, id_col, parquet_path) -> pa.Table | None:
    """Each patient id's input file, as (pid, file_index): file_index is DuckDB's, a row's
    position in parquet_files. None for a single file. Every row with an id counts.

    Raises when a patient id has rows in more than one file: MEDS keeps each subject in one
    file, and the readers sort one file at a time, so a split patient would silently lose rows.
    A data column named file_index would hide DuckDB's own, so it is refused first. A query of
    its own: in check_patients' per-patient query the two columns raised the peak memory.
    """
    if len(parquet_files(parquet_path)) == 1:
        return None
    names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
    if "file_index" in names:
        raise ValueError(
            f"The input {parquet_path} has a column named file_index. tracehr checks that every "
            f"patient is in one file with DuckDB's built-in file_index (the number of the file a row "
            f"comes from), and a data column of that name hides it: the check would read your column "
            f"instead and could miss a patient split across files. Rename or drop that column, then "
            f"run again.")
    files = con.execute(
        f'SELECT "{id_col}" AS pid, MIN(file_index) AS file_index, MAX(file_index) AS last_file '
        f'FROM {source} WHERE "{id_col}" IS NOT NULL GROUP BY 1').to_arrow_table()
    split = files.filter(pc.not_equal(files["file_index"], files["last_file"]))
    if split.num_rows:
        raise ValueError(_split_patients_message(con, source, id_col, parquet_path,
                                                 split["pid"].to_pylist()))
    return files.select(["pid", "file_index"])


def _split_patients_message(con, source, id_col, parquet_path, pids, shown=5, files_shown=3) -> str:
    """The error for patients with rows in more than one file: how many, and for the first
    few (sorted, so the message is the same on every run) their files and rows in each."""
    pids = sorted(pids)
    files = parquet_files(parquet_path)
    root = parquet_path if os.path.isdir(str(parquet_path)) else os.path.dirname(str(parquet_path))
    con.register("_split", pa.table({"pid": pids[:shown]}))
    rows = con.execute(
        f'SELECT "{id_col}", file_index, COUNT(*) FROM {source} '
        f'WHERE "{id_col}" IN (SELECT pid FROM _split) GROUP BY 1, 2 ORDER BY 1, 2').fetchall()
    where = {}
    for pid, i, n in rows:
        rows_in = f"{n:,} row{'s' if n != 1 else ''} in {os.path.relpath(files[int(i)], root)}"
        where.setdefault(pid, []).append(rows_in)
    lines = []
    for pid in pids[:shown]:
        more = len(where[pid]) - files_shown
        lines.append(f"  patient {pid!r}: {len(where[pid])} files, " + ", ".join(where[pid][:files_shown])
                     + (f" and {more} more file{'s' if more != 1 else ''}" if more > 0 else ""))
    if len(pids) > shown:
        lines.append(f"  ... and {len(pids) - shown:,} more patient{'s' if len(pids) - shown != 1 else ''}")
    who = "1 patient has" if len(pids) == 1 else f"{len(pids):,} patients have"
    return (
        f"The input breaks the MEDS rule that all data of a patient is in one file.\n"
        f"{who} rows (patient id column {id_col!r}) in more than one parquet file under {root}:\n"
        + "\n".join(lines) + "\n"
        f"MEDS requires subject contiguity: the data of one subject must not be split across multiple "
        f"parquet files. Each patient id must therefore be in exactly one file of the whole input, "
        f"counting the train, tuning and held_out folders together.\n"
        f"tracehr relies on this rule: it reads, sorts and windows each file on its own, so a patient "
        f"whose rows are split would be read with part of their history missing, without any other "
        f"error. The run stops here, at the input check, before anything is computed.\n"
        f"To fix it, put all rows of each of these patient ids into one file (for example, by "
        f"re-sharding the data with the MEDS tools), or, if different people share an id across "
        f"files, give them distinct ids. Then run again.")


def load_event_times(parquet_path, cols, vocab_path) -> EventTimes:
    """Check every patient's rows once, report what the run uses, and return its EventTimes.

    The vocabulary (pipeline 01) names the birth token, the sex and static tokens, the front
    block's size and the time origin. Rows left out, and reported: no patient id; no token;
    a null timestamp that is not a sex or static token. Patients dropped whole, and
    reported: no birth row with a valid timestamp; birth rows at different timestamps; a
    birth before the origin; a real event (not a birth, sex or static row) before the
    birth. A birth row without a timestamp next to a valid one is ignored. The checks are
    check_patients'. Raises when no patient is left.
    """
    # Imported here: vocabulary.py builds on this module (its queries read the event view).
    from .vocabulary import load_vocab
    _, itos, stats = load_vocab(vocab_path)
    birth = stats.get("birth_token")
    if not birth or "earliest_timestamp" not in stats:
        raise ValueError(
            f"{vocab_path} records no birth token or time origin: it was built before MEDS "
            f"input. Delete it and let pipeline 01 rebuild it (sequences.birth_token).")
    sex = [itos[int(i)] for i in stats.get("sex_token_ids", [])]
    static = [itos[int(i)] for i in stats.get("static_token_ids", [])]
    origin = datetime.fromisoformat(stats["earliest_timestamp"])
    if origin != EARLIEST_TIMESTAMP:
        print(f"[event-times] NOTICE: {vocab_path} was built with EARLIEST_TIMESTAMP "
              f"{origin:%Y-%m-%d %H:%M}, not this module's {EARLIEST_TIMESTAMP:%Y-%m-%d %H:%M}; "
              f"the vocabulary's value is used. Rebuild the vocabulary to use the new one.")
    event_times, per_patient, (n_rows, no_id, no_tok) = check_patients(
        parquet_path, cols, birth, sex, static, to_us(origin))

    reasons = ("no birth row", "birth rows without a timestamp", "birth rows at different timestamps",
               f"a birth before {origin:%Y-%m-%d %H:%M}", "a real event before the birth")
    flags = [per_patient[f"r{i}"].to_numpy() for i in range(len(reasons))]
    is_bad = ~per_patient["kept"].to_numpy()
    kept = per_patient.filter(per_patient["kept"])
    pids = per_patient["pid"]
    n_untimed = int(np.sum(per_patient["n_untimed"].to_numpy()))
    left = [(no_id, "without a patient id"), (no_tok, "without a token"),
            (n_untimed, "with a null timestamp that are not sex or static tokens")]
    print(f"[event-times] input {parquet_path} ({len(parquet_files(parquet_path))} file(s)): "
          f"{n_rows:,} rows, {per_patient.num_rows:,} patients")
    if any(n for n, _ in left):
        print("  rows left out: " + ", ".join(f"{n:,} {why}" for n, why in left if n))
    if is_bad.any():
        print(f"  patients dropped: {int(is_bad.sum()):,}, with at least one of")
        no_birth = flags[0] | flags[1]
        if no_birth.any():
            print(f"    no birth row with a valid timestamp: {int(no_birth.sum()):,} "
                  f"({int(flags[0].sum()):,} without a birth row, {int(flags[1].sum()):,} whose birth "
                  f"rows have no timestamp), e.g. {pids.filter(pa.array(no_birth)).to_pylist()[:5]}")
        for why, f in list(zip(reasons, flags))[2:]:
            if f.any():
                print(f"    {why}: {int(f.sum()):,}, e.g. {pids.filter(pa.array(f)).to_pylist()[:5]}")
    print(f"  {int(np.sum(kept['n_birth'].to_numpy())):,} birth rows ({birth}) used as the clock "
          f"origin, not as tokens")
    slots, max_static = int(stats["preamble_slots"]), int(stats["max_preamble_len"]) - 2
    # Each kept patient's front block, by extract_preamble_tokens' rule: one sex token when
    # S >= 1, then the alphabetically first static tokens that fit
    n_sex, n_static = kept["n_sex"].to_numpy(), kept["n_static"].to_numpy()
    sex_kept = ((n_sex > 0) & (slots >= 1)).astype(np.int64)
    static_kept = np.minimum(n_static, np.maximum(0, np.minimum(max_static, slots - sex_kept)))
    k = sex_kept + static_kept
    counts = ", ".join(f"{v}: {c:,}" for v, c in zip(*np.unique(k, return_counts=True)))
    print(f"  front block: SOS/CON + up to {slots} sex/static token(s) (limit "
          f"{stats['max_preamble_len']} positions); tokens per patient: {counts or 'none'}")
    for n, what in (
            (int((n_sex > 1).sum()), "more than one sex token: {:,} patient(s), the first "
                                     "alphabetically is used"),
            (int(((n_sex > 0) & (slots == 0)).sum()), "a sex token but no front position (no "
                                                      "training patient had one): {:,} patient(s), not used"),
            (int((n_static > static_kept).sum()), "more static tokens than fit: {:,} patient(s) keep "
                                                  "the alphabetically first ones")):
        if n:
            print("    " + what.format(n))
    rows_kept = int(np.sum(kept["n_rows"].to_numpy() - kept["n_birth"].to_numpy() - kept["n_untimed"].to_numpy()))
    print(f"  kept: {kept.num_rows:,} patients, {rows_kept:,} rows")
    if kept.num_rows == 0:
        raise ValueError(f"No patient in {parquet_path} has usable data; see above.")
    return event_times


def event_source(con, parquet_path, event_times, id_col, tok_col, type_col, pid=None,
                 births=None) -> str:
    """What every reader selects FROM: the input joined to the births (pid narrows them to
    one patient for a per-patient fetch; births, to these rows of event_times.births), with
    time_age and time_abs computed from the timestamp under the names AGE_COL and ABS_COL,
    plus the id, token and type columns.

    Birth rows are left out (the clock origin only), and so are rows without a token and rows
    with a null timestamp that are not sex or static tokens. Sex and static tokens take the
    birth moment, whatever their own timestamp says.
    """
    if event_times is None:
        raise ValueError("Every reader needs the run's event_times (load_event_times).")
    if births is None:
        births = event_times.births if pid is None else event_times.birth_row(pid)
    con.register("_births", births)
    keep = ", ".join(f'e."{c}"' for c in (id_col, tok_col, type_col) if c is not None)
    ts_col = event_times.timestamp_col
    is_pre = sql_in("e._tracehr_norm", event_times.preamble_tokens)
    ts_us = f'CASE WHEN {is_pre} THEN b.birth_us ELSE epoch_us(e."{ts_col}") END'
    return (f'(SELECT {keep}, {_days_sql(ts_us, "b.birth_us")} AS "{AGE_COL}", '
            f'{_days_sql(ts_us, str(event_times.origin_us))} AS "{ABS_COL}" '
            f'FROM (SELECT *, {normalized_sql(chr(34) + tok_col + chr(34))} AS _tracehr_norm '
            f'FROM {_duck_source(parquet_path)}) AS e '
            f'JOIN _births AS b ON e."{id_col}" = b."{id_col}" '
            f'WHERE e."{tok_col}" IS NOT NULL AND e._tracehr_norm <> {_sql_str(event_times.birth_token)} '
            f'AND (e."{ts_col}" IS NOT NULL OR {is_pre}))')


# ============================================================================ #
# Queries over the event view
# ============================================================================ #
def token_type_counts(parquet_path, tok_col, type_col, event_times):
    """Bounded (token, type, count) aggregation over the rows the readers keep (the run's
    event_times), for per-token frequency + modal-type stats without materialising the
    rows: no birth rows, rows without a token, untimed rows that are not sex or static
    tokens, or rows of dropped patients."""
    con = connect()
    try:
        source = event_source(con, parquet_path, event_times, event_times.id_col, tok_col, type_col)
        return con.execute(
            f'SELECT "{tok_col}" AS tok, "{type_col}" AS typ, COUNT(*) AS n '
            f'FROM {source} GROUP BY "{tok_col}", "{type_col}"').fetchnumpy()
    finally:
        con.close()


def count_rows(parquet_path, event_times, id_col, tok_col, ids) -> tuple[int, int]:
    """(patients, rows) the readers would read for these patient ids."""
    con = connect()
    try:
        source = event_source(con, parquet_path, event_times, id_col, tok_col, None)
        where = register_ids(con, ids, id_col)
        return con.execute(f'SELECT COUNT(DISTINCT "{id_col}"), COUNT(*) FROM {source} AS e{where}').fetchone()
    finally:
        con.close()


# ============================================================================ #
# Per-patient readers
# ============================================================================ #
def canonical_order(age, toks, absv, types):
    """Indices sorting one patient's events by (age, encoded token, absTime, Type),
    using only the non-None columns. Shared by the window-store build and the
    streaming path so genuine ties (same age AND same encoded token id — e.g. two
    out-of-vocab codes both mapping to <UNK> on the same day) resolve identically on
    both paths, keeping the streamed windows byte-identical to the offline build.
    np.lexsort is stable and its LAST key is primary, so keys run least- to
    most-significant: Type, absTime, encoded token, age."""
    keys = []
    if types is not None:
        keys.append(types)
    if absv is not None:
        keys.append(absv)
    keys.append(toks)
    keys.append(age)
    return np.lexsort(tuple(keys))


def canonical_sort(toks, age, absv, types):
    """One patient's (toks, age, absv, types) arrays sorted by canonical_order (absv and types
    may be None). Every reader applies it to the encoded tokens, so ties resolve identically
    whichever path built the windows."""
    if len(toks) > 1:
        order = canonical_order(age, toks, absv, types)
        toks, age = toks[order], age[order]
        absv = absv[order] if absv is not None else None
        types = types[order] if types is not None else None
    return toks, age, absv, types


def _finish(parts, tok_col, time_age_col, time_abs_col, type_col):
    """Concatenate one patient's batches into raw numpy arrays (toks, age, abs, type); only
    the times are cast. Age and absTime are float64 all the way into the model: float32
    holds adult Ages and absTimes only to ~3-6 minutes, coarser than the 1-minute time
    resolution."""
    toks = np.concatenate(parts[tok_col])
    age = np.concatenate(parts[time_age_col]).astype(np.float64, copy=False)
    absv = np.concatenate(parts[time_abs_col]).astype(np.float64, copy=False) if time_abs_col else None
    typ = np.concatenate(parts[type_col]) if type_col else None
    return toks, age, absv, typ


def _iter_patient_groups(con, sql, cols, id_col, batch_rows):
    """Yield (pid, {col: list of numpy segments}) per patient from an id-ordered result.

    Only the current patient plus one Arrow batch are resident. Every caller orders by
    id, so the ids can only go forwards; the check below asserts that invariant instead
    of trusting it, because an out-of-order id means one patient was silently split into
    several sequences - a corruption no downstream stage could detect.
    """
    cur_pid, parts = None, None
    # con.sql streams; con.execute(...).to_arrow_reader copies the whole result first in DuckDB 2
    for batch in con.sql(sql).to_arrow_reader(batch_rows):
        col_np = {c: batch.column(c).to_numpy(zero_copy_only=False) for c in cols}
        pid_col = col_np[id_col]
        n = len(pid_col)
        if n == 0:
            continue
        change = np.nonzero(pid_col[1:] != pid_col[:-1])[0] + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [n]))
        for s, e in zip(starts, ends):
            gid = pid_col[s]
            seg = {c: col_np[c][s:e] for c in cols}
            if cur_pid is None:
                cur_pid, parts = gid, {c: [seg[c]] for c in cols}
            elif gid == cur_pid:
                for c in cols:
                    parts[c].append(seg[c])
            else:
                if gid < cur_pid:
                    raise ValueError(
                        f"patient id {gid!r} follows {cur_pid!r}: the query did not return "
                        f"rows ordered by {id_col!r}, so patients would be split across "
                        f"sequences. This is a bug in the query, not in the input.")
                yield cur_pid, parts
                cur_pid, parts = gid, {c: [seg[c]] for c in cols}
    if cur_pid is not None:
        yield cur_pid, parts


# Units DuckDB prints a memory setting in: current_setting('memory_limit') -> e.g. "25.5 GiB"
_MEMORY_UNITS = {"bytes": 1, "KiB": 2 ** 10, "MiB": 2 ** 20, "GiB": 2 ** 30, "TiB": 2 ** 40,
                 "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "TB": 10 ** 12}
# Every connection spills into a folder of its own under this one (see connect)
SPILL_ROOT = ".tmp_tracehr"


def _memory_bytes(text) -> int | None:
    """A DuckDB memory setting as printed ("25.5 GiB") in bytes; None in another form."""
    number, _, unit = str(text).strip().partition(" ")
    try:
        return int(float(number) * _MEMORY_UNITS[unit])
    except (KeyError, ValueError):
        return None


def _cpu_count(duckdb_threads: int) -> int:
    """The CPUs this job may use: the smallest of DuckDB's own count (which follows a cgroup CPU
    quota), the CPUs this process may run on (taskset, cpusets) and a PBS job's cores
    (PBS_NUM_PPN, NCPUS), which DuckDB cannot see."""
    counts = [duckdb_threads]
    if hasattr(os, "sched_getaffinity"):
        counts.append(len(os.sched_getaffinity(0)))
    counts += [int(os.environ[v]) for v in ("PBS_NUM_PPN", "NCPUS") if os.environ.get(v, "").isdigit()]
    return max(1, min(counts))


def connect():
    """A DuckDB connection: DuckDB's default memory limit (80% of the RAM, or of a Slurm
    allocation or container limit; on PBS, set SLURM_MEM_PER_NODE in the job script) and the
    job's CPUs, both divided equally between the DataLoader workers when opened in one, and a
    spill folder of its own, so concurrent queries never share a spill file.

    The folder is SPILL_ROOT/<random id> in the working folder: DuckDB creates it at the first
    spill and removes it when the connection closes. A killed run leaves it behind (see
    _report_settings)."""
    root = os.path.abspath(SPILL_ROOT)
    try:
        os.makedirs(root, exist_ok=True)
    except OSError:
        pass  # a read-only working folder only matters if a query spills
    con = duckdb.connect(config={"temp_directory": os.path.join(root, uuid.uuid4().hex)})
    worker = get_worker_info()
    n = worker.num_workers if worker is not None else 1
    cpus = _cpu_count(int(con.execute("SELECT current_setting('threads')").fetchone()[0]))
    con.execute(f"SET threads = {max(1, cpus // n)}")
    if n > 1:
        total = _memory_bytes(con.execute("SELECT current_setting('memory_limit')").fetchone()[0])
        if total is not None:
            con.execute(f"SET memory_limit = '{total // n // 2 ** 20}MiB'")
    if worker is None:
        _report_settings(con, root, cpus)
    return con


def _report_settings(con, root, cpus) -> None:
    """Print, once per run, the memory limit, the CPUs and the spill folder, and warn about
    spill folders already there: a killed run's (or a running job's), which nothing removes."""
    if "duckdb" in _WARNED:
        return
    _WARNED.add("duckdb")
    memory = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    print(f"[duckdb] memory limit {memory} and {cpus} CPUs, both shared equally by the "
          f"DataLoader workers; spill in {root}")
    left = [e.path for e in os.scandir(root) if e.is_dir()] if os.path.isdir(root) else []
    if left:
        size = sum(os.path.getsize(os.path.join(d, f)) for p in left for d, _, fs in os.walk(p) for f in fs)
        print(f"[duckdb] WARNING: {root} holds {len(left)} folder(s) ({size / 1e9:.1f} GB) from a killed "
              f"run or a job still running here; delete them by hand once no job runs in this folder.")


def stream_patient_sequences(parquet_path, id_col, tok_col, time_age_col,
                             time_abs_col, type_col, ids=None, batch_rows=131072,
                             event_times=None):
    """Yield (pid, toks, age, abs, typ) per patient, one input file after the other, each
    file's patients ordered by (id, age, tok). One out-of-core DuckDB sort per file; only the
    current patient plus one Arrow batch are resident. Tokens/types are raw (caller encodes).
    event_times (load_event_times) names the files (parquet_path's), leaves out the dropped
    rows and patients and computes the times."""
    cols = _select_cols(id_col, tok_col, time_age_col, time_abs_col, type_col)
    con = connect()
    try:
        select = ", ".join(f'"{c}"' for c in cols)
        order = f'"{id_col}", "{time_age_col}", "{tok_col}"'
        for path, births in event_times.births_by_file(ids):
            source = event_source(con, path, event_times, id_col, tok_col, type_col, births=births)
            sql = f"SELECT {select} FROM {source} AS e ORDER BY {order}"
            for pid, parts in _iter_patient_groups(con, sql, cols, id_col, batch_rows):
                yield (pid, *_finish(parts, tok_col, time_age_col, time_abs_col, type_col))
    finally:
        con.close()


def stream_encoded_patient_sequences(
        parquet_path, id_col, tok_col, time_age_col, time_abs_col, type_col,
        *, stoi, unk_id, type_to_id=None, anchors=None, remove_cutoff_token=True,
        ids=None, batch_rows=131072, event_times=None):
    """Yield (pid, tok_ids, age, abs, type_ids) per patient with the encoding done in SQL.

    Same contract as stream_patient_sequences (event_times included), except tokens and
    types come back as int32 ids instead of raw strings. The vocabulary and type maps are
    registered as Arrow tables and applied with a hash join, so no Python runs per event and
    no Python string object is ever materialised for the token column - the two costs that
    dominate at hundreds of millions of rows.

    anchors: optional {patient_id: cutoff_age}. Applied as a join, so truncated rows are
    dropped inside DuckDB and never cross into Python; it also filters the cohort. Kept
    events satisfy Age < cutoff, or <= when remove_cutoff_token is False - the same
    contract as StreamingAnchorsDataset._truncate.

    Rows arrive grouped by patient but NOT ordered within a patient; the caller applies
    canonical_order, exactly as the raw stream's callers already do. Patients come one input
    file after the other, in id order within a file.
    """
    con = connect()
    try:
        _register_lookup(con, "_vocab", stoi, "token", "tid")
        tok_ref = f'e."{tok_col}"'
        joins = [f'LEFT JOIN _vocab v ON {normalized_sql(tok_ref)} = v.token']
        select = [f'e."{id_col}" AS pid',
                  f'COALESCE(v.tid, {int(unk_id)})::INTEGER AS tok',
                  f'e."{time_age_col}" AS age']
        out_cols = ["pid", "tok", "age"]

        if time_abs_col is not None:
            select.append(f'e."{time_abs_col}" AS abs')
            out_cols.append("abs")
        if type_col is not None:
            _register_lookup(con, "_types", type_to_id or {}, "type", "tyid")
            type_ref = f'e."{type_col}"'
            joins.append(f'LEFT JOIN _types t ON {normalized_sql(type_ref)} = t.type')
            # 0 is reserved for missing/unknown/pad, matching encode_type_value.
            select.append("COALESCE(t.tyid, 0)::INTEGER AS typ")
            out_cols.append("typ")

        if anchors is not None:
            op = "<" if remove_cutoff_token else "<="
            con.register("_anchors", pa.table({
                id_col: pa.array(list(anchors.keys())),
                "cutoff": pa.array([float(c) for c in anchors.values()], type=pa.float64()),
            }))
            joins.append(f'JOIN _anchors a ON e."{id_col}" = a."{id_col}" '
                         f'AND e."{time_age_col}" {op} a.cutoff')

        # Only ORDER BY id: the per-patient (age, token) order is redone by canonical_order,
        # so carrying those keys through an external sort of the whole table is dead work.
        # The sort is not optional even for a parquet already sorted by id - DuckDB does not
        # preserve the scan order across the vocabulary/type/anchor hash joins above, so the
        # rows of one patient would arrive in several disjoint runs. It costs almost nothing
        # here because it sorts what survives the anchor join (the cohort), not the table.
        # One file at a time, with only the wanted patients' births (ids, else the anchored
        # ones), so a file without one of them is never read.
        for path, births in event_times.births_by_file(ids if ids is not None else anchors):
            source = event_source(con, path, event_times, id_col, tok_col, type_col, births=births)
            sql = (f'SELECT {", ".join(select)} FROM {source} AS e '
                   f'{" ".join(joins)} ORDER BY e."{id_col}"')
            for pid, parts in _iter_patient_groups(con, sql, out_cols, "pid", batch_rows):
                toks = np.concatenate(parts["tok"]).astype(np.int32, copy=False)
                age = np.concatenate(parts["age"]).astype(np.float64, copy=False)  # float64: see _finish
                absv = (np.concatenate(parts["abs"]).astype(np.float64, copy=False)
                        if time_abs_col is not None else None)
                typ = (np.concatenate(parts["typ"]).astype(np.int32, copy=False)
                       if type_col is not None else None)
                yield pid, toks, age, absv, typ
    finally:
        con.close()


class PatientEventReader:
    """Random-access per-patient fetch from the MEDS input, for map-style
    __getitem__. Opens one duckdb connection per process (reopened on PID change,
    so DataLoader workers are safe) and keeps a small LRU of the patients' unencoded
    arrays. Fast when the parquet is clustered by patient id."""

    def __init__(self, parquet_path, id_col, tok_col, time_age_col,
                 time_abs_col, type_col, cache=128, event_times=None):
        """cache is the number of patients kept (least recently used dropped first);
        event_times is the run's checked time basis (load_event_times)."""
        self.parquet_path = parquet_path
        self.id_col = id_col
        self.tok_col, self.age_col, self.abs_col, self.type_col = (
            tok_col, time_age_col, time_abs_col, type_col)
        self.cols = _select_cols(id_col, tok_col, time_age_col, time_abs_col, type_col)
        self.event_times = event_times
        self._con = None
        self._pid_owner = None
        self._cache = OrderedDict()
        self._cache_max = max(1, int(cache))

    def _conn(self):
        """This process's connection: duckdb connections are not fork-safe, so a new one is
        opened when the process changes (DataLoader workers)."""
        if self._con is None or self._pid_owner != os.getpid():
            self._con = connect()
            self._pid_owner = os.getpid()
        return self._con

    def fetch(self, pid):
        """Return (toks, age, abs, typ) for one patient, ordered by (age, tok)."""
        if pid in self._cache:
            self._cache.move_to_end(pid)
            return self._cache[pid]
        con = self._conn()
        source = event_source(con, self.event_times.file_of(pid), self.event_times, self.id_col,
                               self.tok_col, self.type_col, pid=pid)
        select = ", ".join(f'"{c}"' for c in self.cols)
        sql = (f'SELECT {select} FROM {source} '
               f'WHERE "{self.id_col}" = ? ORDER BY "{self.age_col}", "{self.tok_col}"')
        tbl = con.execute(sql, [pid]).to_arrow_table()
        parts = {c: [tbl.column(c).to_numpy(zero_copy_only=False)] for c in self.cols}
        arrays = _finish(parts, self.tok_col, self.age_col, self.abs_col, self.type_col)
        self._cache[pid] = arrays
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return arrays
