# src/tracehr/dataset/column_mapping.py
"""The `columns:` and `anchor_columns:` config blocks: which input column holds what.

Every pipeline resolves its column names here (resolve_columns for the MEDS input,
resolve_anchor_columns for an anchors/outcomes file, enabled_col for the optional type
column). Each column must be named explicitly in the config - no silent defaults - and the
names and types are checked against every input file up front, so a typo fails fast with a
clear message.
"""


import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .cohort import DURATION_COL
from .meds_reader import ABS_COL, AGE_COL, parquet_files

# The complete key set of each block. Anything else is rejected (see _reject_unknown_keys).
_COLUMNS_KEYS = ("id_col", "tok_col", "timestamp_col", "type_col")
_ANCHOR_KEYS = ("id_col", "timestamp_col", "target_col", "censor_timestamp_col")

_COLUMNS_BLOCK = (
    'columns:\n'
    '  id_col: "subject_id"      # MEDS subject_id (int64; string ids are accepted too)\n'
    '  tok_col: "code"           # MEDS code (string)\n'
    '  timestamp_col: "time"     # MEDS time (timestamp[us]); null only for sex and static tokens\n'
    '  type_col: "type"          # optional extra column (string); null allowed only when\n'
    '                            # model.use_type is false'
)
_ANCHOR_BLOCK = (
    'anchor_columns:\n'
    '  id_col: "subject_id"\n'
    '  timestamp_col: "prediction_time"  # timestamp[us] of the anchor\n'
    '  target_col: "target"              # required only where the pipeline consumes targets\n'
    '  censor_timestamp_col: "end_time"  # survival only: timestamp[us] of the event (target_col 1)\n'
    '                                    # or of the censoring (target_col 0). Omit the key\n'
    '                                    # entirely for every other task.'
)


def _reject_unknown_keys(block: dict, accepted: tuple, name: str, expected: str) -> None:
    """Fail on any key the block does not define.

    A key this resolver does not read is either a typo or - the way this actually goes
    wrong - a whole run of settings indented one level too deep, which YAML happily
    parses into whichever block precedes them. Nothing downstream would notice: the
    misplaced keys are simply never read, and the pipeline runs on defaults or dies far
    away with a KeyError naming a key the user did set.
    """
    unknown = [k for k in block if k not in accepted]
    if unknown:
        raise ValueError(
            f"Unknown key(s) in `{name}:`: {', '.join(map(repr, sorted(unknown)))}. "
            f"Accepted keys are {', '.join(accepted)}. A block of settings indented under "
            f"the wrong section is the usual cause - check the indentation of the keys "
            f"listed above.\nExpected:\n" + expected)


def _validate_against_schema(names: dict[str, str | None], parquet_path: str,
                             block: str) -> None:
    """Fail unless every non-null configured name is a column of the parquet (a schema-only
    read, no data scan)."""
    available = ds.dataset(parquet_path, format="parquet").schema.names
    missing = {k: v for k, v in names.items() if v is not None and v not in available}
    if missing:
        listed = ", ".join(f"{block}.{k}={v!r}" for k, v in missing.items())
        raise ValueError(f"Configured column(s) not found in {parquet_path}: {listed}. "
                         f"Parquet columns are: {available}")


def _check_timestamp_column(parquet_path: str, col: str, name: str) -> None:
    """Fail unless the column is exactly timestamp[us] without a time zone (schema-only read)."""
    found = ds.dataset(parquet_path, format="parquet").schema.field(col).type
    if found != pa.timestamp("us"):
        raise ValueError(
            f"{name}={col!r} in {parquet_path} is {found}, but tracehr takes "
            f"timestamp[us] without a time zone. Convert it first, e.g. in pandas "
            f"df[{col!r}] = df[{col!r}].astype('datetime64[us]') (a time-zone-aware column "
            f"needs .dt.tz_localize(None) before that).")


def _check_input_files(parquet_path: str, cols: dict[str, str | None]) -> None:
    """Fail unless every file of the input carries the configured columns with their types:
    the MEDS ones, plus string ids (a tracehr extra). Schema-only reads of the file footers."""
    def is_string(t):
        """Whether the arrow type t is a string type (string or large_string)."""
        return pa.types.is_string(t) or pa.types.is_large_string(t)

    rules = {"id_col": ("an integer or string column", lambda t: pa.types.is_integer(t) or is_string(t)),
             "tok_col": ("a string column", is_string),
             "timestamp_col": ("timestamp[us] without a time zone", lambda t: t == pa.timestamp("us")),
             "type_col": ("a string column", is_string)}
    for path in parquet_files(parquet_path):
        schema = pq.read_schema(path)
        for key, (want, ok) in rules.items():
            col = cols[key]
            if col is None:
                continue
            if col not in schema.names:
                raise ValueError(f"columns.{key}={col!r} is not a column of {path}. Its columns "
                                 f"are: {schema.names}")
            found = schema.field(col).type
            if not ok(found):
                raise ValueError(f"columns.{key}={col!r} in {path} is {found}, but it must be {want}.")


def _check_not_computed_names(block: dict, keys: tuple[str, ...], name: str,
                              computed: tuple[str, ...], problems: list) -> None:
    """Add a problem for every configured column named like a column tracehr computes
    (computed), since the computed one takes that name. Names are compared ignoring case,
    as DuckDB matches column names."""
    reserved = {c.lower() for c in computed}
    for key in keys:
        value = block.get(key)
        if isinstance(value, str) and value.lower() in reserved:
            problems.append(f"{name}.{key} must not be named {value!r}: the computed "
                            f"{'/'.join(computed)} column(s) take that name (column names "
                            f"are compared ignoring case)")


def resolve_columns(config: dict, parquet_path: str | None = None) -> dict[str, str | None]:
    """Resolve the input's column names from the config's `columns:` block.

    id_col, tok_col and timestamp_col must be non-empty strings; the type_col key must be
    present and may be null (only valid while model.use_type is off, which enabled_col
    enforces at the call site). time_age and time_abs are computed from timestamp_col
    (load_event_times): the returned age_col / abstime_col are the names of the computed
    columns, which every reader and pipeline is wired with. When parquet_path (a file or a
    folder of shards) is given, every file is checked against the names and their types.
    """
    block = config.get("columns")
    if not isinstance(block, dict):
        raise ValueError("Config is missing the required top-level `columns:` section:\n"
                         + _COLUMNS_BLOCK)
    _reject_unknown_keys(block, _COLUMNS_KEYS, "columns", _COLUMNS_BLOCK)
    problems = []
    for key in ("id_col", "tok_col", "timestamp_col"):
        if not isinstance(block.get(key), str) or not block.get(key).strip():
            problems.append(f"columns.{key} must be a non-empty string")
    if "type_col" not in block:
        problems.append("columns.type_col must be present (string, or null when unused)")
    elif block["type_col"] is not None and (not isinstance(block["type_col"], str)
                                            or not block["type_col"].strip()):
        problems.append("columns.type_col must be a non-empty string or null")
    _check_not_computed_names(block, ("id_col", "tok_col", "timestamp_col", "type_col"), "columns",
                              (AGE_COL, ABS_COL), problems)
    if problems:
        raise ValueError("Invalid `columns:` section: " + "; ".join(problems)
                         + ".\nExpected:\n" + _COLUMNS_BLOCK)
    cols = {"id_col": block["id_col"], "tok_col": block["tok_col"], "age_col": AGE_COL,
            "abstime_col": ABS_COL, "type_col": block["type_col"],
            "timestamp_col": block["timestamp_col"]}
    if parquet_path is not None:
        _check_input_files(parquet_path, cols)
    print(f"[columns] resolved: {cols}")
    return cols


def enabled_col(cols: dict[str, str | None], key: str, enabled: bool,
                flag_name: str) -> str | None:
    """Gate an optional column by its model flag: the column name when enabled, else None."""
    if not enabled:
        return None
    if cols[key] is None:
        raise ValueError(f"{flag_name} is true but columns.{key} is null; "
                         f"name the column or turn the flag off.")
    return cols[key]


def resolve_anchor_columns(config: dict, parquet_path: str | None = None,
                           require_target: bool = False,
                           require_duration: bool = False) -> dict[str, str | None]:
    """Resolve the anchors/outcomes-parquet column names from `anchor_columns:`.

    id_col and timestamp_col (the anchor time, timestamp[us]) must be non-empty strings;
    target_col is required non-null only when the pipeline consumes targets
    (require_target=True), otherwise it may be null or absent.

    censor_timestamp_col follows target_col's looser convention: the key may be absent
    entirely, so every non-survival config stays valid untouched. It is required only
    when the caller asks (require_duration=True), which the survival task does - there
    target_col carries the 0/1 event indicator and censor_timestamp_col the time of the
    event or of the censoring (timestamp[us]).

    The returned age_col is the name load_anchors gives the anchor time_age it computes,
    and duration_col the name of the survival duration in days (None without
    censor_timestamp_col).
    """
    block = config.get("anchor_columns")
    if not isinstance(block, dict):
        raise ValueError("Config is missing the required top-level `anchor_columns:` section:\n"
                         + _ANCHOR_BLOCK)
    _reject_unknown_keys(block, _ANCHOR_KEYS, "anchor_columns", _ANCHOR_BLOCK)
    problems = []
    for key in ("id_col", "timestamp_col"):
        if not isinstance(block.get(key), str) or not block.get(key).strip():
            problems.append(f"anchor_columns.{key} must be a non-empty string")
    _check_not_computed_names(block, ("id_col", "target_col"), "anchor_columns",
                              (AGE_COL, DURATION_COL), problems)
    target = block.get("target_col")
    if target is not None and (not isinstance(target, str) or not target.strip()):
        problems.append("anchor_columns.target_col must be a non-empty string or null")
    if require_target and (not isinstance(target, str) or not target.strip()):
        problems.append("anchor_columns.target_col is required for this pipeline")
    censor = block.get("censor_timestamp_col")
    if censor is not None and (not isinstance(censor, str) or not censor.strip()):
        problems.append("anchor_columns.censor_timestamp_col must be a non-empty string or null")
    if require_duration and (not isinstance(censor, str) or not censor.strip()):
        problems.append("anchor_columns.censor_timestamp_col is required when the outcome is "
                        "survival (the time of the event or of the censoring; target_col is the "
                        "0/1 event indicator)")
    if problems:
        raise ValueError("Invalid `anchor_columns:` section: " + "; ".join(problems)
                         + ".\nExpected:\n" + _ANCHOR_BLOCK)
    censor = censor if isinstance(censor, str) else None
    acols = {"id_col": block["id_col"], "age_col": AGE_COL,
             "timestamp_col": block["timestamp_col"],
             "target_col": target if isinstance(target, str) else None,
             "censor_timestamp_col": censor,
             "duration_col": DURATION_COL if censor is not None else None}
    if parquet_path is not None:
        configured = {k: acols[k] for k in ("id_col", "timestamp_col", "target_col",
                                            "censor_timestamp_col")}
        _validate_against_schema(configured, parquet_path, "anchor_columns")
        _check_timestamp_column(parquet_path, acols["timestamp_col"], "anchor_columns.timestamp_col")
        if censor is not None:
            _check_timestamp_column(parquet_path, censor, "anchor_columns.censor_timestamp_col")
    print(f"[anchor_columns] resolved: {acols}")
    return acols
