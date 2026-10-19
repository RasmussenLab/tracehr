# src/tracehr/ntp_model/config.py
"""NTPTransformerConfig: the architecture every checkpoint records under "model_config"."""

import dataclasses
from dataclasses import dataclass


@dataclass
class NTPTransformerConfig:
    """Complete architecture spec needed to reconstruct an NTPTransformer.

    Holds exactly the architecture-defining constructor arguments (the runtime
    flags use_sdpa/store_attn are intentionally excluded — they do not affect
    the weights). Persisted in checkpoints as the single source of truth.
    """
    vocab_size: int
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 512
    use_time_age: bool = True
    use_time_abs: bool = True
    time_fourier_dims: int = 16
    use_type: bool = True
    num_types: int | None = None
    dropout: float = 0.1
    rope_base: float = 10000.0  # index-block RoPE base; the Age block's base is derived from the fixed day ladder
    rope_mode: str = "index"  # "index" (token order) | "age" (elapsed days) | "hybrid" (split head dims between both)
    rope_hybrid_split: float = 0.5  # hybrid only: fraction of head-dim rotation PAIRS given to the index block
    pad_id: int = 0
    mlp_inner_dim: int | None = None
    # The only T2E setting: 0 disables it; any finite value > 0 enables the coupled
    # competing-risks objective and is its multiplier in the combined loss.
    t2e_weight: float = 0.0
    loss_mode: str = "cross_entropy"  # token objective; one of ntp_transformer.TOKEN_LOSSES
    # No further loss fields by design: an objective's settings are module constants in its
    # own file, so loss_mode alone determines everything a reload has to rebuild. The
    # objective also owns the scorer, which is why this is architecture and not training config.
    t2e_exclude_ids: list[int] | None = None
    # Age-ladder slow end. Architecture-determining and therefore pinned here: it sets the
    # Age block's frequency base, so a later edit to the module constant must not
    # silently reinterpret a checkpoint that was trained with another value. (The fast
    # end, the T2E grid and Time2Vec's daily cycle all follow the fixed 1-minute time
    # resolution, ntp_transformer.TIME_RESOLUTION_DAYS.)
    # Appended (not inserted) so positional construction can never shift meaning.
    rope_time_max_days: float = 73000.0     # slowest Age wavelength (200 y)

    def to_dict(self) -> dict:
        """The config as a plain dict, the form checkpoints store."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NTPTransformerConfig":
        """Rebuild a config from a checkpoint's dict, refusing one that does not match.

        An unknown key or a missing vocab_size raises TypeError (loud on purpose).
        cls(**d) reports one offending key at a time and says nothing about the cause, so
        the whole diff is re-raised instead: a checkpoint written before a field was added
        or renamed fails here, and the fix is always to retrain rather than to coerce the
        old keys into the new ones (which would silently reinterpret the weights).
        """
        try:
            return cls(**d)
        except TypeError as exc:
            known = {f.name for f in dataclasses.fields(cls)}
            unknown = sorted(set(d) - known)
            missing = sorted(known - set(d))
            if not unknown and not missing:
                raise
            raise TypeError(
                f"This checkpoint's model_config does not match the current "
                f"NTPTransformerConfig: unknown key(s) {unknown or 'none'}, missing key(s) "
                f"{missing or 'none'}. It was written before a config change and cannot be "
                f"loaded; retrain the model. (Original error: {exc})") from exc
