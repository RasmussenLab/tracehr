# src/tracehr/generation/token_analysis.py
"""What has the model learned about each vocabulary token?
Analyzes the token embedding table (model.tok.emb.weight — weight-tied, so these
ARE the model's token representations): nearest-neighbor table, a 2D map colored
by clinical Type and sized by patients per code, and three arm-comparable metrics
(cosine/rank geometry only).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, TextArea, VPacker
from matplotlib.patches import Polygon
from matplotlib.text import Text
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ..dataset.meds_reader import connect, event_source
from ..dataset.vocabulary import load_vocab, normalize_tokens, normalize_type_value, type_counts_per_token

# Okabe-Ito: distinguishable with every common colour-vision deficiency
TYPE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# The size scale of the map starts at MIN_PATIENTS distinct patients: a code seen in fewer is
# drawn at that smallest size, so no marker size singles out a code of one patient (_diameter)
MIN_PATIENTS = 5
PNG_DPI = 400


def _patients_per_token(parquet_path: str, tok_col: str, event_times, stoi: dict[str, int],
                        unk_id: int) -> dict[int, int]:
    """{token id: distinct patients} over the rows the readers keep (the run's event_times).
    Each raw code is mapped to its token id by the Python encoder (normalize_tokens, else
    unk_id), so a code spelled several ways counts each patient once; DuckDB then counts the
    distinct patients per id without materialising the rows."""
    con = connect()
    try:
        id_col = event_times.id_col
        source = event_source(con, parquet_path, event_times, id_col, tok_col, None)
        raw = con.execute(f'SELECT DISTINCT "{tok_col}" AS tok FROM {source}').fetchnumpy()["tok"].tolist()
        con.register("_tok_ids", pa.table({
            "tok": pa.array(raw),
            "tid": pa.array([stoi.get(normalize_tokens(t), unk_id) for t in raw], type=pa.int32())}))
        agg = con.execute(f'SELECT l.tid AS tid, COUNT(DISTINCT e."{id_col}") AS n FROM {source} AS e '
                          f'JOIN _tok_ids AS l ON e."{tok_col}" = l.tok GROUP BY l.tid').fetchnumpy()
    finally:
        con.close()
    return dict(zip(agg["tid"].tolist(), agg["n"].tolist()))


def _diameter(n) -> np.ndarray:
    """Marker diameter (points) of a code seen in n patients: 3 pt at MIN_PATIENTS or fewer and
    3.5 pt more per tenfold, one fixed scale, so the maps of different models compare directly."""
    return 3.0 + 3.5 * np.log10(np.maximum(np.asarray(n, dtype=np.float64), MIN_PATIENTS) / MIN_PATIENTS)


def _size_wedge(n_max: int) -> DrawingArea:
    """The size key, read like a colour bar: a wedge whose thickness at each number of patients is
    the marker diameter of a code seen in that many (_diameter), from MIN_PATIENTS (codes seen in
    fewer share that size) to n_max, with ticks at MIN_PATIENTS x 10^k. Drawn in points, as the
    markers are."""
    n_max = max(n_max, MIN_PATIENTS)
    length, d_max = 95.0, float(_diameter(n_max))
    lo, hi = np.log10(MIN_PATIENTS), np.log10(n_max)

    def x(n):
        """The position of n patients along the wedge (points)."""
        return 3.0 + length * (np.log10(n) - lo) / max(hi - lo, 1e-9)

    centre = 12.0 + d_max / 2                           # room under the wedge for the tick labels
    half = [(x(n), float(_diameter(n)) / 2) for n in np.logspace(lo, hi, 60)]
    wedge = DrawingArea(length + 12, centre + d_max / 2 + 2, 0, 0)
    wedge.add_artist(Polygon([(xi, centre + h) for xi, h in half] + [(xi, centre - h) for xi, h in half[::-1]],
                             closed=True, facecolor="0.72", edgecolor="0.45", linewidth=0.4))
    tick = MIN_PATIENTS
    while tick <= n_max:
        wedge.add_artist(Line2D([x(tick)] * 2, [centre - d_max / 2 - 1, centre - d_max / 2 - 3.5],
                                color="0.3", linewidth=0.5))
        wedge.add_artist(Text(x(tick), 0, f"{tick:,}", ha="center", va="bottom", fontsize=6.5))
        tick *= 10
    return wedge


def plot_token_map(En, types, patients, out_stem: Path) -> None:
    """Publication figure of the token embeddings, saved as out_stem.png (PNG_DPI) and .svg: every
    code, placed by a PCA of the embeddings, colored by Type and sized by patients per code
    (_diameter: a code seen in fewer than MIN_PATIENTS patients gets the smallest size). No code is
    named on the map; the Type legend and the size wedge sit in a column beside it, inside the
    layout, so nothing is cut off. En, types and patients (distinct patients per code) are
    aligned per code."""
    print(f"[token-analysis] map: {len(patients)} codes; {int((patients < MIN_PATIENTS).sum())} seen in "
          f"fewer than {MIN_PATIENTS} patients are drawn at the {MIN_PATIENTS}-patient size")
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(En)
    var = pca.explained_variance_ratio_ * 100

    names, n_per_type = np.unique(types, return_counts=True)
    order = names[np.argsort(-n_per_type, kind="stable")]   # largest first, rare types on top
    # Okabe-Ito up to 8 types, matplotlib's 20-colour palette beyond that
    palette = TYPE_COLORS if len(order) <= len(TYPE_COLORS) else plt.get_cmap("tab20").colors
    colors = {t: palette[i % len(palette)] for i, t in enumerate(order)}
    point_style = {"alpha": 0.75, "edgecolors": "white", "linewidths": 0.4}

    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, (ax, key) = plt.subplots(1, 2, figsize=(7.0, 4.6), width_ratios=[3.3, 1], layout="constrained")
        for t in order:
            m = types == t
            ax.scatter(xy[m, 0], xy[m, 1], s=_diameter(patients[m]) ** 2, color=colors[t], **point_style)
        ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")

        # The key column: the Type legend (one marker size: size encodes patients, not Type)
        # and, under it, the size wedge
        key.set_axis_off()
        n_codes = {t: int((types == t).sum()) for t in order}
        key.legend([Line2D([], [], marker="o", linestyle="", markersize=5, color=colors[t], alpha=0.75)
                    for t in order],
                   [f"{t.capitalize()} ({n_codes[t]} code{'' if n_codes[t] == 1 else 's'})" for t in order],
                   title="Type", loc="upper left", frameon=False, alignment="left")
        size_key = AnchoredOffsetbox(
            loc="lower left", frameon=False, pad=0, borderpad=0, bbox_to_anchor=(0.0, 0.06),
            bbox_transform=key.transAxes,
            child=VPacker(children=[TextArea("Patients per code", textprops={"fontsize": 8}),
                                    _size_wedge(int(patients.max()))], align="left", pad=0, sep=5))
        size_key.set_clip_on(False)              # never clipped to the key column, so never cropped
        key.add_artist(size_key)

        for suffix, kwargs in ((".png", {"dpi": PNG_DPI}), (".svg", {})):
            fig.savefig(out_stem.parent / f"{out_stem.name}{suffix}", bbox_inches="tight", **kwargs)
        plt.close(fig)


def analyze_token_embeddings(model, vocab_path: str, parquet_path: str, output_dir,
                             top_k: int = 10, tok_col: str = "code", type_col: str = "type",
                             *, event_times) -> dict:
    """Write token_neighbors.csv, token_embeddings_2d.png (+ .svg) and token_metrics.json; return the metrics.
    event_times (load_event_times): the run's checked rows, the only ones counted. The map sizes
    each code by its distinct patients, from MIN_PATIENTS up (plot_token_map).

    Metrics (comparable across loss_mode/t2e arms):
      - type_neighbor_purity@k: mean fraction of a token's k cosine-nearest neighbors sharing its
        modal clinical Type (does embedding proximity respect clinical structure)
      - type_silhouette: silhouette score (cosine) of embeddings labeled by Type
      - norm_freq_spearman: Spearman(embedding norm, log frequency) — norms encoding frequency
        rather than semantics
    Specials and tokens never seen in the data are excluded (PAD's frozen zero row has no cosine).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stoi, itos, stats = load_vocab(vocab_path)
    E = model.tok.emb.weight.detach().cpu().float().numpy()          # [V, D]
    unk_id = int(stoi["<UNK>"])

    # Modal Type + frequency per token id: a bounded DuckDB (Token, Type) count over the
    # rows the readers keep. Types are the data's own (normalized) values, not the
    # vocabulary's type ids, which are empty for a model trained with use_type: false;
    # rows without a type count as "unknown".
    type_counts = type_counts_per_token(                 # token id -> {type name: count}
        parquet_path, tok_col, type_col, event_times, stoi, unk_id,
        type_key=lambda tp: normalize_type_value(tp) or "unknown")
    freq = {tid: sum(c.values()) for tid, c in type_counts.items()}   # token id -> total count

    special_ids = set((stats.get("special_tokens") or {}).values())
    ids = np.array(sorted(i for i in freq if 0 <= i < len(itos) and i not in special_ids))
    # Modal type per token (ties -> first name alphabetically, the order of pandas .mode())
    types = np.array([min(type_counts[i], key=lambda t: (-type_counts[i][t], t)) for i in ids])
    counts = np.array([freq[i] for i in ids])

    # Cosine similarities between analyzed tokens
    En = E[ids]
    norms = np.linalg.norm(En, axis=1)
    sims = (En / np.clip(norms, 1e-12, None)[:, None]) @ (En / np.clip(norms, 1e-12, None)[:, None]).T
    np.fill_diagonal(sims, -np.inf)
    k = min(top_k, len(ids) - 1)
    nn = np.argsort(-sims, axis=1)[:, :k]                            # [N, k] neighbor row indices

    # Metrics
    purity = float((types[nn] == types[:, None]).mean())
    silhouette = (float(silhouette_score(En, types, metric="cosine"))
                  if len(np.unique(types)) >= 2 else None)
    spearman = (float(spearmanr(norms, np.log(counts)).statistic)
                if len(ids) >= 3 and len(np.unique(counts)) > 1 else None)
    metrics = {
        "n_tokens_analyzed": int(len(ids)),
        f"type_neighbor_purity@{k}": purity,
        "type_silhouette": silhouette,
        "norm_freq_spearman": spearman,
    }

    # Qualitative outputs: per-token neighbor table + 2D map colored by Type
    rows = [{"token": itos[t], "type": types[i], "frequency": int(counts[i]),
             **{f"neighbor_{j+1}": f"{itos[ids[n]]} ({sims[i, n]:.3f})" for j, n in enumerate(nn[i])}}
            for i, t in enumerate(ids)]
    pd.DataFrame(rows).to_csv(output_dir / "token_neighbors.csv", index=False)

    patients = _patients_per_token(parquet_path, tok_col, event_times, stoi, unk_id)
    plot_token_map(En, types, np.array([patients.get(int(i), 0) for i in ids]),
                   output_dir / "token_embeddings_2d")

    with open(output_dir / "token_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[token-analysis] {len(ids)} tokens: purity@{k}={purity:.3f}, "
          f"silhouette={silhouette}, norm~freq spearman={spearman} -> {output_dir}")
    return metrics
