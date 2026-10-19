# src/tracehr/generation/token_analysis.py
"""What has the model learned about each vocabulary token?
Analyzes the token embedding table (model.tok.emb.weight — weight-tied, so these
ARE the model's token representations): nearest-neighbor table, a 2D map colored
by clinical Type, and three arm-comparable metrics (cosine/rank geometry only).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ..dataset.vocabulary import load_vocab, normalize_type_value, type_counts_per_token

# Okabe-Ito: distinguishable with every common colour-vision deficiency
TYPE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

N_LABELED_TOKENS = 20


def plot_token_map(En, itos, ids, types, counts, out_png: Path) -> None:
    """Publication figure of the token embeddings: PCA 2D map colored by Type, point area by
    log frequency, the N_LABELED_TOKENS most frequent tokens named. Writes out_png (300 dpi)
    and the same figure as a vector .pdf next to it."""
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(En)
    var = pca.explained_variance_ratio_ * 100

    logc = np.log10(counts)
    span = max(logc.max() - logc.min(), 1e-12)

    def area(c):
        """Marker area of a token seen c times: 6 to 66 points^2, linear in log frequency."""
        return 6 + 60 * (np.log10(c) - logc.min()) / span

    names, n_per_type = np.unique(types, return_counts=True)
    order = names[np.argsort(-n_per_type, kind="stable")]   # largest first, rare types on top
    # Okabe-Ito up to 8 types, matplotlib's 20-colour palette beyond that
    palette = TYPE_COLORS if len(order) <= len(TYPE_COLORS) else plt.get_cmap("tab20").colors
    colors = {t: palette[i % len(palette)] for i, t in enumerate(order)}
    point_style = {"alpha": 0.75, "edgecolors": "white", "linewidths": 0.4}

    with plt.rc_context({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        for t in order:
            m = types == t
            ax.scatter(xy[m, 0], xy[m, 1], s=area(counts[m]), color=colors[t], **point_style)

        # Type legend from empty proxy markers (as the size key below), so every matplotlib
        # version draws them at one size: size encodes frequency, not Type
        type_handles = [ax.scatter([], [], s=30, color=colors[t], **point_style) for t in order]
        type_labels = [f"{t.capitalize()} (n={(types == t).sum()})" for t in order]
        type_legend = ax.legend(type_handles, type_labels, title="Type", loc="upper left",
                                bbox_to_anchor=(1.02, 1.0), frameon=False)
        ax.add_artist(type_legend)

        # Point-size key: three reference frequencies spanning the data
        ref = np.unique(np.round(10 ** np.linspace(logc.min(), logc.max(), 3)).astype(int))
        handles = [ax.scatter([], [], s=area(c), color="grey", alpha=0.6) for c in ref]
        ax.legend(handles, [f"{c:,}" for c in ref], title="Frequency", loc="lower left",
                  bbox_to_anchor=(1.02, 0.0), frameon=False)

        for i in np.argsort(-counts, kind="stable")[:N_LABELED_TOKENS]:
            ax.annotate(itos[ids[i]], xy[i], xytext=(3, 3), textcoords="offset points",
                        fontsize=6, path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])

        ax.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(Path(out_png).with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def analyze_token_embeddings(model, vocab_path: str, parquet_path: str, output_dir,
                             top_k: int = 10, tok_col: str = "code", type_col: str = "type",
                             *, event_times) -> dict:
    """Write token_neighbors.csv, token_embeddings_2d.png (+ .pdf) and token_metrics.json; return the metrics.
    event_times (load_event_times): the run's checked rows, the only ones counted.

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

    plot_token_map(En, itos, ids, types, counts, output_dir / "token_embeddings_2d.png")

    with open(output_dir / "token_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[token-analysis] {len(ids)} tokens: purity@{k}={purity:.3f}, "
          f"silhouette={silhouette}, norm~freq spearman={spearman} -> {output_dir}")
    return metrics
