"""Static figures for the write-up (PNG, light theme).

Colors follow a validated categorical order (blue, orange, aqua, yellow). With four
representations on a scatter, marker shape is a second encoding so identity never relies on
color alone. Magnitudes use a single-hue blue ramp; signed changes use a zero line, not hue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

ENCODER_ORDER = ["pixels", "resnet50", "dinov2_vitb14", "clip_vitb16"]
REDUCER_ORDER = ["none", "pca50", "umap15", "umap50"]
CLUSTERER_ORDER = ["kmeans", "gmm", "ward", "hdbscan"]


def _order(values: list[str] | np.ndarray, preferred: list[str]) -> list[str]:
    vals = list(dict.fromkeys(values))
    return [v for v in preferred if v in vals] + sorted(v for v in vals if v not in preferred)


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)


def _title(fig: plt.Figure, text: str) -> None:
    """Figure title above the panel titles (never overlapping them)."""
    fig.suptitle(text, fontsize=10, color=TEXT, x=0.01, y=1.06, ha="left")


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def variance_figure(
    var: pd.DataFrame, path: Path, title: str = "A1 · What drives clustering quality"
) -> Path:
    """A1: share of ARI variance per component, one panel per dataset."""
    order = var.groupby("component")["eta2"].mean().sort_values().index.tolist()
    datasets = sorted(var["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        d = var[var["dataset"] == dataset].set_index("component").reindex(order)
        y = np.arange(len(order))
        ax.barh(y, d["eta2"], color=SERIES[0], height=0.6)
        for yi, v in zip(y, d["eta2"], strict=True):
            if v >= 0.05:
                ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=7, color=TEXT_2)
        ax.set_yticks(y, order, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(dataset, fontsize=9, color=TEXT)
        _style(ax)
    axes[0].set_xlabel("share of ARI variance (η²)", fontsize=8)
    _title(fig, title)
    return _save(fig, path)


def heatmap_figure(cells: pd.DataFrame, path: Path) -> Path:
    """Mean ARI per configuration: rows = datasets, columns = algorithms."""
    datasets = sorted(cells["dataset"].unique())
    clusterers = _order(cells["clusterer"].unique(), CLUSTERER_ORDER)
    encoders = _order(cells["encoder"].unique(), ENCODER_ORDER)
    reducers = _order(cells["reducer"].unique(), REDUCER_ORDER)
    fig, axes = plt.subplots(
        len(datasets), len(clusterers),
        figsize=(2.3 * len(clusterers) + 1.2, 1.9 * len(datasets) + 0.6),
        squeeze=False, layout="constrained",
    )  # fmt: skip
    im = None
    for i, dataset in enumerate(datasets):
        for j, clusterer in enumerate(clusterers):
            ax = axes[i, j]
            d = cells[(cells["dataset"] == dataset) & (cells["clusterer"] == clusterer)]
            grid = d.pivot_table(index="encoder", columns="reducer", values="ari").reindex(
                index=encoders, columns=reducers
            )
            im = ax.imshow(grid.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
            for (r, c), v in np.ndenumerate(grid.to_numpy()):
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v > 0.55 else TEXT)  # fmt: skip
            ax.set_xticks(range(len(reducers)), reducers if i == len(datasets) - 1 else [],
                          fontsize=7, rotation=30)  # fmt: skip
            ax.set_yticks(range(len(encoders)), encoders if j == 0 else [], fontsize=7)
            ax.tick_params(length=0, colors=TEXT_2)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(clusterer, fontsize=9, color=TEXT)
            if j == 0:
                ax.set_ylabel(dataset, fontsize=8, color=TEXT)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.6)
        cbar.set_label("mean ARI over seeds", fontsize=8, color=TEXT)
        cbar.ax.tick_params(labelsize=7, colors=TEXT_2)
        cbar.outline.set_visible(False)
    fig.suptitle("Mean ARI per configuration", fontsize=10, color=TEXT, x=0.01, ha="left")
    return _save(fig, path)


def internal_vs_ari_figure(cells: pd.DataFrame, path: Path, metric: str = "silhouette") -> Path:
    """A2: internal metric vs ARI, one point per configuration, by representation."""
    datasets = sorted(cells["dataset"].unique())
    encoders = _order(cells["encoder"].unique(), ENCODER_ORDER)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.0 * len(datasets), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets, strict=True):
        d = cells[cells["dataset"] == dataset]
        for k, enc in enumerate(encoders):
            e = d[d["encoder"] == enc]
            ax.scatter(e[metric], e["ari"], s=22, color=SERIES[k % 4], marker=MARKERS[k % 4],
                       edgecolor="white", linewidth=0.8, label=enc, zorder=3)  # fmt: skip
        ax.set_title(dataset, fontsize=9, color=TEXT)
        ax.set_xlabel(metric, fontsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].set_ylabel("ARI", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.12), labelcolor=TEXT)  # fmt: skip
    _title(fig, f"A2 · Does {metric} track ARI? (one point per configuration)")
    return _save(fig, path)


def reducer_effect_figure(effects: pd.DataFrame, path: Path) -> Path:
    """A3: ARI change vs no reduction, per algorithm. Dots = (dataset, representation)."""
    reducers = _order(effects["reducer"].unique(), REDUCER_ORDER)
    clusterers = _order(effects["clusterer"].unique(), CLUSTERER_ORDER)
    fig, axes = plt.subplots(1, len(reducers), figsize=(3.0 * len(reducers), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, reducer in zip(axes, reducers, strict=True):
        e = effects[effects["reducer"] == reducer]
        for x, clusterer in enumerate(clusterers):
            v = e.loc[e["clusterer"] == clusterer, "delta_ari"].dropna().to_numpy()
            ax.scatter(x + rng.uniform(-0.15, 0.15, len(v)), v, s=12, color=TEXT_2, alpha=0.45,
                       linewidth=0, zorder=2)  # fmt: skip
            if len(v):
                ax.plot([x - 0.28, x + 0.28], [v.mean()] * 2, color=SERIES[0], linewidth=2.2,
                        solid_capstyle="round", zorder=3)  # fmt: skip
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_xticks(range(len(clusterers)), clusterers, fontsize=8)
        ax.set_title(f"{reducer} vs none", fontsize=9, color=TEXT)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].set_ylabel("Δ ARI (reduced − none)", fontsize=8)
    _title(
        fig, "A3 · Effect of dimensionality reduction (bar = mean, dots = dataset × representation)"
    )
    return _save(fig, path)
