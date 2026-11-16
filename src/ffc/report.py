"""``ffc analyze``: integrity check, analyses A1-A3, figures and a Markdown summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ffc import analysis as an
from ffc import figures as fg
from ffc.config import ExperimentConfig
from ffc.results import integrity_report, load_results

log = logging.getLogger(__name__)

CELL_METRICS = ("ari", "nmi", "acc", "coverage", *an.INTERNAL_METRICS)


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal Markdown table (avoids a tabulate dependency)."""

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else format(v, floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def run_analysis(cfg: ExperimentConfig, out_dir: Path, allow_incomplete: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    integrity = integrity_report(cfg.paths.results_dir, (r.run_id for r in cfg.iter_runs()))
    (out_dir / "integrity.json").write_text(json.dumps(integrity, indent=2))
    if not integrity["complete"] and not allow_incomplete:
        raise RuntimeError(
            f"Grid incomplete: {integrity['n_missing']} missing, {integrity['n_errors']} errors "
            f"(see {out_dir / 'integrity.json'}). Use --allow-incomplete for a provisional report."
        )

    runs = an.prepare(load_results(cfg.paths.results_dir))
    cells = an.cell_means(runs, CELL_METRICS)
    cells.to_csv(out_dir / "cells.csv", index=False)

    sections: list[str] = []

    # A1
    try:
        var = an.variance_decomposition(runs)
        var.to_csv(out_dir / "a1_variance.csv", index=False)
        fg.variance_figure(var, out_dir / "fig_a1_variance.png")
        a1_table = var.pivot(index="component", columns="dataset", values="eta2")
        a1_table = a1_table.loc[a1_table.mean(axis=1).sort_values(ascending=False).index]
        sections.append(
            "## A1 · Share of ARI variance (η²)\n\n"
            + to_markdown(a1_table.reset_index(), ".1%")
            + "\n\n![A1](fig_a1_variance.png)"
        )
    except ValueError as err:  # incomplete grid
        log.warning("A1 skipped: %s", err)
        sections.append(f"## A1 · skipped\n\n{err}")

    # A2
    taus, regrets = an.internal_vs_external(cells)
    taus.to_csv(out_dir / "a2_kendall.csv", index=False)
    regrets.to_csv(out_dir / "a2_regret.csv", index=False)
    a2 = an.summarize_a2(taus, regrets)
    a2.to_csv(out_dir / "a2_summary.csv", index=False)
    fg.internal_vs_ari_figure(cells, out_dir / "fig_a2_silhouette_vs_ari.png")
    global_choice = regrets[regrets["scope"] == "global"][
        ["dataset", "metric", "chosen", "chosen_ari", "best", "best_ari", "regret"]
    ]
    sections.append(
        "## A2 · Choosing without labels\n\n"
        "Kendall τ between each internal metric (oriented so higher = better) and ARI, over all "
        "configurations (`global`) and averaged within each representation (`within`). Regret = "
        "best ARI minus the ARI of the configuration the metric ranks first; `random_regret` is "
        "the expected regret of a blind pick.\n\n"
        + to_markdown(a2)
        + "\n\n### Configuration picked by each metric (global scope)\n\n"
        + to_markdown(global_choice)
        + "\n\n![A2](fig_a2_silhouette_vs_ari.png)"
    )

    # A3
    effects = an.reducer_effect(cells)
    effects.to_csv(out_dir / "a3_reducer_effect.csv", index=False)
    fg.reducer_effect_figure(effects, out_dir / "fig_a3_reducer_effect.png")
    by_clusterer = (
        effects.groupby(["reducer", "clusterer"])["delta_ari"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    tests = [
        an.compare_reducer_gain(effects, r)
        for r in sorted(effects["reducer"].unique())
        if {"hdbscan", "kmeans"} <= set(effects["clusterer"])
    ]
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(out_dir / "a3_hdbscan_vs_kmeans.csv", index=False)
    sections.append(
        "## A3 · Effect of dimensionality reduction\n\n"
        "Δ ARI relative to no reduction, over datasets × representations.\n\n"
        + to_markdown(by_clusterer)
        + "\n\n### Does the reducer help HDBSCAN more than k-means? (paired Wilcoxon)\n\n"
        + to_markdown(tests_df)
        + "\n\n![A3](fig_a3_reducer_effect.png)"
    )

    fg.heatmap_figure(cells, out_dir / "fig_ari_heatmaps.png")
    header = (
        "# Analysis report\n\n"
        f"Results: `{cfg.paths.results_dir}`. Integrity: {integrity['n_unique']}/"
        f"{integrity['n_expected']} runs, {integrity['n_missing']} missing, "
        f"{integrity['n_errors']} errors, {integrity['n_duplicate_records']} duplicate records, "
        f"{integrity['n_corrupt_lines']} corrupt lines."
        + ("" if integrity["complete"] else "\n\n> **Provisional: the grid is incomplete.**")
        + "\n\n![Mean ARI per configuration](fig_ari_heatmaps.png)"
    )
    summary = out_dir / "summary.md"
    summary.write_text("\n\n".join([header, *sections]) + "\n")
    return summary
