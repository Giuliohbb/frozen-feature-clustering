"""``ffc analyze``: integrity check, analyses A1-A3 with sensitivity checks, figures, summary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ffc import analysis as an
from ffc import figures as fg
from ffc.config import ExperimentConfig
from ffc.results import integrity_from_frame, integrity_report, load_results

log = logging.getLogger(__name__)

CELL_METRICS = ("ari", "nmi", "acc", "coverage", *an.INTERNAL_METRICS)
A1_SUBSETS = ("all", "no pixels", "no pixels, no HDBSCAN")
A2_SUBSETS = ("all", "no HDBSCAN")


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


def _load(
    cfg: ExperimentConfig, runs_path: Path | None
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    expected = [r.run_id for r in cfg.iter_runs()]
    if runs_path is not None:
        df = pd.read_parquet(runs_path)
        if "status" not in df:
            df["status"] = "ok"
        return df, integrity_from_frame(df, expected), str(runs_path)
    return (
        load_results(cfg.paths.results_dir),
        integrity_report(cfg.paths.results_dir, expected),
        str(cfg.paths.results_dir),
    )


def _a1(runs: pd.DataFrame, out_dir: Path) -> str:
    frames = []
    for name in A1_SUBSETS:
        try:
            frames.append(an.variance_decomposition(an.subset(runs, name)).assign(subset=name))
        except ValueError as err:  # incomplete grid
            log.warning("A1 (%s) skipped: %s", name, err)
    if not frames:
        return "## A1 · skipped (incomplete grid)"
    var = pd.concat(frames, ignore_index=True)
    var.to_csv(out_dir / "a1_variance.csv", index=False)

    parts = []
    main = var[var["subset"] == "all"]
    if not main.empty:
        fg.variance_figure(main, out_dir / "fig_a1_variance.png")
        table = main.pivot(index="component", columns="dataset", values="eta2")
        table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
        parts.append(to_markdown(table.reset_index(), ".1%") + "\n\n![A1](fig_a1_variance.png)")

    core = ["encoder", "clusterer", "reducer", "seed (within cell)"]
    sens = var[var["component"].isin(core)].pivot_table(
        index=["subset", "component"], columns="dataset", values="eta2"
    )
    sens = sens.reindex([(s, c) for s in A1_SUBSETS for c in core if (s, c) in sens.index])
    if "no pixels, no HDBSCAN" in set(var["subset"]):
        fg.variance_figure(
            var[var["subset"] == "no pixels, no HDBSCAN"],
            out_dir / "fig_a1_variance_no_pixels_no_hdbscan.png",
            title="A1 · Without raw pixels and HDBSCAN",
        )
    parts.append(
        "### Sensitivity: main effects without the extreme levels\n\n"
        "Raw pixels score near 0 everywhere and HDBSCAN collapses without UMAP; each inflates "
        "its own factor's share.\n\n" + to_markdown(sens.reset_index(), ".1%")
    )
    return "## A1 · Share of ARI variance (η²)\n\n" + "\n\n".join(parts)


def _a2(cells: pd.DataFrame, out_dir: Path) -> str:
    summaries, choices = [], []
    for name in A2_SUBSETS:
        taus, regrets = an.internal_vs_external(an.subset(cells, name))
        taus.assign(subset=name).to_csv(out_dir / f"a2_kendall_{_slug(name)}.csv", index=False)
        regrets.assign(subset=name).to_csv(out_dir / f"a2_regret_{_slug(name)}.csv", index=False)
        summaries.append(an.summarize_a2(taus, regrets).assign(subset=name))
        g = regrets[regrets["scope"] == "global"]
        cols = ["dataset", "metric", "chosen", "chosen_ari", "best", "best_ari", "regret"]
        choices.append(g[cols].assign(subset=name))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "a2_summary.csv", index=False)
    choice = pd.concat(choices, ignore_index=True)
    fg.internal_vs_ari_figure(cells, out_dir / "fig_a2_silhouette_vs_ari.png")
    shown = ["subset", "dataset", "metric", "tau_global", "tau_within", "regret_global",
             "random_regret_global"]  # fmt: skip
    return (
        "## A2 · Choosing without labels\n\n"
        "Kendall τ between each internal metric (oriented so higher = better) and ARI, over all "
        "configurations (`global`) and averaged within each representation (`within`). Regret = "
        "best ARI minus the ARI of the configuration the metric ranks first; "
        "`random_regret` is the expected regret of a blind pick. `no HDBSCAN` drops the only "
        "algorithm that leaves points unassigned (internal metrics ignore its noise points).\n\n"
        + to_markdown(summary[shown])
        + "\n\n### Configuration picked by each metric (global scope)\n\n"
        + to_markdown(choice[["subset", *[c for c in choice.columns if c != "subset"]]])
        + "\n\n![A2](fig_a2_silhouette_vs_ari.png)"
    )


def _a3(runs: pd.DataFrame, cells: pd.DataFrame, out_dir: Path) -> str:
    effects = an.reducer_effect(cells)
    effects.to_csv(out_dir / "a3_reducer_effect.csv", index=False)
    fg.reducer_effect_figure(effects, out_dir / "fig_a3_reducer_effect.png")
    by_clusterer = (
        effects.groupby(["reducer", "clusterer"])["delta_ari"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    tests = pd.DataFrame(
        [an.compare_reducer_gain(effects, r) for r in sorted(effects["reducer"].unique())]
        if {"hdbscan", "kmeans"} <= set(effects["clusterer"])
        else []
    )
    tests.to_csv(out_dir / "a3_hdbscan_vs_kmeans.csv", index=False)
    coverage = an.hdbscan_coverage(runs)
    coverage.to_csv(out_dir / "a3_hdbscan_coverage.csv", index=False)
    cov_by_reducer = coverage.groupby("reducer")[["coverage", "n_clusters_found", "ari"]].mean()
    return (
        "## A3 · Effect of dimensionality reduction\n\n"
        "Δ ARI relative to no reduction, over datasets × representations.\n\n"
        + to_markdown(by_clusterer)
        + "\n\n### Does the reducer help HDBSCAN more than k-means? (paired Wilcoxon)\n\n"
        + to_markdown(tests)
        + "\n\n### HDBSCAN: fraction of points kept (coverage) by reducer\n\n"
        "Part of HDBSCAN's ARI gain under UMAP comes from labelling more points.\n\n"
        + to_markdown(cov_by_reducer.reset_index())
        + "\n\n![A3](fig_a3_reducer_effect.png)"
    )


def _slug(name: str) -> str:
    return name.lower().replace(",", "").replace(" ", "_")


def run_analysis(
    cfg: ExperimentConfig,
    out_dir: Path,
    allow_incomplete: bool = False,
    runs_path: Path | None = None,
) -> Path:
    """Write tables, figures and ``summary.md`` to ``out_dir``.

    Reads the result files in ``cfg.paths.results_dir``, or a frozen Parquet export
    (``runs_path``) such as ``docs/results/grid_runs.parquet``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, integrity, source = _load(cfg, runs_path)
    (out_dir / "integrity.json").write_text(json.dumps(integrity, indent=2))
    if not integrity["complete"] and not allow_incomplete:
        raise RuntimeError(
            f"Grid incomplete: {integrity['n_missing']} missing, {integrity['n_errors']} errors "
            f"(see {out_dir / 'integrity.json'}). Use --allow-incomplete for a provisional report."
        )

    runs = an.prepare(df)
    cells = an.cell_means(runs, CELL_METRICS)
    cells.to_csv(out_dir / "cells.csv", index=False)
    fg.heatmap_figure(cells, out_dir / "fig_ari_heatmaps.png")

    header = (
        "# Analysis report\n\n"
        f"Results: `{source}`. Integrity: {integrity['n_unique']}/"
        f"{integrity['n_expected']} runs, {integrity['n_missing']} missing, "
        f"{integrity['n_errors']} errors, {integrity['n_duplicate_records']} duplicate records, "
        f"{integrity['n_corrupt_lines']} corrupt lines."
        + ("" if integrity["complete"] else "\n\n> **Provisional: the grid is incomplete.**")
        + "\n\n![Mean ARI per configuration](fig_ari_heatmaps.png)"
    )
    sections = [header, _a1(runs, out_dir), _a2(cells, out_dir), _a3(runs, cells, out_dir)]
    summary = out_dir / "summary.md"
    summary.write_text("\n\n".join(sections) + "\n")
    return summary
