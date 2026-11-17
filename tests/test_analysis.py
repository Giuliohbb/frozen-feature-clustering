from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ffc import analysis as an
from ffc.report import run_analysis
from ffc.results import load_results
from synthetic import write_grid


@pytest.fixture
def runs(tmp_path: Path) -> pd.DataFrame:
    write_grid(tmp_path / "res")
    return an.prepare(load_results(tmp_path / "res"))


def test_short_method_labels() -> None:
    assert an.short_method("umap(n_components=10,n_neighbors=15)") == "umap15"
    assert an.short_method("pca(n_components=50)") == "pca50"
    assert an.short_method("hdbscan(min_cluster_size=50)") == "hdbscan"
    assert an.short_method("none") == "none"


def test_variance_decomposition_pure_main_effect() -> None:
    """If ARI depends only on the encoder, eta2(encoder) is 1 and everything else 0."""
    rows = [
        {"dataset": "d", "encoder": e, "reducer": r, "clusterer": c, "seed": s, "ari": q}
        for e, q in [("a", 0.1), ("b", 0.5), ("c", 0.9)]
        for r in ("none", "pca50") for c in ("kmeans", "ward") for s in range(3)
    ]  # fmt: skip
    var = an.variance_decomposition(pd.DataFrame(rows)).set_index("component")["eta2"]
    assert var["encoder"] == pytest.approx(1.0)
    assert var.drop("encoder").abs().max() == pytest.approx(0.0, abs=1e-12)


def test_variance_decomposition_shares_sum_to_one(runs: pd.DataFrame) -> None:
    var = an.variance_decomposition(runs)
    for _, d in var.groupby("dataset"):
        assert d["eta2"].sum() == pytest.approx(1.0)
        top = d.sort_values("eta2", ascending=False)["component"].iloc[0]
        assert top == "encoder"  # the synthetic grid is built so representation dominates


def test_variance_decomposition_rejects_incomplete_grid(runs: pd.DataFrame) -> None:
    incomplete = runs[~((runs["encoder"] == "pixels") & (runs["clusterer"] == "ward"))]
    with pytest.raises(ValueError, match="empty cells"):
        an.variance_decomposition(incomplete)


def test_internal_metrics_work_within_but_not_across_representations(runs: pd.DataFrame) -> None:
    cells = an.cell_means(runs, ("ari", "silhouette", "davies_bouldin", "calinski_harabasz"))
    taus, regrets = an.internal_vs_external(cells)
    sil = taus[taus["metric"] == "silhouette"]
    within = sil[sil["scope"] != "global"]["tau"].mean()
    across = sil[sil["scope"] == "global"]["tau"].mean()
    assert within > 0.8 and within > across + 0.3
    assert (regrets["regret"] >= -1e-12).all()
    glob = regrets[(regrets["scope"] == "global") & (regrets["metric"] == "silhouette")]
    assert glob["chosen"].str.startswith("pixels").all()  # fooled by the pixel offset
    summary = an.summarize_a2(taus, regrets)
    assert {"tau_global", "tau_within", "regret_global", "regret_within"} <= set(summary.columns)


def test_reducer_effect_and_hdbscan_gain(runs: pd.DataFrame) -> None:
    cells = an.cell_means(runs)
    effects = an.reducer_effect(cells)
    assert set(effects["reducer"]) == {"pca50", "umap15"}
    res = an.compare_reducer_gain(effects, "umap15")
    assert res["mean_delta_a"] > res["mean_delta_b"]
    assert res["fraction_a_gains_more"] >= 0.75  # pixels: both near 0 after clipping
    assert res["wilcoxon_p"] < 0.05


def test_run_analysis_writes_report(tmp_path: Path) -> None:
    cfg = write_grid(tmp_path / "res")
    summary = run_analysis(cfg, tmp_path / "report")
    text = summary.read_text()
    for section in ("A1", "A2", "A3", "Integrity"):
        assert section in text
    for fig in ("fig_a1_variance.png", "fig_ari_heatmaps.png",
                "fig_a2_silhouette_vs_ari.png", "fig_a3_reducer_effect.png"):  # fmt: skip
        assert (tmp_path / "report" / fig).stat().st_size > 10_000
    var = pd.read_csv(tmp_path / "report" / "a1_variance.csv")
    assert set(var["subset"]) == {"all", "no pixels", "no pixels, no HDBSCAN"}
    sums = var.groupby(["subset", "dataset"])["eta2"].sum()
    assert np.allclose(sums, 1.0)  # shares sum to one within every subset and dataset
    assert (tmp_path / "report" / "fig_a1_variance_no_pixels_no_hdbscan.png").exists()
    a2 = pd.read_csv(tmp_path / "report" / "a2_summary.csv")
    assert set(a2["subset"]) == {"all", "no HDBSCAN"}
    assert (tmp_path / "report" / "a3_hdbscan_coverage.csv").exists()


def test_run_analysis_refuses_incomplete_grid_unless_allowed(tmp_path: Path) -> None:
    cfg = write_grid(tmp_path / "res", drop=5)
    with pytest.raises(RuntimeError, match="Grid incomplete: 5 missing"):
        run_analysis(cfg, tmp_path / "report")
    summary = run_analysis(cfg, tmp_path / "report", allow_incomplete=True)
    assert "Provisional" in summary.read_text()


def test_subsets_drop_the_extreme_levels(runs: pd.DataFrame) -> None:
    assert "pixels" not in set(an.subset(runs, "no pixels")["encoder"])
    core = an.subset(runs, "no pixels, no HDBSCAN")
    assert "pixels" not in set(core["encoder"]) and "hdbscan" not in set(core["clusterer"])
    assert len(an.subset(runs, "all")) == len(runs)


def test_analysis_from_frozen_parquet(tmp_path: Path) -> None:
    """The committed Parquet export must reproduce the report without the raw files."""
    from ffc.results import load_results

    cfg = write_grid(tmp_path / "res")
    frozen = tmp_path / "grid_runs.parquet"
    load_results(tmp_path / "res").to_parquet(frozen, index=False)
    summary = run_analysis(cfg, tmp_path / "report", runs_path=frozen)
    text = summary.read_text()
    assert str(frozen) in text and "216/216 runs" in text
