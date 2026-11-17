"""Analyses A1-A3 of the proposal, as pure functions over the results table.

All analyses work on *successful* runs (``status == "ok"``), one table per dataset (label set).
A configuration (cell) is ``(encoder, reducer, clusterer)``; seeds are replicates of a cell.

A1  Variance decomposition of ARI across factors (eta squared).
A2  Can internal metrics pick configurations without labels? (Kendall tau, regret.)
A3  Effect of UMAP before clustering, per algorithm.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

FACTORS = ("encoder", "reducer", "clusterer")

# Sensitivity subsets. Raw pixels sit near ARI 0 everywhere and HDBSCAN collapses without
# UMAP; both inflate their own factor's share, so headline numbers are re-checked without them.
SUBSETS: dict[str, tuple[tuple[str, str], ...]] = {
    "all": (),
    "no pixels": (("encoder", "pixels"),),
    "no pixels, no HDBSCAN": (("encoder", "pixels"), ("clusterer", "hdbscan")),
    "no HDBSCAN": (("clusterer", "hdbscan"),),
}


def subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows of ``df`` that survive the exclusions of sensitivity subset ``name``."""
    mask = pd.Series(True, index=df.index)
    for column, value in SUBSETS[name]:
        mask &= df[column] != value
    return df[mask]


# Internal metrics and their direction: +1 = higher is better, -1 = lower is better.
INTERNAL_METRICS: dict[str, int] = {
    "silhouette": +1,
    "davies_bouldin": -1,
    "calinski_harabasz": +1,
}


def short_method(method_id: str) -> str:
    """Compact labels for tables and figures: ``umap(n_components=10,n_neighbors=15)`` ->
    ``umap15``, ``pca(n_components=50)`` -> ``pca50``, ``hdbscan(min_cluster_size=50)`` ->
    ``hdbscan``."""
    name = method_id.split("(")[0]
    if name == "umap":
        m = re.search(r"n_neighbors=(\d+)", method_id)
        return f"umap{m.group(1)}" if m else "umap"
    if name == "pca":
        m = re.search(r"n_components=(\d+)", method_id)
        return f"pca{m.group(1)}" if m else "pca"
    return name


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Successful runs only, with short method labels."""
    ok = df[df["status"] == "ok"].copy() if "status" in df else df.copy()
    ok["reducer"] = ok["reducer"].map(short_method)
    ok["clusterer"] = ok["clusterer"].map(short_method)
    return ok


def cell_means(df: pd.DataFrame, metrics: tuple[str, ...] = ("ari",)) -> pd.DataFrame:
    """Mean over seeds per (dataset, encoder, reducer, clusterer), plus seed count and std."""
    g = df.groupby(["dataset", *FACTORS])
    out = g[list(metrics)].mean()
    out["n_seeds"] = g.size()
    out["ari_std"] = g["ari"].std(ddof=0)
    return out.reset_index()


# --------------------------------------------------------------------------------------- A1


def variance_decomposition(df: pd.DataFrame, metric: str = "ari") -> pd.DataFrame:
    """Share of run-level variance of ``metric`` explained by each factor and interaction.

    Classic sum-of-squares decomposition of a full factorial (encoder x reducer x clusterer)
    on cell means, scaled by the number of seeds, plus a within-cell (seed) term. Shares sum
    to 1 per dataset. This is a descriptive eta squared: no F tests, because for
    deterministic cells (e.g. Ward without reduction) the five seeds are identical rather
    than independent replicates.
    """
    rows = []
    for dataset, d in df.groupby("dataset"):
        cells = d.groupby(list(FACTORS))[metric].agg(["mean", "size"])
        levels = [sorted(d[f].unique()) for f in FACTORS]
        full = pd.MultiIndex.from_product(levels, names=FACTORS)
        if len(cells) != len(full):
            missing = full.difference(cells.index)
            raise ValueError(
                f"{dataset}: {len(missing)} empty cells, e.g. {list(missing[:3])}. "
                "The decomposition needs a complete grid."
            )
        M = cells["mean"].reindex(full).to_numpy().reshape([len(lv) for lv in levels])
        n_bar = float(cells["size"].mean())
        mu = M.mean()

        effects: dict[str, float] = {}
        mains = {}
        for i, f in enumerate(FACTORS):
            axes = tuple(j for j in range(3) if j != i)
            a = M.mean(axis=axes, keepdims=True) - mu
            mains[i] = a
            effects[f] = float((np.broadcast_to(a, M.shape) ** 2).sum())
        pairs = {}
        for i, j in combinations(range(3), 2):
            k = 3 - i - j
            ab = M.mean(axis=k, keepdims=True) - mu - mains[i] - mains[j]
            pairs[(i, j)] = ab
            effects[f"{FACTORS[i]} x {FACTORS[j]}"] = float(
                (np.broadcast_to(ab, M.shape) ** 2).sum()
            )
        resid = M - mu - sum(mains.values()) - sum(pairs.values())
        effects[" x ".join(FACTORS)] = float((resid**2).sum())

        cell_mean = d.groupby(list(FACTORS))[metric].transform("mean")
        ss_within = float(((d[metric] - cell_mean) ** 2).sum())
        ss = {k: v * n_bar for k, v in effects.items()} | {"seed (within cell)": ss_within}
        total = sum(ss.values())
        for component, value in ss.items():
            rows.append(
                {
                    "dataset": dataset,
                    "component": component,
                    "eta2": value / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- A2


def _signed(values: pd.Series, metric: str) -> pd.Series:
    return values * INTERNAL_METRICS[metric]


def internal_vs_external(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kendall tau between each internal metric and ARI, and the regret of selecting by it.

    Scopes:
      ``global``          all configurations of a dataset (representations compared too)
      ``within:<enc>``    configurations sharing one representation

    Regret = best ARI in scope - ARI of the configuration the internal metric ranks first.
    ``random_regret`` (best - mean ARI) is the expected regret of picking blindly.
    """
    tau_rows, regret_rows = [], []
    for dataset, d in cells.groupby("dataset"):
        scopes = [("global", d)] + [(f"within:{e}", g) for e, g in d.groupby("encoder")]
        for metric in INTERNAL_METRICS:
            for scope, g in scopes:
                g = g.dropna(subset=[metric, "ari"])
                if len(g) < 3:
                    continue
                score = _signed(g[metric], metric)
                tau, p = stats.kendalltau(score, g["ari"])
                tau_rows.append(
                    {"dataset": dataset, "metric": metric, "scope": scope,
                     "tau": tau, "p_value": p, "n_configs": len(g)}
                )  # fmt: skip
                chosen = g.loc[score.idxmax()]
                best = g.loc[g["ari"].idxmax()]
                regret_rows.append(
                    {
                        "dataset": dataset, "metric": metric, "scope": scope,
                        "chosen": _config_name(chosen), "chosen_ari": chosen["ari"],
                        "best": _config_name(best), "best_ari": best["ari"],
                        "regret": best["ari"] - chosen["ari"],
                        "random_regret": best["ari"] - g["ari"].mean(),
                    }
                )  # fmt: skip
    return pd.DataFrame(tau_rows), pd.DataFrame(regret_rows)


def _config_name(row: pd.Series) -> str:
    return f"{row['encoder']} / {row['reducer']} / {row['clusterer']}"


def summarize_a2(taus: pd.DataFrame, regrets: pd.DataFrame) -> pd.DataFrame:
    """Global vs mean-within-representation tau and regret, per dataset and metric (tests H2)."""
    t = taus.assign(level=np.where(taus["scope"] == "global", "global", "within"))
    r = regrets.assign(level=np.where(regrets["scope"] == "global", "global", "within"))
    tau = t.groupby(["dataset", "metric", "level"])["tau"].mean().unstack("level")
    reg = r.groupby(["dataset", "metric", "level"])["regret"].mean().unstack("level")
    rnd = r.groupby(["dataset", "metric", "level"])["random_regret"].mean().unstack("level")
    out = pd.concat({"tau": tau, "regret": reg, "random_regret": rnd}, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


# --------------------------------------------------------------------------------------- A3


def reducer_effect(cells: pd.DataFrame, baseline: str = "none") -> pd.DataFrame:
    """ARI change from ``baseline`` to every other reducer, per (dataset, encoder, clusterer).

    Also reports the seed standard deviation under each reducer (stability)."""
    wide = cells.pivot_table(
        index=["dataset", "encoder", "clusterer"], columns="reducer", values="ari"
    )
    std = cells.pivot_table(
        index=["dataset", "encoder", "clusterer"], columns="reducer", values="ari_std"
    )
    rows = []
    for reducer in wide.columns:
        if reducer == baseline:
            continue
        delta = (wide[reducer] - wide[baseline]).rename("delta_ari")
        frame = delta.to_frame()
        frame["reducer"] = reducer
        frame["ari_baseline"] = wide[baseline]
        frame["ari_reduced"] = wide[reducer]
        frame["seed_std_reduced"] = std[reducer]
        rows.append(frame.reset_index())
    return pd.concat(rows, ignore_index=True)


def compare_reducer_gain(
    effects: pd.DataFrame, reducer: str, a: str = "hdbscan", b: str = "kmeans"
) -> dict[str, Any]:
    """Is the gain from ``reducer`` larger for clusterer ``a`` than for ``b``? (tests H3)

    Paired over (dataset, encoder): Wilcoxon signed-rank on delta_a - delta_b.
    """
    e = effects[effects["reducer"] == reducer]
    wide = e.pivot_table(index=["dataset", "encoder"], columns="clusterer", values="delta_ari")
    diff = (wide[a] - wide[b]).dropna()
    p = float(stats.wilcoxon(diff).pvalue) if len(diff) >= 6 and (diff != 0).any() else np.nan
    return {
        "reducer": reducer,
        "clusterer_a": a,
        "clusterer_b": b,
        "mean_delta_a": float(wide[a].mean()),
        "mean_delta_b": float(wide[b].mean()),
        "mean_difference": float(diff.mean()),
        "fraction_a_gains_more": float((diff > 0).mean()),
        "n_pairs": len(diff),
        "wilcoxon_p": p,
    }


def hdbscan_coverage(runs: pd.DataFrame) -> pd.DataFrame:
    """HDBSCAN labels low-density points as noise. ARI counts noise as one extra group and
    internal metrics ignore it, so both must be read next to the fraction of points kept."""
    h = runs[runs["clusterer"] == "hdbscan"]
    return (
        h.groupby(["reducer", "encoder"])[["coverage", "n_clusters_found", "ari"]]
        .mean()
        .reset_index()
    )
