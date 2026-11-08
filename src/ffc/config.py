"""Experiment configuration: YAML loading and expansion of the factorial grid."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ffc.data import DATASETS


def _format_params(params: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={params[k]}" for k in sorted(params))


@dataclass(frozen=True)
class MethodSpec:
    """A named method plus its hyperparameters (used for both reducers and clusterers)."""

    name: str
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_obj(cls, obj: str | Mapping[str, Any]) -> MethodSpec:
        if isinstance(obj, str):
            return cls(name=obj)
        obj = dict(obj)
        name = obj.pop("name")
        return cls(name=name, params=tuple(sorted(obj.items())))

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def id(self) -> str:
        """Stable, human-readable identifier, e.g. ``umap(n_components=10,n_neighbors=15)``."""
        return f"{self.name}({_format_params(self.kwargs)})" if self.params else self.name


@dataclass(frozen=True)
class RunSpec:
    """One cell of the grid: a single clustering run."""

    dataset: str
    encoder: str
    reducer: MethodSpec
    clusterer: MethodSpec
    seed: int

    @property
    def run_id(self) -> str:
        return "|".join(
            [self.dataset, self.encoder, self.reducer.id, self.clusterer.id, f"seed={self.seed}"]
        )


@dataclass
class Paths:
    data_dir: Path = Path("data")
    cache_dir: Path = Path("cache")
    results_dir: Path = Path("results/default")


@dataclass
class ExperimentConfig:
    datasets: list[str]
    encoders: list[str]
    reducers: list[MethodSpec]
    clusterers: list[MethodSpec]
    seeds: list[int]
    paths: Paths = field(default_factory=Paths)
    normalize: str = "l2"
    max_samples: int | None = None
    internal_metrics_sample: int | None = None
    device: str = "cuda"
    batch_size: int = 256

    def __post_init__(self) -> None:
        unknown = [d for d in self.datasets if d not in DATASETS]
        if unknown:
            raise ValueError(f"Unknown datasets {unknown}; available: {sorted(DATASETS)}")
        if self.normalize not in {"l2", "none"}:
            raise ValueError(f"normalize must be 'l2' or 'none', got {self.normalize!r}")

    @property
    def sources(self) -> list[str]:
        """Image sources needed by the configured datasets (order preserved, no duplicates)."""
        return list(dict.fromkeys(DATASETS[d].source for d in self.datasets))

    def iter_runs(self) -> Iterator[RunSpec]:
        for dataset in self.datasets:
            for encoder in self.encoders:
                for reducer in self.reducers:
                    for clusterer in self.clusterers:
                        for seed in self.seeds:
                            yield RunSpec(dataset, encoder, reducer, clusterer, seed)

    def n_runs(self) -> int:
        return (
            len(self.datasets)
            * len(self.encoders)
            * len(self.reducers)
            * len(self.clusterers)
            * len(self.seeds)
        )


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    paths = Paths(**{k: Path(v) for k, v in raw.pop("paths", {}).items()})
    return ExperimentConfig(
        datasets=list(raw.pop("datasets")),
        encoders=list(raw.pop("encoders")),
        reducers=[MethodSpec.from_obj(r) for r in raw.pop("reducers")],
        clusterers=[MethodSpec.from_obj(c) for c in raw.pop("clusterers")],
        seeds=[int(s) for s in raw.pop("seeds")],
        paths=paths,
        **raw,
    )
