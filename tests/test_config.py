from __future__ import annotations

from pathlib import Path

import pytest

from ffc.config import ExperimentConfig, MethodSpec, load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def test_method_spec_id_is_stable_and_sorted() -> None:
    a = MethodSpec.from_obj({"name": "umap", "n_neighbors": 15, "n_components": 10})
    b = MethodSpec.from_obj({"n_components": 10, "name": "umap", "n_neighbors": 15})
    assert a == b
    assert a.id == "umap(n_components=10,n_neighbors=15)"
    assert MethodSpec.from_obj("kmeans").id == "kmeans"


def test_full_grid_matches_proposal() -> None:
    cfg = load_config(CONFIGS / "grid.yaml")
    assert cfg.n_runs() == 4 * 4 * 4 * 4 * 5 == 1280
    run_ids = [r.run_id for r in cfg.iter_runs()]
    assert len(run_ids) == len(set(run_ids)) == cfg.n_runs()
    # CIFAR-100 coarse and fine share one image source, so features are extracted once
    assert cfg.sources == ["cifar10", "cifar100", "stl10"]


@pytest.mark.parametrize("name", ["grid.yaml", "smoke.yaml"])
def test_shipped_configs_load(name: str) -> None:
    assert isinstance(load_config(CONFIGS / name), ExperimentConfig)


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown datasets"):
        ExperimentConfig(
            datasets=["imagenet"],
            encoders=["pixels"],
            reducers=[MethodSpec("none")],
            clusterers=[MethodSpec("kmeans")],
            seeds=[0],
        )
