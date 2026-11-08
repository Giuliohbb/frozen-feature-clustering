from __future__ import annotations

from pathlib import Path

import yaml
from sklearn.datasets import make_blobs
from typer.testing import CliRunner

from ffc.cli import app
from ffc.store import FeatureStore


def test_run_and_report_commands(tmp_path: Path) -> None:
    X, y = make_blobs(n_samples=200, centers=10, n_features=8, cluster_std=0.3, random_state=0)
    store = FeatureStore(tmp_path / "cache")
    store.save_labels("cifar10", "fine", y)
    store.save_features("cifar10", "good", X)

    config = {
        "datasets": ["cifar10"],
        "encoders": ["good"],
        "reducers": ["none"],
        "clusterers": ["kmeans"],
        "seeds": [0],
        "paths": {"cache_dir": str(tmp_path / "cache"), "results_dir": str(tmp_path / "res")},
    }
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(yaml.safe_dump(config))

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "Executed 1 runs" in result.output

    result = runner.invoke(app, ["report", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "res" / "runs.parquet").exists()
