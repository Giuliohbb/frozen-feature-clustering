"""Result storage under concurrent / duplicated writers (the LAB150 job-332 incident)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ffc import results


def _rec(run_id: str, ts: str, status: str = "ok", ari: float = 0.5) -> str:
    return json.dumps({"run_id": run_id, "timestamp": ts, "status": status, "ari": ari})


def test_writer_path_is_unique_per_process_and_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "332")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    p = results.writer_path("/r")
    assert p.parent == Path("/r")
    assert p.name.startswith("runs-332r1-") and p.name.endswith(
        f"-{__import__('os').getpid()}.jsonl"
    )
    prov = results.provenance()
    assert prov["slurm_job_id"] == "332" and prov["slurm_restart_count"] == 1


def test_reader_merges_legacy_and_per_process_files(tmp_path: Path) -> None:
    (tmp_path / "runs.jsonl").write_text(_rec("a", "2026-11-13T01:00:00") + "\n")
    (tmp_path / "runs-1r0-b4-10.jsonl").write_text(_rec("b", "2026-11-13T02:00:00") + "\n")
    (tmp_path / "runs-1r1-b4-11.jsonl").write_text(_rec("b", "2026-11-13T02:00:05") + "\n")
    # a manual backup in the same folder must not be read
    (tmp_path / "runs.backup-2026-11-16.jsonl").write_text(_rec("z", "2026-11-13T00:00:00") + "\n")
    assert results.read_done(tmp_path) == {"a", "b"}


def test_latest_timestamp_wins_regardless_of_file_order(tmp_path: Path) -> None:
    (tmp_path / "runs-2r0-e4-1.jsonl").write_text(
        _rec("a", "2026-11-16T10:00:00", "ok", 0.9) + "\n"
    )
    (tmp_path / "runs.jsonl").write_text(
        _rec("a", "2026-11-13T10:00:00", "error", float("nan")) + "\n"
    )
    df = results.load_results(tmp_path)
    assert len(df) == 1 and df.loc[0, "status"] == "ok" and df.loc[0, "ari"] == 0.9


def test_integrity_report(tmp_path: Path) -> None:
    lines = [
        _rec("a", "t1"),
        _rec("a", "t2"),  # duplicate from a second writer
        _rec("b", "t1", status="error"),
        "\x00\x00{torn",
        _rec("x", "t1"),  # not part of the grid
    ]
    (tmp_path / "runs.jsonl").write_text("\n".join(lines) + "\n")
    rep = results.integrity_report(tmp_path, ["a", "b", "c"])
    assert rep["n_expected"] == 3
    assert rep["n_missing"] == 1 and rep["missing"] == ["c"]
    assert rep["n_errors"] == 1
    assert rep["n_duplicate_records"] == 1
    assert rep["n_corrupt_lines"] == 1
    assert rep["n_unexpected"] == 1
    assert rep["complete"] is False


def _full(seed: int, reducer: str, ari: float, dim: int) -> str:
    rid = f"cifar10|pixels|{reducer}|kmeans|seed={seed}"
    return json.dumps(
        {
            "run_id": rid,
            "dataset": "cifar10",
            "encoder": "pixels",
            "reducer": reducer,
            "clusterer": "kmeans",
            "seed": seed,
            "timestamp": f"2026-11-13T0{seed}:00:00",
            "ari": ari,
            "dim_reduced": dim,
            "status": "ok",
        }
    )


def test_spliced_lines_are_rejected(tmp_path: Path) -> None:
    """Job 332: the head of one record glued to the tail of another can still be valid JSON."""
    a = _full(0, "pca50", 0.0, 50)
    b = _full(1, "umap15", 0.9, 10)
    # torn mid-key, like the real ``"mestamp"`` / ``"sklearn_veource"`` lines
    spliced = a[: a.index('"timestamp"') + 1] + b[b.index('"encoder"') + 3 :]
    assert json.loads(spliced)["run_id"] == json.loads(a)["run_id"]  # it does parse
    # same design fields, but a run id from another record
    mismatched = json.dumps({**json.loads(a), "run_id": json.loads(b)["run_id"]})
    (tmp_path / "runs.jsonl").write_text("\n".join([spliced, mismatched, a, b]) + "\n")
    records, stats = results.read_records(tmp_path)
    assert stats.n_skipped == 2
    assert [r["ari"] for r in records] == [0.0, 0.9]
