"""Result storage: one append-only JSONL file per writer process, merged on read.

Why one file per process: on LAB150 a job whose node stopped responding was restarted while
the first copy kept running, and both appended to the same ``runs.jsonl`` on the NFS home.
Concurrent appends from two machines interleave and corrupt lines. With a file per process,
two writers can at worst produce duplicate records, which the reader resolves.

Layout of ``<results_dir>``::

    runs.jsonl                                   legacy single file (read, never written)
    runs-<slurm job>r<restart>-<host>-<pid>.jsonl one file per writer process

A run id may appear more than once (retries, duplicate writers); the record with the latest
``timestamp`` wins.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

LEGACY_FILE = "runs.jsonl"
WRITER_GLOB = "runs-*.jsonl"


def _host() -> str:
    return socket.gethostname().split(".")[0]


def provenance() -> dict[str, Any]:
    """Who wrote a record: Slurm job, restart count, machine and process."""
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0),
        "host": _host(),
        "pid": os.getpid(),
    }


def writer_path(results_dir: str | Path) -> Path:
    """A results file owned by this process alone."""
    p = provenance()
    name = f"runs-{p['slurm_job_id']}r{p['slurm_restart_count']}-{p['host']}-{p['pid']}.jsonl"
    return Path(results_dir) / name


def ensure_trailing_newline(path: Path) -> None:
    """Before appending, make sure a torn last record cannot swallow the next one."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            f.write(b"\n")


def result_files(results_dir: str | Path) -> list[Path]:
    """The legacy file (if any) followed by per-process files, oldest first."""
    root = Path(results_dir)
    files = [root / LEGACY_FILE] if (root / LEGACY_FILE).exists() else []
    files += sorted(root.glob(WRITER_GLOB), key=lambda f: f.stat().st_mtime)
    return files


@dataclass
class ReadStats:
    n_files: int = 0
    n_records: int = 0
    n_skipped: int = 0  # lines that could not be parsed (torn writes, garbage)


class _DuplicateKeyError(ValueError):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKeyError(sorted({k for k in keys if keys.count(k) > 1}))
    return dict(pairs)


_ID_FIELDS = ("dataset", "encoder", "reducer", "clusterer", "seed")


def _run_id_consistent(record: dict[str, Any]) -> bool:
    """A record's run id must spell out its own design fields (when they are present)."""
    if not all(f in record for f in _ID_FIELDS):
        return True
    d = record
    expected = f"{d['dataset']}|{d['encoder']}|{d['reducer']}|{d['clusterer']}|seed={d['seed']}"
    return bool(record["run_id"] == expected)


def _parse_file(path: Path, stats: ReadStats) -> list[dict[str, Any]]:
    """Parse one JSONL file, tolerating lines damaged by an interrupted or concurrent write.

    NUL padding is stripped, so a record that follows it is recovered. Anything still
    unparsable is skipped with a warning, which simply makes that run pending again.

    Two interleaved writes can also splice the head of one record onto the tail of another
    and still form valid JSON. A real writer never emits a key twice, while a splice repeats
    the keys of the tail, so lines with duplicate keys (or whose run id disagrees with their
    own design fields) are rejected as well.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.replace("\x00", "").strip()
            if not line:
                continue
            try:
                record = json.loads(line, object_pairs_hook=_no_duplicate_keys)
            except json.JSONDecodeError:
                log.warning("%s:%d is not valid JSON, skipping: %r", path, lineno, line[:80])
                stats.n_skipped += 1
                continue
            except _DuplicateKeyError as e:
                log.warning("%s:%d repeats keys %s (spliced write), skipping", path, lineno, e)
                stats.n_skipped += 1
                continue
            if not isinstance(record, dict) or "run_id" not in record:
                log.warning("%s:%d has no run_id, skipping", path, lineno)
                stats.n_skipped += 1
                continue
            if not _run_id_consistent(record):
                log.warning("%s:%d run id disagrees with its fields, skipping", path, lineno)
                stats.n_skipped += 1
                continue
            records.append(record)
    return records


def read_records(results_dir: str | Path) -> tuple[list[dict[str, Any]], ReadStats]:
    stats = ReadStats()
    records: list[dict[str, Any]] = []
    for path in result_files(results_dir):
        stats.n_files += 1
        records += _parse_file(path, stats)
    stats.n_records = len(records)
    return records, stats


def latest_by_run(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One record per run id: the most recent timestamp wins (ties: the later one read)."""
    ordered = sorted(records, key=lambda r: str(r.get("timestamp", "")))  # stable sort
    return {r["run_id"]: r for r in ordered}


def read_done(results_dir: str | Path, include_errors: bool = True) -> set[str]:
    latest = latest_by_run(read_records(results_dir)[0])
    return {
        run_id for run_id, r in latest.items() if include_errors or r.get("status", "ok") != "error"
    }


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """All runs, one row per run id. Records without ``status`` (older code) count as ok."""
    latest = latest_by_run(read_records(results_dir)[0])
    df = pd.DataFrame(list(latest.values()))
    if df.empty:
        return df
    if "status" not in df:
        df["status"] = "ok"
    df["status"] = df["status"].fillna("ok")
    return df.reset_index(drop=True)


def integrity_report(results_dir: str | Path, expected_run_ids: Iterable[str]) -> dict[str, Any]:
    """Is the grid complete and clean? Run this before trusting any analysis."""
    records, stats = read_records(results_dir)
    latest = latest_by_run(records)
    counts = Counter(r["run_id"] for r in records)
    expected = set(expected_run_ids)
    missing = sorted(expected - latest.keys())
    unexpected = sorted(latest.keys() - expected)
    errors = sorted(
        rid for rid in expected & latest.keys() if latest[rid].get("status", "ok") == "error"
    )
    return {
        "n_files": stats.n_files,
        "n_records": stats.n_records,
        "n_corrupt_lines": stats.n_skipped,
        "n_duplicate_records": sum(c - 1 for c in counts.values() if c > 1),
        "n_expected": len(expected),
        "n_unique": len(latest),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "n_errors": len(errors),
        "complete": not missing and not errors,
        "missing": missing[:20],
        "errors": [f"{rid}: {latest[rid].get('error', '')}" for rid in errors[:20]],
    }


def integrity_from_frame(df: pd.DataFrame, expected_run_ids: Iterable[str]) -> dict[str, Any]:
    """Same report as ``integrity_report`` for an already deduplicated table (e.g. a frozen
    Parquet export). File-level counts (corrupt lines, duplicates) are not recoverable here."""
    expected = set(expected_run_ids)
    present = set(df["run_id"])
    status = df.set_index("run_id")["status"] if "status" in df else pd.Series(dtype=str)
    missing = sorted(expected - present)
    errors = sorted(rid for rid in expected & present if status.get(rid, "ok") == "error")
    return {
        "n_files": 0,
        "n_records": len(df),
        "n_corrupt_lines": 0,
        "n_duplicate_records": int(df["run_id"].duplicated().sum()),
        "n_expected": len(expected),
        "n_unique": len(present),
        "n_missing": len(missing),
        "n_unexpected": len(present - expected),
        "n_errors": len(errors),
        "complete": not missing and not errors,
        "missing": missing[:20],
        "errors": errors[:20],
    }
