"""Command-line interface: ``ffc extract``, ``ffc run``, ``ffc report``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from ffc.config import ExperimentConfig, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ConfigOpt = typer.Option(..., "--config", "-c", exists=True, help="Path to experiment YAML")
DataDirOpt = typer.Option(None, help="Override paths.data_dir (raw datasets)")
CacheDirOpt = typer.Option(None, help="Override paths.cache_dir (features and labels)")
ResultsDirOpt = typer.Option(None, help="Override paths.results_dir (result files)")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Hugging Face downloads log every HTTP request at INFO; keep the job logs readable.
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load(
    config: Path,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    results_dir: Path | None = None,
) -> ExperimentConfig:
    overrides = {"data_dir": data_dir, "cache_dir": cache_dir, "results_dir": results_dir}
    return load_config(config, overrides)


@app.command()
def extract(
    config: Path = ConfigOpt,
    data_dir: Path | None = DataDirOpt,
    cache_dir: Path | None = CacheDirOpt,
    overwrite: bool = typer.Option(False, help="Recompute features even if cached"),
    verbose: bool = False,
) -> None:
    """Download datasets and cache features for every encoder in the config."""
    from ffc.features import extract_all

    _setup_logging(verbose)
    extract_all(_load(config, data_dir, cache_dir), overwrite=overwrite)


@app.command()
def run(
    config: Path = ConfigOpt,
    cache_dir: Path | None = CacheDirOpt,
    results_dir: Path | None = ResultsDirOpt,
    retry_errors: bool = typer.Option(False, help="Re-run cells recorded with status=error"),
    verbose: bool = False,
) -> None:
    """Run all pending cells of the grid on cached features (resumable)."""
    from ffc.runner import read_done, run_grid

    _setup_logging(verbose)
    cfg = _load(config, cache_dir=cache_dir, results_dir=results_dir)
    n = run_grid(cfg, retry_errors=retry_errors)
    n_errors = len(read_done(cfg.paths.results_dir)) - len(
        read_done(cfg.paths.results_dir, include_errors=False)
    )
    typer.echo(f"Executed {n} runs. Cells with status=error: {n_errors}.")


@app.command()
def report(
    config: Path = ConfigOpt,
    results_dir: Path | None = ResultsDirOpt,
    metric: str = typer.Option("ari", help="Metric to summarize"),
) -> None:
    """Check grid integrity, export results to Parquet and print a summary table."""
    from ffc.results import integrity_report, load_results
    from ffc.runner import summarize

    cfg = _load(config, results_dir=results_dir)
    rep = integrity_report(cfg.paths.results_dir, (r.run_id for r in cfg.iter_runs()))
    typer.echo(
        f"Integrity: {rep['n_unique']}/{rep['n_expected']} runs, {rep['n_missing']} missing, "
        f"{rep['n_errors']} errors, {rep['n_duplicate_records']} duplicate records, "
        f"{rep['n_corrupt_lines']} corrupt lines, {rep['n_files']} files"
    )
    df = load_results(cfg.paths.results_dir)
    out = Path(cfg.paths.results_dir) / "runs.parquet"
    df.to_parquet(out, index=False)
    typer.echo(f"{len(df)} runs -> {out}\n")
    typer.echo(summarize(df, metric=metric, by=("encoder", "reducer")).to_string())


if __name__ == "__main__":
    app()
