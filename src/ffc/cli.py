"""Command-line interface: ``ffc extract``, ``ffc run``, ``ffc report``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from ffc.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ConfigOpt = typer.Option(..., "--config", "-c", exists=True, help="Path to experiment YAML")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def extract(
    config: Path = ConfigOpt,
    overwrite: bool = typer.Option(False, help="Recompute features even if cached"),
    verbose: bool = False,
) -> None:
    """Download datasets and cache features for every encoder in the config."""
    from ffc.features import extract_all

    _setup_logging(verbose)
    extract_all(load_config(config), overwrite=overwrite)


@app.command()
def run(config: Path = ConfigOpt, verbose: bool = False) -> None:
    """Run all pending cells of the grid on cached features (resumable)."""
    from ffc.runner import run_grid

    _setup_logging(verbose)
    n = run_grid(load_config(config))
    typer.echo(f"Executed {n} runs.")


@app.command()
def report(
    config: Path = ConfigOpt,
    metric: str = typer.Option("ari", help="Metric to summarize"),
) -> None:
    """Export results to Parquet and print a summary table (mean over seeds)."""
    from ffc.runner import load_results, summarize

    cfg = load_config(config)
    df = load_results(cfg.paths.results_dir)
    out = Path(cfg.paths.results_dir) / "runs.parquet"
    df.to_parquet(out, index=False)
    typer.echo(f"{len(df)} runs -> {out}\n")
    with_reducer = summarize(df, metric=metric, by=("encoder", "reducer"))
    typer.echo(with_reducer.to_string())


if __name__ == "__main__":
    app()
