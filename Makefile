.PHONY: setup setup-gpu lint format typecheck test check extract run smoke clean

CONFIG ?= configs/grid.yaml

setup:            ## Install core + dev dependencies
	uv sync
	uv run pre-commit install

setup-gpu:        ## Install core + dev + feature-extraction dependencies
	uv sync --extra features

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test   ## Everything CI runs

extract:          ## Download datasets and cache features (needs `make setup-gpu`)
	uv run ffc extract --config $(CONFIG)

run:              ## Run the clustering grid on cached features (resumable)
	uv run ffc run --config $(CONFIG)

smoke:            ## Small end-to-end run (CIFAR-10, pixels + DINOv2, 1 seed)
	uv run ffc extract --config configs/smoke.yaml
	uv run ffc run --config configs/smoke.yaml

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
