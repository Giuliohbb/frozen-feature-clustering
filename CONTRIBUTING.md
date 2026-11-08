# Contributing

## Workflow

1. Branch from `main`: `feat/<topic>`, `fix/<topic>`, `exp/<topic>` (experiments) or `docs/<topic>`.
2. Commit with [Conventional Commits](https://www.conventionalcommits.org/): `feat: add CLIP encoder`, `fix: handle empty HDBSCAN partition`, `docs: ...`, `test: ...`, `chore: ...`.
3. Run `make check` before pushing (pre-commit runs ruff on every commit after `make setup`).
4. Open a pull request to `main`. CI must be green before merging.

## Rules

- Never commit datasets, features or results. They live in `data/`, `cache/` and `results/`, which are ignored.
- Every new method is registered in one place and gets a test:
  - encoder: add a function to `ENCODERS` in `src/ffc/features.py`
  - reducer: add a branch to `reduce()` in `src/ffc/reduce.py`
  - clusterer: add a branch to `cluster()` in `src/ffc/cluster.py`
- Experiments are defined by YAML files in `configs/`, not by editing code. A result is only valid if it can be regenerated with `ffc run -c <config>`.
- Code and docs in English; the proposal and the final presentation may be in Portuguese.
