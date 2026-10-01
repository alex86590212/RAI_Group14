# RAI Group 14: Fairness Analysis of Readmission Prediction

Case 5 (Healthcare: Clinical Risk Prediction). Predict 30-day hospital readmission on the
UCI Diabetes 130-US Hospitals (1999-2008) dataset and audit disparities, primarily by race
(secondary: gender or age).

## Installation

```bash
uv sync --frozen
```

Python 3.11. Dependencies are pinned in `uv.lock`.

## Data access

Fetched automatically from the UCI repository (dataset id 296) into `data/raw/` on first run.

## Reproduce

```bash
uv run python scripts/01_audit.py
uv run python scripts/02_baseline.py
uv run python scripts/03_fairness.py
uv run python scripts/04_intervention.py
```

Seeds and settings live in `configs/default.yaml`. Outputs go to `results/`.

## Layout

- `src/rai/`: library code (data, preprocessing, splitting, audit, models, fairness, intervention, evaluation)
- `scripts/`: numbered entry points, run in order
- `configs/`: seeds and settings
- `tests/`: pytest suite
- `notebooks/`: exploration only
- `DATA_CARD.md`, `ATTRIBUTION.md`, `CONTRIBUTIONS.md`: required deliverable documents
