# Richer Evaluations (2.3 + 2.1)

This repo implements a paired-decoding pipeline for:
- **Discovery (2.3):** bias away from dominant modes to surface unknown-unknown answers.
- **Measurement (2.1):** estimate rare-mode probabilities under the original prompt using
  logit interpolation + importance sampling (with diagnostics).

## Quick start

1) Install PyTorch for your backend (CUDA or Apple Silicon):
   - https://pytorch.org/get-started/locally/
2) Create a Python 3.13 environment with uv and install deps:
   ```bash
   uv venv --python 3.13
   uv pip install -r requirements.txt
   uv pip install -e .
   ```

## Repo layout

- `src/richerevals/`: core library
- `scripts/`: runnable entry points
- `configs/`: YAML configs
- `data/`: local datasets (ignored by git)
- `runs/`: experiment outputs (ignored by git)

## Experiment tracking

Each run writes a timestamped folder under `runs/` containing:
- `config.json`
- `env.json` (versions, git hash)
- `metrics.jsonl`
- `artifacts/`

Set `RICHEREVALS_RUNS_DIR` to customize the output location.

## Notes

- Estimation runs disable top-k/top-p truncation to avoid proposal bias.
- Compute parity is tracked by total forward passes or tokens.
