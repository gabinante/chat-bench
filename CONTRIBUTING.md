# Contributing to ChatBench

Thank you for your interest in contributing to ChatBench! This document provides guidelines for contributing.

## Getting Started

### Development Setup

```bash
git clone https://github.com/gabinante/chat-bench.git
cd chat-bench
uv sync --dev
```

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Skip network-dependent tests
uv run pytest tests/ -v --ignore=tests/test_data.py
```

### Linting

```bash
make lint       # auto-fix
make lint-check # check only
```

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting (line length: 100, target: Python 3.11).

## Ways to Contribute

### Submit Model Results

The easiest way to contribute is to evaluate a model and submit results:

```bash
chat-bench evaluate your-org/your-model
```

Open a PR adding your results JSON to `results/` and we'll add them to the leaderboard.

### Add a Model

To add a new model:

1. Edit `src/chat_bench/models.py` — add an entry to the `MODELS` dict
2. Add the model's metadata to `leaderboard/app.py` in `MODEL_META`
3. Run the evaluation to verify: `chat-bench evaluate your-model-id`
4. Submit a PR with the model config and results

### Add a New Task

New retrieval tasks should:

1. Produce `BenchmarkTask` objects matching the schema in `src/chat_bench/schemas.py`
2. Include at least 100 queries (300+ preferred for statistical significance)
3. Use the shared corpus (or extend it with justification)
4. Add an MTEB-compatible task class in `src/chat_bench/mteb_tasks.py`
5. Include tests

### Report Bugs or Request Features

Open an issue at https://github.com/gabinante/chat-bench/issues.

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Make your changes
3. Ensure tests pass: `make test`
4. Ensure linting passes: `make lint-check`
5. Submit a PR with a clear description of the changes

## Code Style

- Python 3.11+ features are welcome
- Type hints encouraged but not required
- Pydantic models for data schemas
- Click for CLI commands
- Keep dependencies minimal — core deps are for evaluation, optional deps for generation/visualization
