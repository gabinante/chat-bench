.PHONY: install lint lint-check format test test-cov evaluate model remaining publish publish-results run-and-publish remaining-and-publish leaderboard

install:
	uv sync

lint:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

lint-check:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=chat_bench --cov-report=term-missing

# --- Benchmark workflow ---

# Evaluate a single model: make evaluate MODEL=BAAI/bge-base-en-v1.5 OUTPUT=results/bge-base.json
evaluate:
	uv run chat-bench evaluate $(MODEL) --tasks-dir data/tasks --output $(OUTPUT)

# Run a named model: make model NAME=bge-base
model:
	uv run python scripts/run_models.py --model $(NAME) --tasks-dir data/tasks

# Run all remaining models
remaining:
	uv run python scripts/run_remaining.py

# Publish results + app to the HF Space leaderboard
publish:
	uv run python scripts/publish_leaderboard.py

# Publish results only (no app update)
publish-results:
	uv run python scripts/publish_leaderboard.py --results-only

# Run a model and publish: make run-and-publish NAME=bge-base
run-and-publish: model publish-results

# Run all remaining models then publish
remaining-and-publish: remaining publish

# Launch leaderboard locally
leaderboard:
	uv run gradio leaderboard/app.py
