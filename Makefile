.PHONY: install lint lint-check format test test-cov

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
