#!/usr/bin/env python3
"""Run all registered models on all ChatBench tasks and save results.

Usage:
    python scripts/run_models.py                      # all models, all tasks
    python scripts/run_models.py --model bge-base     # single model
    python scripts/run_models.py --tasks-dir data/tasks --output-dir results/

Results are saved as JSON files in the output directory, one per model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path for local dev
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chat_bench.models import MODELS
from chat_bench.runner import evaluate_task, load_task, print_results_table
from chat_bench.schemas import EvalResult

DEFAULT_TASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "tasks"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"


def run_single_model(
    name: str,
    config: dict,
    task_files: list[Path],
    batch_size: int,
    args_device: str | None = None,
) -> list[EvalResult]:
    """Evaluate a single model on all tasks."""
    model_id = config["model_id"]
    is_bm25 = config.get("type") == "lexical"

    model = None
    if not is_bm25:
        from sentence_transformers import SentenceTransformer

        load_kwargs: dict = {}
        if config.get("trust_remote_code"):
            load_kwargs["trust_remote_code"] = True
        if args_device:
            load_kwargs["device"] = args_device
        print(f"\nLoading model: {model_id}")
        model = SentenceTransformer(model_id, **load_kwargs)

    results = []
    for tf in task_files:
        task = load_task(tf)
        result = evaluate_task(
            model,
            task,
            model_name=model_id,
            batch_size=batch_size,
            use_bm25=is_bm25,
            model_config=config,
        )
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ChatBench models")
    parser.add_argument("--model", default=None,
                        help="Run a specific model by short name (e.g. 'bge-base')")
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR),
                        help="Directory containing task JSON files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None,
                        help="Force device (e.g. 'cpu', 'mps', 'cuda')")
    args = parser.parse_args()
    args_device = args.device

    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_files = sorted(tasks_dir.glob("*.json"))
    if not task_files:
        print(f"No task files found in {tasks_dir}")
        sys.exit(1)

    print(f"Found {len(task_files)} tasks in {tasks_dir}")

    # Select models
    if args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}. Available: {list(MODELS.keys())}")
            sys.exit(1)
        models = {args.model: MODELS[args.model]}
    else:
        models = MODELS

    all_results: list[EvalResult] = []

    for name, config in models.items():
        model_id = config["model_id"]
        print(f"\n{'='*60}")
        print(f"Model: {name} ({model_id})")
        print(f"{'='*60}")

        results = run_single_model(name, config, task_files, args.batch_size, args_device)
        all_results.extend(results)

        # Save per-model results
        safe_name = name.replace("/", "__")
        out_path = output_dir / f"{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump([r.model_dump() for r in results], f, indent=2)
        print(f"\nSaved results to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")
    print_results_table(all_results)

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
