#!/usr/bin/env python3
"""Run all models that don't yet have results."""
import json
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Order: smallest/fastest first, largest last
MODEL_ORDER = [
    "bge-large",
    "gte-base",
    "nomic-v1.5",
    "mxbai-large",
    "arctic-l-v2",
    "bge-m3",
    "modernbert-large",
    "nomic-v2-moe",
    "stella-1.5b",
    "gte-qwen2",
    "e5-mistral",
]

for model in MODEL_ORDER:
    result_file = RESULTS_DIR / f"{model}.json"
    if result_file.exists():
        print(f"SKIP {model} (already has results)")
        continue

    print(f"\n{'='*60}")
    print(f"RUNNING: {model}")
    print(f"{'='*60}", flush=True)

    ret = subprocess.run(
        [sys.executable, "scripts/run_models.py", "--model", model],
        cwd=Path(__file__).resolve().parent.parent,
    )

    if ret.returncode != 0:
        print(f"FAILED: {model} (exit code {ret.returncode})")
    else:
        print(f"DONE: {model}")
        # Publish updated results to leaderboard
        print(f"Publishing results to leaderboard...")
        subprocess.run(
            [sys.executable, "scripts/publish_leaderboard.py", "--results-only"],
            cwd=Path(__file__).resolve().parent.parent,
        )

print("\n\nAll models complete. Results:")
for f in sorted(RESULTS_DIR.glob("*.json")):
    print(f"  {f.name}")
