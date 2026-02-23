"""ChatBench Leaderboard — Gradio app for HuggingFace Spaces.

Displays benchmark results for embedding models evaluated on ChatBench tasks.
Reads results from local JSON files or from the HF Hub results dataset.

Usage:
    gradio leaderboard/app.py           # local dev
    # Deploy: push to HF Space with gradio SDK
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

# Task display order
TASKS = [
    "Thread Retrieval",
    "Response Retrieval",
    "Conversation Similarity",
    "Cross-Platform Transfer",
    "Topic Retrieval",
    "Specific Detail",
    "Cross-Channel",
    "Thread Discrimination",
]

# In the HF Space, app.py is at the root so results/ is a sibling.
# Locally, app.py is in leaderboard/ so results/ is one level up.
_app_dir = Path(__file__).resolve().parent
RESULTS_DIR = _app_dir / "results" if (_app_dir / "results").exists() else _app_dir.parent / "results"
HF_RESULTS_REPO = "GabeA/chatbench-results"

# Known model metadata (dims, year, type)
MODEL_META = {
    "BAAI/bge-base-en-v1.5": {"dims": 768, "year": 2023, "type": "open"},
    "BAAI/bge-large-en-v1.5": {"dims": 1024, "year": 2023, "type": "open"},
    "Alibaba-NLP/gte-base-en-v1.5": {"dims": 768, "year": 2024, "type": "open"},
    "all-MiniLM-L6-v2": {"dims": 384, "year": 2021, "type": "open"},
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": {"dims": 1536, "year": 2024, "type": "open"},
    "dunzhang/stella_en_1.5B_v5": {"dims": 1024, "year": 2024, "type": "open"},
    "nomic-ai/nomic-embed-text-v1.5": {"dims": 768, "year": 2024, "type": "open"},
    "Snowflake/snowflake-arctic-embed-l-v2.0": {"dims": 1024, "year": 2025, "type": "open"},
    "mixedbread-ai/mxbai-embed-large-v1": {"dims": 1024, "year": 2024, "type": "open"},
    "BAAI/bge-m3": {"dims": 1024, "year": 2024, "type": "open"},
    "lightonai/modernbert-embed-large": {"dims": 1024, "year": 2025, "type": "open"},
    "nomic-ai/nomic-embed-text-v2-moe": {"dims": 768, "year": 2025, "type": "open"},
    "intfloat/e5-mistral-7b-instruct": {"dims": 4096, "year": 2024, "type": "open"},
    "bm25": {"dims": None, "year": None, "type": "lexical"},
}


def load_results_local() -> pd.DataFrame:
    """Load results from local JSON files."""
    if not RESULTS_DIR.exists():
        return pd.DataFrame()

    rows = []
    for rf in sorted(RESULTS_DIR.glob("*.json")):
        with open(rf) as f:
            data = json.load(f)
        for entry in data:
            rows.append(entry)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_results_hub() -> pd.DataFrame:
    """Load results from HF Hub dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_RESULTS_REPO, split="test")
        return ds.to_pandas()
    except Exception:
        return pd.DataFrame()


def build_leaderboard(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build the leaderboard table from raw results."""
    if results_df.empty:
        return pd.DataFrame()

    # Pivot: one row per model, columns for each task's NDCG@10
    metric = "ndcg_at_10"
    pivot = results_df.pivot_table(
        index="model_name",
        columns="task_name",
        values=metric,
        aggfunc="first",
    )

    # Reorder columns to match TASKS order
    ordered_cols = [t for t in TASKS if t in pivot.columns]
    pivot = pivot[ordered_cols]

    # Add average
    pivot["Average"] = pivot.mean(axis=1)

    # Add model metadata
    meta_rows = []
    for model in pivot.index:
        meta = MODEL_META.get(model, {})
        meta_rows.append({
            "Model": model,
            "Dims": meta.get("dims", ""),
            "Year": meta.get("year", ""),
            "Type": meta.get("type", ""),
        })
    meta_df = pd.DataFrame(meta_rows).set_index("Model")

    # Combine
    leaderboard = pd.concat([meta_df, pivot], axis=1)
    leaderboard = leaderboard.sort_values("Average", ascending=False)
    leaderboard = leaderboard.reset_index().rename(columns={"index": "Model"})

    # Format numeric columns
    for col in ordered_cols + ["Average"]:
        if col in leaderboard.columns:
            leaderboard[col] = leaderboard[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )

    return leaderboard


def create_app() -> gr.Blocks:
    """Create the Gradio leaderboard app."""
    # Try local first, then hub
    results_df = load_results_local()
    if results_df.empty:
        results_df = load_results_hub()

    leaderboard = build_leaderboard(results_df)

    with gr.Blocks(title="ChatBench Leaderboard") as app:
        gr.Markdown("""
# ChatBench Leaderboard

Benchmark results for embedding models on conversational retrieval tasks.
All scores are **NDCG@10** (higher is better).

[GitHub](https://github.com/gabinante/chat-bench) |
[Submit Results](https://github.com/gabinante/chat-bench/issues)
        """)

        if leaderboard.empty:
            gr.Markdown("""
**No results available yet.**

Run baselines to populate:
```bash
python scripts/run_baselines.py
```
            """)
        else:
            gr.Dataframe(
                value=leaderboard,
                interactive=False,
                wrap=False,
            )

        gr.Markdown("""
---
**Tasks:**
- **Thread Retrieval**: Find the correct thread for a message
- **Response Retrieval**: Find the continuation of a conversation
- **Conversation Similarity**: Find topically similar conversations
- **Cross-Platform Transfer**: Thread retrieval on held-out platform
- **Topic Retrieval**: Find conversations by topic description
- **Specific Detail**: Find conversations with a specific detail
- **Cross-Channel**: Find related conversations across channels
- **Thread Discrimination**: Distinguish similar conversations
        """)

    return app


app = create_app()

if __name__ == "__main__":
    app.launch()
