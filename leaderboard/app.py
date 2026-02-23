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
import plotly.graph_objects as go

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
    "text-embedding-3-small": {"dims": 1536, "year": 2024, "type": "api"},
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
            "Embedding Dims": meta.get("dims", ""),
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


def build_scatter(leaderboard: pd.DataFrame):
    """Build a Performance vs. Embedding Dims scatter plot."""
    if leaderboard.empty or "Embedding Dims" not in leaderboard.columns:
        return None

    plot_df = leaderboard[["Model", "Embedding Dims", "Average"]].copy()
    plot_df["Embedding Dims"] = pd.to_numeric(plot_df["Embedding Dims"], errors="coerce")
    plot_df["Average"] = pd.to_numeric(plot_df["Average"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Embedding Dims", "Average"])

    if plot_df.empty:
        return None

    # Use go.Scatter with plain Python lists to avoid plotly binary serialization
    fig = go.Figure(data=go.Scatter(
        x=plot_df["Embedding Dims"].tolist(),
        y=plot_df["Average"].tolist(),
        text=plot_df["Model"].tolist(),
        mode="markers",
        marker=dict(size=10),
        hovertemplate="%{text}<br>Dims: %{x}<br>NDCG@10: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Efficiency Frontier: Score vs. Embedding Size",
        xaxis_title="Embedding Dims",
        yaxis_title="Average NDCG@10",
        template="plotly_white",
    )
    return fig


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
All scores are **NDCG@10** (higher is better). Models are ranked by **Average** score across all tasks.

**Column guide:** *Embedding Dims* = output vector size | *Type* = open-weight, API, or lexical | *Year* = model release year

[GitHub](https://github.com/gabinante/chat-bench) |
[Submit Results](https://github.com/gabinante/chat-bench/issues)
        """)

        with gr.Tabs():
            with gr.Tab("Leaderboard"):
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

                    fig = build_scatter(leaderboard)
                    if fig is not None:
                        gr.Plot(value=fig)

            with gr.Tab("About"):
                gr.Markdown("""
## What is ChatBench?

ChatBench is a benchmark for evaluating embedding models on **conversational retrieval** tasks.
It measures how well models can encode and retrieve chat messages, threads, and conversations
across a variety of realistic scenarios.

## Metric: NDCG@10

All scores use **Normalized Discounted Cumulative Gain at rank 10 (NDCG@10)**.
This metric rewards models that place the correct result higher in a ranked list of 10 candidates.
A perfect score of 1.0 means the correct answer is always ranked first.

## Tasks

| Task | Description |
|------|-------------|
| **Thread Retrieval** | Find the correct thread for a given message |
| **Response Retrieval** | Find the continuation of a conversation |
| **Conversation Similarity** | Find topically similar conversations |
| **Cross-Platform Transfer** | Thread retrieval on a held-out platform |
| **Topic Retrieval** | Find conversations matching a topic description |
| **Specific Detail** | Find conversations containing a specific detail |
| **Cross-Channel** | Find related conversations across channels |
| **Thread Discrimination** | Distinguish between similar conversations |

## How to Submit Results

1. Run the benchmark on your model:
   ```bash
   pip install chat-bench
   chat-bench evaluate your-model-name
   ```
2. Open an issue on the [GitHub repo](https://github.com/gabinante/chat-bench/issues) with your results JSON file attached.

## Links

- [GitHub Repository](https://github.com/gabinante/chat-bench)
- [Submit Results](https://github.com/gabinante/chat-bench/issues)
                """)

    return app


app = create_app()

if __name__ == "__main__":
    app.launch()
