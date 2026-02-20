"""Benchmark runner — evaluates models on ChatBench tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from .metrics import compute_all_metrics
from .schemas import BenchmarkTask, EvalResult

console = Console()


def encode_and_retrieve(
    model: SentenceTransformer,
    queries: list[str],
    corpus: list[str],
    batch_size: int = 128,
) -> list[list[int]]:
    """Encode queries and corpus, return ranked corpus indices for each query."""
    console.print(f"  Encoding {len(queries)} queries...")
    q_emb = model.encode(queries, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    console.print(f"  Encoding {len(corpus)} corpus docs...")
    c_emb = model.encode(corpus, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    # Cosine similarity (already normalized)
    sims = np.array(q_emb) @ np.array(c_emb).T
    rankings = np.argsort(-sims, axis=1)
    return rankings.tolist()


def evaluate_task(
    model: SentenceTransformer,
    task: BenchmarkTask,
    model_name: str = "",
    batch_size: int = 128,
) -> EvalResult:
    """Evaluate a model on a single benchmark task."""
    console.print(f"\n[bold cyan]{task.task_name}[/] ({len(task.queries)} queries, {len(task.corpus)} docs)")

    query_texts = [q.query_text for q in task.queries]
    corpus_texts = [d.text for d in task.corpus]
    doc_id_map = {i: d.doc_id for i, d in enumerate(task.corpus)}

    start = time.time()
    rankings = encode_and_retrieve(model, query_texts, corpus_texts, batch_size)
    elapsed = time.time() - start
    console.print(f"  Retrieval completed in {elapsed:.1f}s")

    # Build results for metric computation
    results = []
    for i, query in enumerate(task.queries):
        retrieved_ids = [doc_id_map[idx] for idx in rankings[i]]
        results.append({
            "relevant_ids": query.relevant_doc_ids,
            "retrieved_ids": retrieved_ids,
        })

    metrics = compute_all_metrics(results)

    return EvalResult(
        task_id=task.task_id,
        task_name=task.task_name,
        model_name=model_name,
        mrr_at_10=metrics["MRR@10"],
        recall_at_1=metrics["R@1"],
        recall_at_5=metrics["R@5"],
        recall_at_10=metrics["R@10"],
        ndcg_at_10=metrics["NDCG@10"],
        num_queries=len(task.queries),
        num_corpus_docs=len(task.corpus),
    )


def load_task(task_path: Path) -> BenchmarkTask:
    """Load a benchmark task from a JSON file."""
    with open(task_path) as f:
        data = json.load(f)
    return BenchmarkTask.model_validate(data)


def print_results_table(results: list[EvalResult]) -> None:
    """Print a formatted results table."""
    table = Table(title="ChatBench Results")
    table.add_column("Task", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("MRR@10", justify="right")
    table.add_column("R@1", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("R@10", justify="right")
    table.add_column("NDCG@10", justify="right")

    for r in results:
        row = r.to_row()
        table.add_row(
            row["Task"], row["Model"],
            row["MRR@10"], row["R@1"], row["R@5"], row["R@10"], row["NDCG@10"],
        )

    console.print(table)
