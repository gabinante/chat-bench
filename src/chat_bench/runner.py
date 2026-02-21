"""Benchmark runner — evaluates models on ChatBench tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from .metrics import (
    compute_hard_negative_metrics,
    compute_metrics_with_ci,
    compute_per_difficulty_metrics,
)
from .schemas import BenchmarkTask, EvalResult

console = Console()


def encode_and_retrieve(
    model: SentenceTransformer,
    queries: list[str],
    corpus: list[str],
    batch_size: int = 128,
    query_instruction: str = "",
    doc_instruction: str = "",
) -> list[list[int]]:
    """Encode queries and corpus, return ranked corpus indices for each query."""
    console.print(f"  Encoding {len(queries)} queries...")
    q_kwargs: dict = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "normalize_embeddings": True,
    }
    if query_instruction:
        q_kwargs["prompt"] = query_instruction
    q_emb = model.encode(queries, **q_kwargs)

    console.print(f"  Encoding {len(corpus)} corpus docs...")
    c_kwargs: dict = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "normalize_embeddings": True,
    }
    if doc_instruction:
        c_kwargs["prompt"] = doc_instruction
    c_emb = model.encode(corpus, **c_kwargs)

    # Cosine similarity (already normalized)
    sims = np.array(q_emb) @ np.array(c_emb).T
    rankings = np.argsort(-sims, axis=1)
    return rankings.tolist()


def bm25_retrieve(
    queries: list[str],
    corpus: list[str],
) -> list[list[int]]:
    """Retrieve using BM25 (lexical baseline). Returns ranked corpus indices."""
    console.print(f"  BM25 scoring {len(queries)} queries against {len(corpus)} docs...")
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    rankings = []
    for q in queries:
        tokenized_query = q.lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked = np.argsort(-scores).tolist()
        rankings.append(ranked)

    return rankings


def evaluate_task(
    model: SentenceTransformer | None,
    task: BenchmarkTask,
    model_name: str = "",
    batch_size: int = 128,
    use_bm25: bool = False,
    model_config: dict | None = None,
) -> EvalResult:
    """Evaluate a model (or BM25) on a single benchmark task."""
    n_q, n_d = len(task.queries), len(task.corpus)
    console.print(f"\n[bold cyan]{task.task_name}[/] ({n_q} queries, {n_d} docs)")

    query_texts = [q.query_text for q in task.queries]
    corpus_texts = [d.text for d in task.corpus]
    doc_id_map = {i: d.doc_id for i, d in enumerate(task.corpus)}

    query_instruction = (model_config or {}).get("query_instruction", "")
    doc_instruction = (model_config or {}).get("doc_instruction", "")

    start = time.time()
    if use_bm25:
        rankings = bm25_retrieve(query_texts, corpus_texts)
    else:
        assert model is not None, "model required for neural retrieval"
        rankings = encode_and_retrieve(
            model, query_texts, corpus_texts, batch_size,
            query_instruction=query_instruction,
            doc_instruction=doc_instruction,
        )
    elapsed = time.time() - start
    console.print(f"  Retrieval completed in {elapsed:.1f}s")

    # Build results for metric computation
    results = []
    for i, query in enumerate(task.queries):
        retrieved_ids = [doc_id_map[idx] for idx in rankings[i]]
        entry = {
            "relevant_ids": query.relevant_doc_ids,
            "retrieved_ids": retrieved_ids,
        }
        # Carry through metadata for per-difficulty and hard-negative analysis
        if query.metadata.get("difficulty"):
            entry["difficulty"] = query.metadata["difficulty"]
        if query.metadata.get("hard_negative_ids"):
            entry["hard_negative_ids"] = query.metadata["hard_negative_ids"]
        results.append(entry)

    # Core metrics with bootstrap CIs
    metrics_with_ci = compute_metrics_with_ci(results)
    metrics = metrics_with_ci["metrics"]
    cis = metrics_with_ci["confidence_intervals"]

    # Per-difficulty breakdown
    per_difficulty = compute_per_difficulty_metrics(results)

    # Hard-negative metrics
    hn_metrics = compute_hard_negative_metrics(results)

    # Convert CI tuples to lists for JSON serialization
    ci_lists = {k: list(v) for k, v in cis.items()}

    return EvalResult(
        task_id=task.task_id,
        task_name=task.task_name,
        model_name=model_name or ("bm25" if use_bm25 else "unknown"),
        mrr_at_10=metrics["MRR@10"],
        map_at_10=metrics["MAP@10"],
        recall_at_1=metrics["R@1"],
        recall_at_5=metrics["R@5"],
        recall_at_10=metrics["R@10"],
        ndcg_at_10=metrics["NDCG@10"],
        num_queries=len(task.queries),
        num_corpus_docs=len(task.corpus),
        confidence_intervals=ci_lists,
        per_difficulty=per_difficulty,
        hard_negative_metrics=hn_metrics,
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
    table.add_column("MAP@10", justify="right")
    table.add_column("R@1", justify="right")
    table.add_column("R@5", justify="right")
    table.add_column("R@10", justify="right")
    table.add_column("NDCG@10", justify="right")

    for r in results:
        row = r.to_row()
        table.add_row(
            row["Task"], row["Model"],
            row["MRR@10"], row["MAP@10"],
            row["R@1"], row["R@5"], row["R@10"], row["NDCG@10"],
        )

    console.print(table)

    # Print CIs if available
    has_cis = any(r.confidence_intervals for r in results)
    if has_cis:
        ci_table = Table(title="95% Bootstrap Confidence Intervals")
        ci_table.add_column("Task", style="cyan")
        ci_table.add_column("Model", style="green")
        ci_table.add_column("MRR@10", justify="right")
        ci_table.add_column("MAP@10", justify="right")
        ci_table.add_column("R@1", justify="right")
        ci_table.add_column("NDCG@10", justify="right")

        for r in results:
            ci = r.confidence_intervals
            def fmt_ci(name: str) -> str:
                if name in ci:
                    return f"[{ci[name][0]:.3f}, {ci[name][1]:.3f}]"
                return "-"
            ci_table.add_row(
                r.task_name, r.model_name,
                fmt_ci("MRR@10"), fmt_ci("MAP@10"),
                fmt_ci("R@1"), fmt_ci("NDCG@10"),
            )

        console.print(ci_table)

    # Print per-difficulty breakdown if available
    has_difficulty = any(r.per_difficulty for r in results)
    if has_difficulty:
        diff_table = Table(title="Per-Difficulty Breakdown (MRR@10)")
        diff_table.add_column("Task", style="cyan")
        diff_table.add_column("Model", style="green")
        diff_table.add_column("Easy", justify="right")
        diff_table.add_column("Medium", justify="right")
        diff_table.add_column("Hard", justify="right")

        for r in results:
            if not r.per_difficulty:
                continue
            easy = r.per_difficulty.get("easy", {})
            medium = r.per_difficulty.get("medium", {})
            hard = r.per_difficulty.get("hard", {})
            diff_table.add_row(
                r.task_name, r.model_name,
                f"{easy.get('MRR@10', 0):.4f} (n={easy.get('count', 0)})" if easy else "-",
                f"{medium.get('MRR@10', 0):.4f} (n={medium.get('count', 0)})" if medium else "-",
                f"{hard.get('MRR@10', 0):.4f} (n={hard.get('count', 0)})" if hard else "-",
            )

        console.print(diff_table)

    # Print hard-negative metrics if available
    has_hn = any(r.hard_negative_metrics for r in results)
    if has_hn:
        hn_table = Table(title="Hard Negative Analysis")
        hn_table.add_column("Task", style="cyan")
        hn_table.add_column("Model", style="green")
        hn_table.add_column("HN Mean Rank", justify="right")
        hn_table.add_column("HN Above Relevant", justify="right")
        hn_table.add_column("HN Queries", justify="right")

        for r in results:
            if not r.hard_negative_metrics:
                continue
            hn = r.hard_negative_metrics
            hn_table.add_row(
                r.task_name, r.model_name,
                f"{hn.get('hn_mean_rank', 0):.1f}",
                f"{hn.get('hn_above_relevant_rate', 0):.1%}",
                str(hn.get("hn_query_count", 0)),
            )

        console.print(hn_table)

    # Print robustness metrics if available
    has_robustness = any(r.robustness_score is not None for r in results)
    if has_robustness:
        rob_table = Table(title="Robustness Analysis")
        rob_table.add_column("Task", style="cyan")
        rob_table.add_column("Model", style="green")
        rob_table.add_column("Robustness", justify="right")
        rob_table.add_column("MRR StdDev", justify="right")
        rob_table.add_column("R@1 StdDev", justify="right")
        rob_table.add_column("NDCG StdDev", justify="right")
        rob_table.add_column("Paraphrases", justify="right")

        for r in results:
            if r.robustness_score is None:
                continue
            std = r.metric_std_devs
            rob_table.add_row(
                r.task_name, r.model_name,
                f"{r.robustness_score:.4f}",
                f"{std.get('MRR@10', 0):.4f}",
                f"{std.get('R@1', 0):.4f}",
                f"{std.get('NDCG@10', 0):.4f}",
                str(r.n_paraphrases),
            )

        console.print(rob_table)
