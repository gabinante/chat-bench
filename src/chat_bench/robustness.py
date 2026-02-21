"""PTEB-style robustness evaluation for retrieval models.

Measures how stable retrieval metrics are across paraphrased queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rich.console import Console

from .metrics import compute_all_metrics
from .paraphrase import llm_paraphrases, rule_based_paraphrases
from .schemas import BenchmarkTask

console = Console()


@dataclass
class RobustnessResult:
    """Results from robustness evaluation."""

    robustness_score: float  # 1 - mean per-query MRR std dev (higher = more robust)
    metric_std_devs: dict[str, float] = field(default_factory=dict)
    mean_query_stability: float = 0.0  # mean per-query MRR std dev
    n_paraphrases: int = 0
    original_metrics: dict[str, float] = field(default_factory=dict)
    paraphrase_metrics: dict[str, float] = field(default_factory=dict)


def evaluate_robustness(
    model,
    task: BenchmarkTask,
    n_paraphrases: int = 5,
    use_llm: bool = False,
    llm_model: str = "claude-sonnet-4-20250514",
    batch_size: int = 128,
    use_bm25: bool = False,
    seed: int = 42,
    model_config: dict | None = None,
) -> RobustnessResult:
    """Evaluate retrieval robustness by testing with paraphrased queries.

    For each query, generates N paraphrases and measures:
    - Per-query metric stability (std dev of MRR across original + paraphrases)
    - Aggregate metric std devs across all paraphrase sets
    - Overall robustness score (1 - mean stability, higher = better)

    Args:
        model: SentenceTransformer model (or None if use_bm25=True).
        task: The benchmark task to evaluate.
        n_paraphrases: Number of paraphrases per query.
        use_llm: If True, use LLM for paraphrasing. Otherwise rule-based.
        llm_model: Anthropic model ID for LLM paraphrasing.
        batch_size: Batch size for encoding.
        use_bm25: If True, use BM25 instead of neural model.
        seed: Random seed.
        model_config: Baseline config dict with instruction prefixes.

    Returns:
        RobustnessResult with stability metrics.
    """
    from .runner import bm25_retrieve, encode_and_retrieve

    query_texts = [q.query_text for q in task.queries]
    corpus_texts = [d.text for d in task.corpus]
    doc_id_map = {i: d.doc_id for i, d in enumerate(task.corpus)}

    query_instruction = (model_config or {}).get("query_instruction", "")
    doc_instruction = (model_config or {}).get("doc_instruction", "")

    # Generate paraphrases
    console.print(f"  Generating {n_paraphrases} paraphrases per query...")
    if use_llm:
        para_map = llm_paraphrases(query_texts, n=n_paraphrases, model=llm_model)
    else:
        para_map = {
            q: rule_based_paraphrases(q, n=n_paraphrases, seed=seed + i)
            for i, q in enumerate(query_texts)
        }

    # Run original queries
    console.print("  Evaluating original queries...")
    if use_bm25:
        orig_rankings = bm25_retrieve(query_texts, corpus_texts)
    else:
        orig_rankings = encode_and_retrieve(
            model, query_texts, corpus_texts, batch_size,
            query_instruction=query_instruction,
            doc_instruction=doc_instruction,
        )

    # Build original results
    orig_results = []
    for i, query in enumerate(task.queries):
        retrieved_ids = [doc_id_map[idx] for idx in orig_rankings[i]]
        orig_results.append({
            "relevant_ids": query.relevant_doc_ids,
            "retrieved_ids": retrieved_ids,
        })

    original_metrics = compute_all_metrics(orig_results)

    # Run each paraphrase set
    per_query_mrr_values: list[list[float]] = [[] for _ in range(len(task.queries))]
    all_paraphrase_metrics: list[dict[str, float]] = []

    for para_idx in range(n_paraphrases):
        para_queries = []
        for q_text in query_texts:
            paras = para_map.get(q_text, [])
            if para_idx < len(paras):
                para_queries.append(paras[para_idx])
            else:
                para_queries.append(q_text)  # fallback to original

        console.print(f"  Evaluating paraphrase set {para_idx + 1}/{n_paraphrases}...")
        if use_bm25:
            para_rankings = bm25_retrieve(para_queries, corpus_texts)
        else:
            para_rankings = encode_and_retrieve(
                model, para_queries, corpus_texts, batch_size,
                query_instruction=query_instruction,
                doc_instruction=doc_instruction,
            )

        para_results = []
        for i, query in enumerate(task.queries):
            retrieved_ids = [doc_id_map[idx] for idx in para_rankings[i]]
            para_results.append({
                "relevant_ids": query.relevant_doc_ids,
                "retrieved_ids": retrieved_ids,
            })

        para_metrics = compute_all_metrics(para_results)
        all_paraphrase_metrics.append(para_metrics)

        # Track per-query MRR for stability calculation
        for i, r in enumerate(para_results):
            relevant = set(r["relevant_ids"])
            mrr = 0.0
            for j, doc_id in enumerate(r["retrieved_ids"][:10]):
                if doc_id in relevant:
                    mrr = 1.0 / (j + 1)
                    break
            per_query_mrr_values[i].append(mrr)

    # Add original MRR to per-query values
    for i, r in enumerate(orig_results):
        relevant = set(r["relevant_ids"])
        mrr = 0.0
        for j, doc_id in enumerate(r["retrieved_ids"][:10]):
            if doc_id in relevant:
                mrr = 1.0 / (j + 1)
                break
        per_query_mrr_values[i].insert(0, mrr)

    # Compute per-query MRR stability (std dev across paraphrases)
    query_stabilities = []
    for mrr_values in per_query_mrr_values:
        if len(mrr_values) > 1:
            query_stabilities.append(float(np.std(mrr_values)))
        else:
            query_stabilities.append(0.0)

    mean_stability = float(np.mean(query_stabilities)) if query_stabilities else 0.0
    robustness_score = 1.0 - mean_stability

    # Compute metric-level std devs
    metric_std_devs: dict[str, float] = {}
    metric_keys = ["MRR@10", "MAP@10", "R@1", "R@5", "R@10", "NDCG@10"]
    for key in metric_keys:
        values = [original_metrics[key]] + [m[key] for m in all_paraphrase_metrics]
        metric_std_devs[key] = float(np.std(values))

    # Average paraphrase metrics
    avg_para_metrics: dict[str, float] = {}
    for key in metric_keys:
        avg_para_metrics[key] = float(np.mean([m[key] for m in all_paraphrase_metrics]))

    return RobustnessResult(
        robustness_score=robustness_score,
        metric_std_devs=metric_std_devs,
        mean_query_stability=mean_stability,
        n_paraphrases=n_paraphrases,
        original_metrics=original_metrics,
        paraphrase_metrics=avg_para_metrics,
    )
