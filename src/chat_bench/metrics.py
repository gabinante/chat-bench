"""Retrieval evaluation metrics for ChatBench."""

from __future__ import annotations

import numpy as np


def mrr_at_k(results: list[dict], k: int = 10) -> float:
    """Mean Reciprocal Rank @ k.

    Each result dict: {"relevant_ids": list[str], "retrieved_ids": list[str]}
    """
    rrs = []
    for r in results:
        relevant = set(r["relevant_ids"])
        for i, doc_id in enumerate(r["retrieved_ids"][:k]):
            if doc_id in relevant:
                rrs.append(1.0 / (i + 1))
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs))


def recall_at_k(results: list[dict], k: int = 10) -> float:
    """Recall @ k — fraction of queries with at least one relevant doc in top-k."""
    hits = 0
    for r in results:
        relevant = set(r["relevant_ids"])
        retrieved = set(r["retrieved_ids"][:k])
        if relevant & retrieved:
            hits += 1
    return hits / len(results) if results else 0.0


def ndcg_at_k(results: list[dict], k: int = 10) -> float:
    """NDCG @ k with binary relevance."""
    ndcgs = []
    for r in results:
        relevant = set(r["relevant_ids"])
        dcg = 0.0
        for i, doc_id in enumerate(r["retrieved_ids"][:k]):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 2)
        # IDCG: best case with min(len(relevant), k) relevant docs at top
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def compute_all_metrics(results: list[dict]) -> dict:
    """Compute all standard retrieval metrics."""
    return {
        "MRR@10": mrr_at_k(results, k=10),
        "R@1": recall_at_k(results, k=1),
        "R@5": recall_at_k(results, k=5),
        "R@10": recall_at_k(results, k=10),
        "NDCG@10": ndcg_at_k(results, k=10),
    }
