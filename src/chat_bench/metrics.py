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
    if not rrs:
        return 0.0
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
    if not ndcgs:
        return 0.0
    return float(np.mean(ndcgs))


def mean_avg_precision(results: list[dict], k: int = 10) -> float:
    """Mean Average Precision @ k.

    For each query, compute the average of precision values at each rank
    position where a relevant document is found.
    """
    aps = []
    for r in results:
        relevant = set(r["relevant_ids"])
        hits = 0
        sum_precision = 0.0
        for i, doc_id in enumerate(r["retrieved_ids"][:k]):
            if doc_id in relevant:
                hits += 1
                sum_precision += hits / (i + 1)
        num_relevant = min(len(relevant), k)
        aps.append(sum_precision / num_relevant if num_relevant > 0 else 0.0)
    if not aps:
        return 0.0
    return float(np.mean(aps))


def hard_negative_rank(results: list[dict], k: int = 10) -> float:
    """Mean rank of the highest-ranked hard negative within top-k.

    Measures how often hard negatives (confounders) appear above relevant docs.
    Only computed for results that have hard_negative_ids in metadata.
    Returns the average rank (1-indexed) of the first hard negative in top-k,
    or 0.0 if no hard negatives are found in top-k for any query.
    """
    ranks = []
    for r in results:
        hn_ids = set(r.get("hard_negative_ids", []))
        if not hn_ids:
            continue
        for i, doc_id in enumerate(r["retrieved_ids"][:k]):
            if doc_id in hn_ids:
                ranks.append(i + 1)
                break
        else:
            ranks.append(k + 1)  # not in top-k
    if not ranks:
        return 0.0
    return float(np.mean(ranks))


def hard_negative_above_relevant(results: list[dict], k: int = 10) -> float:
    """Fraction of queries where a hard negative ranks above all relevant docs.

    Only computed for results that have hard_negative_ids in metadata.
    """
    count = 0
    total = 0
    for r in results:
        hn_ids = set(r.get("hard_negative_ids", []))
        if not hn_ids:
            continue
        total += 1
        relevant = set(r["relevant_ids"])
        first_relevant = k + 1
        first_hn = k + 1
        for i, doc_id in enumerate(r["retrieved_ids"][:k]):
            if doc_id in relevant and first_relevant == k + 1:
                first_relevant = i
            if doc_id in hn_ids and first_hn == k + 1:
                first_hn = i
        if first_hn < first_relevant:
            count += 1
    return count / total if total > 0 else 0.0


def bootstrap_ci(
    results: list[dict],
    metric_fn,
    k: int = 10,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns (lower, upper) bounds of the confidence interval.
    """
    if not results:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    n = len(results)
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = [results[i] for i in indices]
        scores.append(metric_fn(sample, k=k))

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1 - alpha) * 100))
    return (lower, upper)


def compute_all_metrics(results: list[dict]) -> dict:
    """Compute all standard retrieval metrics."""
    return {
        "MRR@10": mrr_at_k(results, k=10),
        "MAP@10": mean_avg_precision(results, k=10),
        "R@1": recall_at_k(results, k=1),
        "R@5": recall_at_k(results, k=5),
        "R@10": recall_at_k(results, k=10),
        "NDCG@10": ndcg_at_k(results, k=10),
    }


def compute_metrics_with_ci(
    results: list[dict],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute metrics with 95% bootstrap confidence intervals."""
    metrics = compute_all_metrics(results)

    metric_fns = {
        "MRR@10": mrr_at_k,
        "MAP@10": mean_avg_precision,
        "R@1": lambda r, k: recall_at_k(r, k=1),
        "R@5": lambda r, k: recall_at_k(r, k=5),
        "R@10": recall_at_k,
        "NDCG@10": ndcg_at_k,
    }

    cis = {}
    for name, fn in metric_fns.items():
        k = 1 if name == "R@1" else 5 if name == "R@5" else 10
        lo, hi = bootstrap_ci(results, fn, k=k, n_bootstrap=n_bootstrap, seed=seed)
        cis[name] = (lo, hi)

    return {"metrics": metrics, "confidence_intervals": cis}


def compute_per_difficulty_metrics(results: list[dict]) -> dict[str, dict]:
    """Compute metrics broken down by difficulty level.

    Results must have a 'difficulty' key in each dict (from query metadata).
    Returns {difficulty_level: {metric: value}}.
    """
    by_difficulty: dict[str, list[dict]] = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        by_difficulty.setdefault(diff, []).append(r)

    output = {}
    for diff, diff_results in sorted(by_difficulty.items()):
        output[diff] = compute_all_metrics(diff_results)
        output[diff]["count"] = len(diff_results)

    return output


def compute_hard_negative_metrics(results: list[dict], k: int = 10) -> dict:
    """Compute hard-negative-aware metrics.

    Returns metrics about how models handle confounder/hard negative documents.
    """
    hn_results = [r for r in results if r.get("hard_negative_ids")]
    if not hn_results:
        return {}

    return {
        "hn_mean_rank": hard_negative_rank(hn_results, k=k),
        "hn_above_relevant_rate": hard_negative_above_relevant(hn_results, k=k),
        "hn_query_count": len(hn_results),
    }
