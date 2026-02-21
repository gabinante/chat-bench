"""Tests for ChatBench metrics."""

from chat_bench.metrics import (
    bootstrap_ci,
    compute_all_metrics,
    compute_hard_negative_metrics,
    compute_metrics_with_ci,
    compute_per_difficulty_metrics,
    hard_negative_above_relevant,
    hard_negative_rank,
    mean_avg_precision,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

# --- MRR ---

def test_mrr_perfect():
    results = [{"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]}]
    assert mrr_at_k(results, k=10) == 1.0


def test_mrr_second():
    results = [{"relevant_ids": ["doc_1"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]}]
    assert mrr_at_k(results, k=10) == 0.5


def test_mrr_not_found():
    results = [{"relevant_ids": ["doc_99"], "retrieved_ids": ["doc_0", "doc_1"]}]
    assert mrr_at_k(results, k=2) == 0.0


def test_mrr_empty():
    assert mrr_at_k([], k=10) == 0.0


# --- Recall ---

def test_recall_at_1():
    results = [
        {"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1"]},
        {"relevant_ids": ["doc_2"], "retrieved_ids": ["doc_0", "doc_1"]},
    ]
    assert recall_at_k(results, k=1) == 0.5


def test_recall_at_5():
    results = [{"relevant_ids": ["doc_3"], "retrieved_ids": ["a", "b", "c", "doc_3", "d"]}]
    assert recall_at_k(results, k=5) == 1.0


def test_recall_empty():
    assert recall_at_k([], k=10) == 0.0


# --- NDCG ---

def test_ndcg_first():
    results = [{"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1"]}]
    assert ndcg_at_k(results, k=10) == 1.0


def test_ndcg_empty():
    assert ndcg_at_k([], k=10) == 0.0


# --- MAP ---

def test_map_perfect():
    results = [{"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]}]
    assert mean_avg_precision(results, k=10) == 1.0


def test_map_second():
    results = [{"relevant_ids": ["doc_1"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]}]
    # precision at rank 2 = 1/2 = 0.5, one relevant doc → AP = 0.5/1 = 0.5
    assert mean_avg_precision(results, k=10) == 0.5


def test_map_multiple_relevant():
    results = [{
        "relevant_ids": ["doc_0", "doc_2"],
        "retrieved_ids": ["doc_0", "doc_1", "doc_2", "doc_3"],
    }]
    # doc_0 at rank 1: precision = 1/1 = 1.0
    # doc_2 at rank 3: precision = 2/3 ≈ 0.667
    # AP = (1.0 + 2/3) / 2 = 5/6 ≈ 0.833
    assert abs(mean_avg_precision(results, k=10) - 5 / 6) < 1e-6


def test_map_not_found():
    results = [{"relevant_ids": ["doc_99"], "retrieved_ids": ["doc_0", "doc_1"]}]
    assert mean_avg_precision(results, k=2) == 0.0


def test_map_empty():
    assert mean_avg_precision([], k=10) == 0.0


# --- Hard Negative Metrics ---

def test_hard_negative_rank_found():
    results = [{
        "relevant_ids": ["doc_0"],
        "retrieved_ids": ["doc_1", "doc_hn", "doc_0", "doc_2"],
        "hard_negative_ids": ["doc_hn"],
    }]
    # hard negative at rank 2
    assert hard_negative_rank(results, k=10) == 2.0


def test_hard_negative_rank_not_in_topk():
    results = [{
        "relevant_ids": ["doc_0"],
        "retrieved_ids": ["doc_0", "doc_1"],
        "hard_negative_ids": ["doc_hn"],
    }]
    # hard negative not in top-2 → rank = k+1 = 3
    assert hard_negative_rank(results, k=2) == 3.0


def test_hard_negative_rank_no_hn():
    results = [{
        "relevant_ids": ["doc_0"],
        "retrieved_ids": ["doc_0", "doc_1"],
    }]
    assert hard_negative_rank(results, k=10) == 0.0


def test_hard_negative_above_relevant_yes():
    results = [{
        "relevant_ids": ["doc_0"],
        "retrieved_ids": ["doc_hn", "doc_0", "doc_1"],
        "hard_negative_ids": ["doc_hn"],
    }]
    # HN at rank 0, relevant at rank 1 → HN above relevant
    assert hard_negative_above_relevant(results, k=10) == 1.0


def test_hard_negative_above_relevant_no():
    results = [{
        "relevant_ids": ["doc_0"],
        "retrieved_ids": ["doc_0", "doc_hn", "doc_1"],
        "hard_negative_ids": ["doc_hn"],
    }]
    # relevant at rank 0, HN at rank 1 → relevant above HN
    assert hard_negative_above_relevant(results, k=10) == 0.0


def test_hard_negative_above_relevant_no_hn():
    results = [{
        "relevant_ids": ["doc_0"],
        "retrieved_ids": ["doc_0", "doc_1"],
    }]
    assert hard_negative_above_relevant(results, k=10) == 0.0


def test_compute_hard_negative_metrics():
    results = [
        {
            "relevant_ids": ["doc_0"],
            "retrieved_ids": ["doc_hn", "doc_0", "doc_1"],
            "hard_negative_ids": ["doc_hn"],
        },
        {
            "relevant_ids": ["doc_2"],
            "retrieved_ids": ["doc_2", "doc_hn2", "doc_3"],
            "hard_negative_ids": ["doc_hn2"],
        },
    ]
    hn = compute_hard_negative_metrics(results)
    assert hn["hn_query_count"] == 2
    assert hn["hn_above_relevant_rate"] == 0.5  # 1 out of 2
    assert hn["hn_mean_rank"] == 1.5  # (1 + 2) / 2


# --- Bootstrap CI ---

def test_bootstrap_ci_returns_tuple():
    results = [
        {"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1"]},
        {"relevant_ids": ["doc_1"], "retrieved_ids": ["doc_0", "doc_1"]},
    ]
    lo, hi = bootstrap_ci(results, mrr_at_k, k=10, n_bootstrap=100, seed=42)
    assert lo <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_bootstrap_ci_perfect():
    results = [{"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0"]}] * 10
    lo, hi = bootstrap_ci(results, mrr_at_k, k=10, n_bootstrap=100, seed=42)
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_ci_empty():
    lo, hi = bootstrap_ci([], mrr_at_k, k=10)
    assert lo == 0.0
    assert hi == 0.0


# --- compute_metrics_with_ci ---

def test_compute_metrics_with_ci():
    results = [
        {"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]},
        {"relevant_ids": ["doc_2"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]},
    ]
    out = compute_metrics_with_ci(results, n_bootstrap=100)
    assert "metrics" in out
    assert "confidence_intervals" in out
    assert "MRR@10" in out["metrics"]
    assert "MAP@10" in out["metrics"]
    assert "MRR@10" in out["confidence_intervals"]


# --- Per-Difficulty Metrics ---

def test_per_difficulty_basic():
    results = [
        {"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1"], "difficulty": "easy"},
        {"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1"], "difficulty": "easy"},
        {"relevant_ids": ["doc_99"], "retrieved_ids": ["doc_0", "doc_1"], "difficulty": "hard"},
    ]
    out = compute_per_difficulty_metrics(results)
    assert "easy" in out
    assert "hard" in out
    assert out["easy"]["count"] == 2
    assert out["hard"]["count"] == 1
    assert out["easy"]["MRR@10"] == 1.0
    assert out["hard"]["MRR@10"] == 0.0


def test_per_difficulty_no_difficulty():
    results = [
        {"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1"]},
    ]
    out = compute_per_difficulty_metrics(results)
    assert "unknown" in out
    assert out["unknown"]["count"] == 1


# --- compute_all_metrics ---

def test_compute_all():
    results = [{"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]}]
    metrics = compute_all_metrics(results)
    assert "MRR@10" in metrics
    assert "MAP@10" in metrics
    assert "R@1" in metrics
    assert "NDCG@10" in metrics
    assert metrics["R@1"] == 1.0
    assert metrics["MAP@10"] == 1.0
