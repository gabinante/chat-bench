"""Tests for ChatBench metrics."""

from chat_bench.metrics import compute_all_metrics, mrr_at_k, ndcg_at_k, recall_at_k


def test_mrr_perfect():
    results = [{"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]}]
    assert mrr_at_k(results, k=10) == 1.0


def test_mrr_second():
    results = [{"relevant_ids": ["doc_1"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]}]
    assert mrr_at_k(results, k=10) == 0.5


def test_mrr_not_found():
    results = [{"relevant_ids": ["doc_99"], "retrieved_ids": ["doc_0", "doc_1"]}]
    assert mrr_at_k(results, k=2) == 0.0


def test_recall_at_1():
    results = [
        {"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1"]},
        {"relevant_ids": ["doc_2"], "retrieved_ids": ["doc_0", "doc_1"]},
    ]
    assert recall_at_k(results, k=1) == 0.5


def test_recall_at_5():
    results = [{"relevant_ids": ["doc_3"], "retrieved_ids": ["a", "b", "c", "doc_3", "d"]}]
    assert recall_at_k(results, k=5) == 1.0


def test_ndcg_first():
    results = [{"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1"]}]
    assert ndcg_at_k(results, k=10) == 1.0


def test_compute_all():
    results = [{"relevant_ids": ["doc_0"], "retrieved_ids": ["doc_0", "doc_1", "doc_2"]}]
    metrics = compute_all_metrics(results)
    assert "MRR@10" in metrics
    assert "R@1" in metrics
    assert "NDCG@10" in metrics
    assert metrics["R@1"] == 1.0
