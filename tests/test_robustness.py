"""Tests for robustness evaluation."""

from chat_bench.robustness import RobustnessResult


def test_robustness_result_fields():
    result = RobustnessResult(
        robustness_score=0.92,
        metric_std_devs={"MRR@10": 0.03, "R@1": 0.05},
        mean_query_stability=0.08,
        n_paraphrases=5,
        original_metrics={"MRR@10": 0.85},
        paraphrase_metrics={"MRR@10": 0.82},
    )
    assert result.robustness_score == 0.92
    assert result.n_paraphrases == 5
    assert result.mean_query_stability == 0.08
    assert "MRR@10" in result.metric_std_devs
    assert result.metric_std_devs["MRR@10"] == 0.03


def test_robustness_result_defaults():
    result = RobustnessResult(robustness_score=0.9)
    assert result.metric_std_devs == {}
    assert result.mean_query_stability == 0.0
    assert result.n_paraphrases == 0
    assert result.original_metrics == {}
    assert result.paraphrase_metrics == {}


def test_robustness_score_range():
    """Robustness score should be between 0 and 1."""
    result = RobustnessResult(
        robustness_score=0.95,
        mean_query_stability=0.05,
    )
    assert 0.0 <= result.robustness_score <= 1.0


def test_schema_robustness_fields():
    """EvalResult should accept robustness fields."""
    from chat_bench.schemas import EvalResult

    result = EvalResult(
        task_id="test",
        task_name="Test Task",
        model_name="test-model",
        mrr_at_10=0.8,
        recall_at_1=0.5,
        recall_at_5=0.7,
        recall_at_10=0.9,
        ndcg_at_10=0.75,
        num_queries=10,
        num_corpus_docs=100,
        robustness_score=0.92,
        metric_std_devs={"MRR@10": 0.03},
        mean_query_stability=0.08,
        n_paraphrases=5,
    )
    assert result.robustness_score == 0.92
    assert result.n_paraphrases == 5
    assert result.metric_std_devs == {"MRR@10": 0.03}


def test_schema_robustness_defaults():
    """EvalResult robustness fields should default to None/empty."""
    from chat_bench.schemas import EvalResult

    result = EvalResult(
        task_id="test",
        task_name="Test Task",
        model_name="test-model",
        mrr_at_10=0.8,
        recall_at_1=0.5,
        recall_at_5=0.7,
        recall_at_10=0.9,
        ndcg_at_10=0.75,
        num_queries=10,
        num_corpus_docs=100,
    )
    assert result.robustness_score is None
    assert result.metric_std_devs == {}
    assert result.mean_query_stability is None
    assert result.n_paraphrases == 0
