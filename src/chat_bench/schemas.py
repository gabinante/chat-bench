"""Data schemas for ChatBench benchmark tasks."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkQuery(BaseModel):
    """A single query in a benchmark task."""

    query_id: str
    query_text: str
    relevant_doc_ids: list[str]  # ground truth relevant document IDs
    metadata: dict = Field(default_factory=dict)


class BenchmarkCorpusDoc(BaseModel):
    """A document in the retrieval corpus."""

    doc_id: str
    text: str
    source: str  # "discord", "slack", "irc"
    metadata: dict = Field(default_factory=dict)


class BenchmarkTask(BaseModel):
    """A complete benchmark task definition."""

    task_id: str
    task_name: str
    description: str
    queries: list[BenchmarkQuery]
    corpus: list[BenchmarkCorpusDoc]


class EvalResult(BaseModel):
    """Results from evaluating a model on a benchmark task."""

    task_id: str
    task_name: str
    model_name: str
    mrr_at_10: float
    map_at_10: float = 0.0
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    num_queries: int
    num_corpus_docs: int
    confidence_intervals: dict[str, list[float]] = Field(default_factory=dict)
    per_difficulty: dict[str, dict] = Field(default_factory=dict)
    hard_negative_metrics: dict = Field(default_factory=dict)
    robustness_score: float | None = None
    metric_std_devs: dict[str, float] = Field(default_factory=dict)
    mean_query_stability: float | None = None
    n_paraphrases: int = 0

    def to_row(self) -> dict:
        row = {
            "Task": self.task_name,
            "Model": self.model_name,
            "MRR@10": f"{self.mrr_at_10:.4f}",
            "MAP@10": f"{self.map_at_10:.4f}",
            "R@1": f"{self.recall_at_1:.4f}",
            "R@5": f"{self.recall_at_5:.4f}",
            "R@10": f"{self.recall_at_10:.4f}",
            "NDCG@10": f"{self.ndcg_at_10:.4f}",
            "Queries": self.num_queries,
            "Corpus": self.num_corpus_docs,
        }
        # Append CI strings if available
        for metric, ci in self.confidence_intervals.items():
            row[f"{metric}_CI"] = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        return row
