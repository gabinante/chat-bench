"""Task 2: Response Retrieval

Given a conversation prefix (first N messages), retrieve the correct continuation.
Tests: temporal coherence and conversational flow understanding.
"""

from __future__ import annotations

import random

from ..schemas import BenchmarkCorpusDoc, BenchmarkQuery, BenchmarkTask


def build_response_retrieval_task(
    conversations: list[dict],
    prefix_size: int = 3,
    response_size: int = 3,
    num_queries: int = 1000,
    seed: int = 42,
) -> BenchmarkTask:
    """Build the response retrieval benchmark task.

    For each query:
    - Take the first `prefix_size` messages as the query
    - The next `response_size` messages are the relevant response
    - All other response windows form the distractor corpus

    Args:
        conversations: List of dicts with 'id', 'source', 'messages'
        prefix_size: Number of messages in the query prefix
        response_size: Number of messages in the response window
        num_queries: Number of queries to generate
        seed: Random seed
    """
    rng = random.Random(seed)
    min_len = prefix_size + response_size

    eligible = [c for c in conversations if len(c["messages"]) >= min_len]
    if len(eligible) < num_queries:
        num_queries = len(eligible)

    selected = rng.sample(eligible, num_queries)

    # Build corpus: response windows from all eligible conversations
    corpus = []
    for conv in eligible:
        response_text = "\n".join(conv["messages"][prefix_size : prefix_size + response_size])
        corpus.append(
            BenchmarkCorpusDoc(
                doc_id=f"resp_{conv['id']}",
                text=response_text,
                source=conv.get("source", "unknown"),
            )
        )

    # Build queries
    queries = []
    for conv in selected:
        prefix_text = "\n".join(conv["messages"][:prefix_size])
        queries.append(
            BenchmarkQuery(
                query_id=f"rr_q_{conv['id']}",
                query_text=prefix_text,
                relevant_doc_ids=[f"resp_{conv['id']}"],
            )
        )

    return BenchmarkTask(
        task_id="response_retrieval",
        task_name="Response Retrieval",
        description="Given a conversation prefix, retrieve the correct continuation",
        queries=queries,
        corpus=corpus,
    )
