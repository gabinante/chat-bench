"""Task 1: Thread Retrieval

Given a query message, retrieve the correct conversation thread from a corpus of threads.
Tests: semantic understanding of thread coherence.
"""

from __future__ import annotations

import random

from ..schemas import BenchmarkCorpusDoc, BenchmarkQuery, BenchmarkTask


def build_thread_retrieval_task(
    conversations: list[dict],
    num_queries: int = 1000,
    seed: int = 42,
) -> BenchmarkTask:
    """Build the thread retrieval benchmark task.

    For each query:
    - Pick a conversation, extract one message as the query
    - The full conversation (minus the query message) is the relevant doc
    - All other conversations form the distractor corpus

    Args:
        conversations: List of dicts with 'id', 'source', 'messages' (list of text strings)
        num_queries: Number of queries to generate
        seed: Random seed
    """
    rng = random.Random(seed)

    # Filter to conversations with enough messages
    eligible = [c for c in conversations if len(c["messages"]) >= 5]
    if len(eligible) < num_queries:
        num_queries = len(eligible)

    selected = rng.sample(eligible, num_queries)

    # Build corpus: each conversation as one document
    corpus = []
    for conv in eligible:
        text = "\n".join(conv["messages"])
        corpus.append(
            BenchmarkCorpusDoc(
                doc_id=conv["id"],
                text=text,
                source=conv.get("source", "unknown"),
            )
        )

    # Build queries: pick a random message from each selected conversation
    queries = []
    for conv in selected:
        msg_idx = rng.randint(0, len(conv["messages"]) - 1)
        query_text = conv["messages"][msg_idx]

        queries.append(
            BenchmarkQuery(
                query_id=f"tr_q_{conv['id']}",
                query_text=query_text,
                relevant_doc_ids=[conv["id"]],
                metadata={"message_idx": msg_idx},
            )
        )

    return BenchmarkTask(
        task_id="thread_retrieval",
        task_name="Thread Retrieval",
        description="Given a message, retrieve the correct conversation thread",
        queries=queries,
        corpus=corpus,
    )
