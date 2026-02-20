"""Task 4: Cross-Platform Transfer

Evaluate on held-out platform data (e.g., Slack) when trained primarily on Discord/IRC.
Tests: generalization across chat platforms.
"""

from __future__ import annotations

import random

from ..schemas import BenchmarkCorpusDoc, BenchmarkQuery, BenchmarkTask


def build_cross_platform_task(
    conversations: list[dict],
    held_out_platform: str = "slack",
    num_queries: int = 500,
    seed: int = 42,
) -> BenchmarkTask:
    """Build the cross-platform transfer benchmark task.

    Uses conversations from the held-out platform as queries,
    with the full mixed-platform corpus for retrieval.

    Args:
        conversations: List of dicts with 'id', 'source', 'messages'
        held_out_platform: Platform to use for queries
        num_queries: Number of queries
        seed: Random seed
    """
    rng = random.Random(seed)

    # Corpus: all conversations
    corpus = []
    for conv in conversations:
        if len(conv["messages"]) < 5:
            continue
        text = "\n".join(conv["messages"])
        corpus.append(
            BenchmarkCorpusDoc(
                doc_id=conv["id"],
                text=text,
                source=conv.get("source", "unknown"),
            )
        )

    # Queries: messages from the held-out platform
    held_out = [c for c in conversations if c.get("source") == held_out_platform and len(c["messages"]) >= 5]
    if len(held_out) < num_queries:
        num_queries = len(held_out)

    selected = rng.sample(held_out, num_queries)

    queries = []
    for conv in selected:
        msg_idx = rng.randint(0, len(conv["messages"]) - 1)
        queries.append(
            BenchmarkQuery(
                query_id=f"cp_q_{conv['id']}",
                query_text=conv["messages"][msg_idx],
                relevant_doc_ids=[conv["id"]],
                metadata={"platform": held_out_platform},
            )
        )

    return BenchmarkTask(
        task_id="cross_platform_transfer",
        task_name=f"Cross-Platform Transfer ({held_out_platform})",
        description=f"Thread retrieval on held-out {held_out_platform} data",
        queries=queries,
        corpus=corpus,
    )
