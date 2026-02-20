"""Task 3: Summary-to-Thread Matching

Given a natural language description/summary, retrieve the matching conversation.
Tests: the practical "find the conversation about X" use case.
"""

from __future__ import annotations

from ..schemas import BenchmarkCorpusDoc, BenchmarkQuery, BenchmarkTask


def build_summary_matching_task(
    conversations: list[dict],
    summaries: list[dict],
) -> BenchmarkTask:
    """Build the summary-to-thread matching benchmark task.

    Args:
        conversations: List of dicts with 'id', 'source', 'messages'
        summaries: List of dicts with 'conversation_id', 'summary_text'
            Each summary maps to exactly one conversation.
    """
    # Build corpus from all conversations
    corpus = []
    for conv in conversations:
        text = "\n".join(conv["messages"])
        corpus.append(
            BenchmarkCorpusDoc(
                doc_id=conv["id"],
                text=text,
                source=conv.get("source", "unknown"),
            )
        )

    # Build queries from summaries
    queries = []
    for s in summaries:
        queries.append(
            BenchmarkQuery(
                query_id=f"sm_q_{s['conversation_id']}",
                query_text=s["summary_text"],
                relevant_doc_ids=[s["conversation_id"]],
            )
        )

    return BenchmarkTask(
        task_id="summary_matching",
        task_name="Summary-to-Thread Matching",
        description="Given a natural language summary, retrieve the matching conversation",
        queries=queries,
        corpus=corpus,
    )
