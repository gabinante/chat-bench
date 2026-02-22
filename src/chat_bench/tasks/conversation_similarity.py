"""Task: Conversation Similarity

Given a full conversation, find other conversations about similar topics.
Tests: high-level semantic understanding without requiring metadata (summaries/titles).

Uses seed-confounder relationships: each seed's confounders are topically similar
conversations that should be retrievable given the seed as a query.
"""

from __future__ import annotations

from ..schemas import BenchmarkCorpusDoc, BenchmarkQuery, BenchmarkTask


def build_conversation_similarity_task(
    conversations: list[dict],
    confounder_map: dict[str, list[str]],
) -> BenchmarkTask:
    """Build conversation similarity task from seed-confounder relationships.

    Query: seed conversation text (messages joined with author attribution)
    Positives: confounder conversations (topically similar)
    Corpus: all conversations

    The seed conversation appears in the corpus but is NOT listed in
    relevant_doc_ids — so the self-match ranks high but doesn't count as
    a hit. Metrics measure finding truly similar conversations.

    Args:
        conversations: List of dicts with 'id', 'source', 'messages'
        confounder_map: Mapping of seed_id -> list of confounder conversation_ids
    """
    # Build lookup for conversation text
    conv_by_id = {c["id"]: c for c in conversations}

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

    # Build queries: one per seed that has confounders in the corpus
    queries = []
    corpus_ids = {c["id"] for c in conversations}
    for seed_id, confounder_ids in confounder_map.items():
        if seed_id not in conv_by_id:
            continue

        # Only include confounders that exist in the corpus
        valid_confounders = [cid for cid in confounder_ids if cid in corpus_ids]
        if not valid_confounders:
            continue

        seed_conv = conv_by_id[seed_id]
        query_text = "\n".join(seed_conv["messages"])

        queries.append(
            BenchmarkQuery(
                query_id=f"cs_q_{seed_id}",
                query_text=query_text,
                relevant_doc_ids=valid_confounders,
                metadata={"seed_id": seed_id},
            )
        )

    return BenchmarkTask(
        task_id="conversation_similarity",
        task_name="Conversation Similarity",
        description="Given a conversation, find other conversations about similar topics",
        queries=queries,
        corpus=corpus,
    )
