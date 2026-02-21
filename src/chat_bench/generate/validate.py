"""Cross-reference and consistency validation for generated corpus."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from .schemas import Conversation, RetrievalQuery

logger = logging.getLogger(__name__)


class ValidationReport:
    """Collects and reports validation issues."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [f"Validation: {'PASSED' if self.passed else 'FAILED'}"]
        lines.append(f"  Errors: {len(self.errors)}")
        lines.append(f"  Warnings: {len(self.warnings)}")
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def validate_corpus(
    conversations: list[Conversation],
    queries: list[RetrievalQuery],
    channel_participants: dict[str, list[str]] | None = None,
) -> ValidationReport:
    """Run all validation checks on the corpus and queries."""
    report = ValidationReport()

    _check_conversation_id_uniqueness(conversations, report)
    _check_message_id_uniqueness(conversations, report)
    _check_timestamp_ordering(conversations, report)
    _check_cross_reference_integrity(conversations, report)

    if channel_participants:
        _check_participant_consistency(conversations, channel_participants, report)

    if queries:
        conv_ids = {c.conversation_id for c in conversations}
        _check_query_reference_validity(queries, conv_ids, report)
        _check_query_coverage(conversations, queries, report)
        _check_difficulty_distribution(queries, report)
        _check_bm25_solvability(queries, conversations, report)

    return report


def _check_conversation_id_uniqueness(
    conversations: list[Conversation], report: ValidationReport
) -> None:
    ids = [c.conversation_id for c in conversations]
    dupes = [cid for cid, count in Counter(ids).items() if count > 1]
    for d in dupes:
        report.error(f"Duplicate conversation ID: {d}")


def _check_message_id_uniqueness(
    conversations: list[Conversation], report: ValidationReport
) -> None:
    all_msg_ids: list[str] = []
    for conv in conversations:
        for msg in conv.messages:
            all_msg_ids.append(msg.message_id)
    dupes = [mid for mid, count in Counter(all_msg_ids).items() if count > 1]
    for d in dupes:
        report.error(f"Duplicate message ID: {d}")


def _check_timestamp_ordering(
    conversations: list[Conversation], report: ValidationReport
) -> None:
    for conv in conversations:
        for i in range(1, len(conv.messages)):
            if conv.messages[i].timestamp < conv.messages[i - 1].timestamp:
                report.warn(
                    f"Timestamp ordering issue in {conv.conversation_id}: "
                    f"message {conv.messages[i].message_id} is before {conv.messages[i - 1].message_id}"
                )


def _check_cross_reference_integrity(
    conversations: list[Conversation], report: ValidationReport
) -> None:
    conv_ids = {c.conversation_id for c in conversations}
    for conv in conversations:
        for ref in conv.cross_references:
            if ref not in conv_ids:
                report.error(
                    f"Cross-reference in {conv.conversation_id} points to non-existent "
                    f"conversation: {ref}"
                )


def _check_participant_consistency(
    conversations: list[Conversation],
    channel_participants: dict[str, list[str]],
    report: ValidationReport,
) -> None:
    for conv in conversations:
        allowed = set(channel_participants.get(conv.channel, []))
        if not allowed:
            continue
        for msg in conv.messages:
            if msg.author not in allowed:
                report.warn(
                    f"Author '{msg.author}' in {conv.conversation_id} is not in "
                    f"channel '{conv.channel}' participant list"
                )


def _check_query_reference_validity(
    queries: list[RetrievalQuery],
    conv_ids: set[str],
    report: ValidationReport,
) -> None:
    for q in queries:
        for rid in q.relevant_conversation_ids:
            if rid not in conv_ids:
                report.error(
                    f"Query {q.query_id} references non-existent relevant "
                    f"conversation: {rid}"
                )
        for hid in q.hard_negative_ids:
            if hid not in conv_ids:
                report.error(
                    f"Query {q.query_id} references non-existent hard negative: {hid}"
                )
            if hid in q.relevant_conversation_ids:
                report.error(
                    f"Query {q.query_id} has {hid} as both relevant and hard negative"
                )


def _check_query_coverage(
    conversations: list[Conversation],
    queries: list[RetrievalQuery],
    report: ValidationReport,
) -> None:
    """Check that every non-noise conversation is relevant to at least one query."""
    covered = set()
    for q in queries:
        covered.update(q.relevant_conversation_ids)

    # Only warn about conversations with substantial content (>5 messages)
    for conv in conversations:
        if len(conv.messages) > 5 and conv.conversation_id not in covered:
            report.warn(f"Conversation {conv.conversation_id} is not relevant to any query")


def _check_difficulty_distribution(
    queries: list[RetrievalQuery], report: ValidationReport
) -> None:
    by_scenario: dict[str, Counter] = {}
    for q in queries:
        if q.scenario not in by_scenario:
            by_scenario[q.scenario] = Counter()
        by_scenario[q.scenario][q.difficulty] += 1

    for scenario, dist in by_scenario.items():
        total = sum(dist.values())
        if total < 5:
            continue
        for difficulty in ["easy", "medium", "hard"]:
            pct = dist.get(difficulty, 0) / total
            if pct < 0.1:
                report.warn(
                    f"Scenario '{scenario}' has low '{difficulty}' representation: "
                    f"{dist.get(difficulty, 0)}/{total} ({pct:.0%})"
                )


def _check_bm25_solvability(
    queries: list[RetrievalQuery],
    conversations: list[Conversation],
    report: ValidationReport,
) -> None:
    """Check BM25 R@1 per scenario and warn if any exceeds 50%."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        report.warn("rank_bm25 not installed, skipping BM25 solvability check")
        return

    # Build corpus
    doc_ids = [c.conversation_id for c in conversations]
    corpus_texts = [
        "\n".join(f"{msg.author}: {msg.content}" for msg in c.messages)
        for c in conversations
    ]
    tokenized_corpus = [text.lower().split() for text in corpus_texts]

    if not tokenized_corpus:
        return

    bm25 = BM25Okapi(tokenized_corpus)

    # Group queries by scenario
    by_scenario: dict[str, list[RetrievalQuery]] = defaultdict(list)
    for q in queries:
        by_scenario[q.scenario].append(q)

    for scenario, scenario_queries in by_scenario.items():
        hits = 0
        for q in scenario_queries:
            tokenized_query = q.query_text.lower().split()
            scores = bm25.get_scores(tokenized_query)
            top_idx = scores.argmax()
            top_doc_id = doc_ids[top_idx]
            if top_doc_id in set(q.relevant_conversation_ids):
                hits += 1

        total = len(scenario_queries)
        if total == 0:
            continue
        bm25_r1 = hits / total
        logger.info(f"BM25 R@1 for {scenario}: {bm25_r1:.0%} ({hits}/{total})")

        if bm25_r1 > 0.50:
            report.warn(
                f"BM25 solvability too high for '{scenario}': "
                f"R@1 = {bm25_r1:.0%} ({hits}/{total}) — target < 50%"
            )
