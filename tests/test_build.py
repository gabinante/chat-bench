"""Tests for build pipeline: platform backfill and programmatic task builders."""

from chat_bench.build import _backfill_platforms
from chat_bench.generate.schemas import Conversation, Message
from chat_bench.tasks.cross_platform import build_cross_platform_task
from chat_bench.tasks.response_retrieval import build_response_retrieval_task


def _make_conversation(conv_id: str, channel: str, n_messages: int = 8) -> Conversation:
    """Create a minimal Conversation for testing."""
    messages = [
        Message(
            message_id=f"{conv_id}_msg{i}",
            author=f"user_{i % 3}",
            timestamp=f"2025-01-01T00:0{i}:00Z",
            content=f"Message {i} in {channel}",
        )
        for i in range(n_messages)
    ]
    return Conversation(
        conversation_id=conv_id,
        channel=channel,
        title=f"Test conversation {conv_id}",
        messages=messages,
    )


def _conv_to_dict(conv: Conversation) -> dict:
    """Convert a Conversation to the dict format used by task builders."""
    return {
        "id": conv.conversation_id,
        "source": conv.platform,
        "messages": [f"{m.author}: {m.content}" for m in conv.messages],
    }


# --- _backfill_platforms tests ---


def test_backfill_platforms_sets_from_channel():
    convs = [
        _make_conversation("c1", "engineering"),
        _make_conversation("c2", "game_design"),
        _make_conversation("c3", "lore_narrative"),
    ]
    # All default to "slack" before backfill
    assert all(c.platform == "slack" for c in convs)

    _backfill_platforms(convs)

    assert convs[0].platform == "slack"      # engineering -> slack
    assert convs[1].platform == "discord"    # game_design -> discord
    assert convs[2].platform == "irc"        # lore_narrative -> irc


def test_backfill_platforms_unknown_channel_unchanged():
    conv = _make_conversation("c1", "nonexistent_channel")
    _backfill_platforms([conv])
    # Unknown channel -> platform stays at default "slack"
    assert conv.platform == "slack"


# --- response_retrieval task builder tests ---


def test_response_retrieval_produces_valid_task():
    convs = [_make_conversation(f"c{i}", "engineering", n_messages=10) for i in range(5)]
    dicts = [_conv_to_dict(c) for c in convs]

    task = build_response_retrieval_task(
        dicts, prefix_size=3, response_size=3, num_queries=3, seed=42,
    )

    assert task.task_id == "response_retrieval"
    assert len(task.queries) == 3
    assert len(task.corpus) == 5  # all eligible conversations form the corpus
    # Each query should reference exactly one relevant doc
    for q in task.queries:
        assert len(q.relevant_doc_ids) == 1
        assert q.relevant_doc_ids[0].startswith("resp_")


def test_response_retrieval_skips_short_conversations():
    short = _make_conversation("short", "engineering", n_messages=2)
    long = _make_conversation("long", "engineering", n_messages=10)
    dicts = [_conv_to_dict(short), _conv_to_dict(long)]

    task = build_response_retrieval_task(
        dicts, prefix_size=3, response_size=3, num_queries=10, seed=42,
    )

    # Only the long conversation is eligible
    assert len(task.corpus) == 1
    assert len(task.queries) == 1


# --- cross_platform task builder tests ---


def test_cross_platform_produces_valid_task():
    _backfill_platforms_for = {
        "engineering": "slack",
        "game_design": "discord",
        "lore_narrative": "irc",
    }
    convs = []
    for i, (channel, platform) in enumerate(_backfill_platforms_for.items()):
        for j in range(4):
            c = _make_conversation(f"c{i}_{j}", channel, n_messages=8)
            c.platform = platform
            convs.append(c)

    dicts = [_conv_to_dict(c) for c in convs]

    task = build_cross_platform_task(
        dicts, held_out_platform="irc", num_queries=100, seed=42,
    )

    assert task.task_id == "cross_platform_transfer"
    # Corpus includes all conversations with >= 5 messages
    assert len(task.corpus) == 12
    # Queries come only from the held-out platform (irc)
    assert len(task.queries) == 4
    for q in task.queries:
        assert q.metadata["platform"] == "irc"


def test_cross_platform_no_held_out_conversations():
    # All conversations are slack, but held_out is "irc" -> 0 queries
    convs = [_make_conversation(f"c{i}", "engineering", n_messages=8) for i in range(5)]
    for c in convs:
        c.platform = "slack"
    dicts = [_conv_to_dict(c) for c in convs]

    task = build_cross_platform_task(
        dicts, held_out_platform="irc", num_queries=10, seed=42,
    )

    assert len(task.queries) == 0
    assert len(task.corpus) == 5
