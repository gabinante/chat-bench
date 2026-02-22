"""Tests for generation pipeline and task builders."""

from chat_bench.build import _build_confounder_map_from_corpus
from chat_bench.generate.pipeline import (
    _assign_ids,
    _assign_query_ids,
    _build_confounder_map,
    _clear_generated_data,
)
from chat_bench.generate.schemas import Conversation, Message, RetrievalQuery
from chat_bench.tasks.conversation_similarity import build_conversation_similarity_task


def test_clear_generated_data_removes_files(tmp_path, monkeypatch):
    """_clear_generated_data removes corpus, query, and stats files."""
    corpus = tmp_path / "corpus" / "conversations.jsonl"
    queries_dir = tmp_path / "queries"
    stats = tmp_path / "metadata" / "statistics.json"

    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"id": "test"}\n')

    queries_dir.mkdir(parents=True)
    (queries_dir / "topic_retrieval.jsonl").write_text('{"q": 1}\n')
    (queries_dir / "cross_channel.jsonl").write_text('{"q": 2}\n')

    stats.parent.mkdir(parents=True)
    stats.write_text('{"total": 1}')

    monkeypatch.setattr("chat_bench.generate.pipeline._CORPUS_PATH", corpus)
    monkeypatch.setattr("chat_bench.generate.pipeline._QUERIES_DIR", queries_dir)
    monkeypatch.setattr("chat_bench.generate.pipeline._STATS_PATH", stats)

    _clear_generated_data()

    assert not corpus.exists()
    assert list(queries_dir.glob("*.jsonl")) == []
    assert not stats.exists()


def test_clear_generated_data_no_error_when_missing(tmp_path, monkeypatch):
    """_clear_generated_data is safe when files don't exist."""
    corpus = tmp_path / "corpus" / "conversations.jsonl"
    queries_dir = tmp_path / "queries"
    stats = tmp_path / "metadata" / "statistics.json"

    # Directories exist but files do not
    corpus.parent.mkdir(parents=True)
    queries_dir.mkdir(parents=True)
    stats.parent.mkdir(parents=True)

    monkeypatch.setattr("chat_bench.generate.pipeline._CORPUS_PATH", corpus)
    monkeypatch.setattr("chat_bench.generate.pipeline._QUERIES_DIR", queries_dir)
    monkeypatch.setattr("chat_bench.generate.pipeline._STATS_PATH", stats)

    _clear_generated_data()  # should not raise


def _make_conv(channel, conv_id, n_messages=3):
    """Helper to create a Conversation with dummy messages."""
    messages = [
        Message(
            message_id=f"{conv_id}_msg_{i}",
            author="alice",
            timestamp=f"2025-01-01T10:{i:02d}:00Z",
            content=f"Message {i}",
        )
        for i in range(1, n_messages + 1)
    ]
    return Conversation(
        conversation_id=conv_id,
        channel=channel,
        title="Test",
        messages=messages,
    )


def test_assign_ids_unique_across_batch(tmp_path, monkeypatch):
    """_assign_ids assigns unique sequential IDs within a batch."""
    monkeypatch.setattr(
        "chat_bench.generate.pipeline._CORPUS_PATH",
        tmp_path / "conversations.jsonl",
    )

    convs = [
        _make_conv("engineering", "engineering_001"),
        _make_conv("engineering", "engineering_001"),
        _make_conv("engineering", "engineering_002"),
    ]
    _assign_ids(convs)

    ids = [c.conversation_id for c in convs]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"
    assert ids == ["engineering_001", "engineering_002", "engineering_003"]


def test_assign_ids_continues_from_existing_corpus(tmp_path, monkeypatch):
    """_assign_ids reads existing corpus to avoid collisions."""
    corpus = tmp_path / "conversations.jsonl"
    # Write 2 existing conversations
    existing = _make_conv("engineering", "engineering_001")
    corpus.write_text(existing.model_dump_json() + "\n")

    monkeypatch.setattr("chat_bench.generate.pipeline._CORPUS_PATH", corpus)

    convs = [_make_conv("engineering", "engineering_001")]
    _assign_ids(convs)

    assert convs[0].conversation_id == "engineering_002"


def test_assign_ids_assigns_message_ids(tmp_path, monkeypatch):
    """_assign_ids also reassigns message_id fields."""
    monkeypatch.setattr(
        "chat_bench.generate.pipeline._CORPUS_PATH",
        tmp_path / "conversations.jsonl",
    )

    conv = _make_conv("art_direction", "whatever_999", n_messages=2)
    _assign_ids([conv])

    assert conv.conversation_id == "art_direction_001"
    assert conv.messages[0].message_id == "art_direction_001_msg_001"
    assert conv.messages[1].message_id == "art_direction_001_msg_002"


def test_assign_ids_multiple_channels(tmp_path, monkeypatch):
    """_assign_ids tracks counters independently per channel."""
    monkeypatch.setattr(
        "chat_bench.generate.pipeline._CORPUS_PATH",
        tmp_path / "conversations.jsonl",
    )

    convs = [
        _make_conv("engineering", "engineering_001"),
        _make_conv("general", "general_001"),
        _make_conv("engineering", "engineering_001"),
    ]
    _assign_ids(convs)

    assert convs[0].conversation_id == "engineering_001"
    assert convs[1].conversation_id == "general_001"
    assert convs[2].conversation_id == "engineering_002"


def test_assign_query_ids_sequential():
    """_assign_query_ids assigns unique sequential IDs."""
    queries = [
        RetrievalQuery(
            query_id="topic_retrieval_001",
            query_text=f"query {i}",
            scenario="topic_retrieval",
            relevant_conversation_ids=["eng_001"],
        )
        for i in range(5)
    ]
    _assign_query_ids(queries, "topic_retrieval")

    ids = [q.query_id for q in queries]
    assert ids == [
        "topic_retrieval_001",
        "topic_retrieval_002",
        "topic_retrieval_003",
        "topic_retrieval_004",
        "topic_retrieval_005",
    ]


def test_assign_query_ids_replaces_duplicates():
    """_assign_query_ids fixes duplicate IDs from multiple batches."""
    # Simulate two batches both generating _001, _002
    queries = [
        RetrievalQuery(
            query_id="topic_retrieval_001",
            query_text="batch 1 query 1",
            scenario="topic_retrieval",
            relevant_conversation_ids=["eng_001"],
        ),
        RetrievalQuery(
            query_id="topic_retrieval_002",
            query_text="batch 1 query 2",
            scenario="topic_retrieval",
            relevant_conversation_ids=["eng_002"],
        ),
        RetrievalQuery(
            query_id="topic_retrieval_001",
            query_text="batch 2 query 1",
            scenario="topic_retrieval",
            relevant_conversation_ids=["eng_003"],
        ),
    ]
    _assign_query_ids(queries, "topic_retrieval")

    ids = [q.query_id for q in queries]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"


def test_build_confounder_map_uses_phase_field():
    """_build_confounder_map uses phase='seed'/'confounder' tags."""
    convs = [
        _make_conv("engineering", "engineering_001"),
        _make_conv("engineering", "engineering_002"),
        _make_conv("engineering", "engineering_003"),
        _make_conv("engineering", "engineering_004"),
        _make_conv("engineering", "engineering_005"),
        _make_conv("engineering", "engineering_006"),
        _make_conv("engineering", "engineering_090"),  # noise
    ]
    # Tag phases explicitly
    convs[0].phase = "seed"
    convs[1].phase = "seed"
    convs[2].phase = "confounder"
    convs[3].phase = "confounder"
    convs[4].phase = "confounder"
    convs[5].phase = "confounder"
    convs[6].phase = "noise"

    # With CONFOUNDERS_PER_SEED=3 (from pipeline constants):
    # seed_001 -> [confounder_003, confounder_004, confounder_005]
    # seed_002 -> [confounder_006]
    from chat_bench.generate import pipeline
    original = pipeline.CONFOUNDERS_PER_SEED
    try:
        pipeline.CONFOUNDERS_PER_SEED = 2
        cmap = _build_confounder_map(convs)
    finally:
        pipeline.CONFOUNDERS_PER_SEED = original

    assert "engineering_001" in cmap
    assert "engineering_002" in cmap
    assert cmap["engineering_001"] == ["engineering_003", "engineering_004"]
    assert cmap["engineering_002"] == ["engineering_005", "engineering_006"]
    # Noise should not appear
    assert "engineering_090" not in cmap
    for conf_list in cmap.values():
        assert "engineering_090" not in conf_list


# -- Conversation similarity task tests --


def _make_conv_dict(conv_id, source="slack", n_messages=5):
    """Helper to create a conversation dict for task builders."""
    return {
        "id": conv_id,
        "source": source,
        "messages": [f"author_{i}: Message {i} about {conv_id}" for i in range(n_messages)],
    }


def test_conversation_similarity_basic():
    """build_conversation_similarity_task produces correct queries from seed-confounder pairs."""
    convs = [
        _make_conv_dict("eng_001"),
        _make_conv_dict("eng_002"),
        _make_conv_dict("eng_003"),
        _make_conv_dict("eng_004"),
        _make_conv_dict("eng_005"),
    ]
    confounder_map = {
        "eng_001": ["eng_002", "eng_003"],
        "eng_004": ["eng_005"],
    }

    task = build_conversation_similarity_task(convs, confounder_map)

    assert task.task_id == "conversation_similarity"
    assert len(task.queries) == 2
    assert len(task.corpus) == 5

    # Check query for eng_001
    q1 = next(q for q in task.queries if q.metadata.get("seed_id") == "eng_001")
    assert set(q1.relevant_doc_ids) == {"eng_002", "eng_003"}

    # Check query for eng_004
    q2 = next(q for q in task.queries if q.metadata.get("seed_id") == "eng_004")
    assert q2.relevant_doc_ids == ["eng_005"]


def test_conversation_similarity_seed_not_in_relevant():
    """Seed conversation is in corpus but NOT in relevant_doc_ids."""
    convs = [
        _make_conv_dict("seed_001"),
        _make_conv_dict("conf_001"),
        _make_conv_dict("conf_002"),
    ]
    confounder_map = {"seed_001": ["conf_001", "conf_002"]}

    task = build_conversation_similarity_task(convs, confounder_map)

    q = task.queries[0]
    corpus_ids = {d.doc_id for d in task.corpus}
    assert "seed_001" in corpus_ids
    assert "seed_001" not in q.relevant_doc_ids


def test_conversation_similarity_skips_missing_confounders():
    """Confounders not in corpus are excluded from relevant_doc_ids."""
    convs = [
        _make_conv_dict("seed_001"),
        _make_conv_dict("conf_001"),
    ]
    confounder_map = {"seed_001": ["conf_001", "conf_missing"]}

    task = build_conversation_similarity_task(convs, confounder_map)

    q = task.queries[0]
    assert q.relevant_doc_ids == ["conf_001"]


def test_conversation_similarity_skips_seeds_without_valid_confounders():
    """Seeds with no valid confounders in corpus produce no queries."""
    convs = [_make_conv_dict("seed_001")]
    confounder_map = {"seed_001": ["conf_missing"]}

    task = build_conversation_similarity_task(convs, confounder_map)

    assert len(task.queries) == 0


def test_conversation_similarity_disco_in_corpus():
    """DISCO conversations appear in corpus as negatives."""
    convs = [
        _make_conv_dict("eng_001"),
        _make_conv_dict("eng_002"),
        _make_conv_dict("disco_python_general_42", source="discord"),
    ]
    confounder_map = {"eng_001": ["eng_002"]}

    task = build_conversation_similarity_task(convs, confounder_map)

    corpus_ids = {d.doc_id for d in task.corpus}
    assert "disco_python_general_42" in corpus_ids
    # DISCO conv is not a relevant doc for any query
    for q in task.queries:
        assert "disco_python_general_42" not in q.relevant_doc_ids


def test_build_confounder_map_from_corpus():
    """_build_confounder_map_from_corpus reads phase and confounder_for fields."""
    convs = [
        _make_conv("engineering", "eng_001", n_messages=5),
        _make_conv("engineering", "eng_002", n_messages=5),
        _make_conv("engineering", "eng_003", n_messages=5),
        _make_conv("engineering", "eng_004", n_messages=5),
    ]
    convs[0].phase = "seed"
    convs[1].phase = "confounder"
    convs[1].confounder_for = "eng_001"
    convs[2].phase = "confounder"
    convs[2].confounder_for = "eng_001"
    convs[3].phase = "noise"

    cmap = _build_confounder_map_from_corpus(convs)

    assert cmap == {"eng_001": ["eng_002", "eng_003"]}
