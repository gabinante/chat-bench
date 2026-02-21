"""Tests for generation pipeline: fresh-run cleanup logic."""

from chat_bench.generate.pipeline import _clear_generated_data


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
