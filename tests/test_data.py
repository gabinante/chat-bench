"""Tests for HuggingFace Hub data loader."""

import pytest


@pytest.mark.network
def test_get_tasks_dir_returns_json_files():
    from chat_bench.data import get_tasks_dir

    tasks_dir = get_tasks_dir()
    assert tasks_dir.is_dir()
    json_files = list(tasks_dir.glob("*.json"))
    assert len(json_files) >= 8
