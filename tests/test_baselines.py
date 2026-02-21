"""Tests for ChatBench baselines configuration."""

from chat_bench.baselines import (
    BASELINES,
    get_baseline_config,
)

REQUIRED_FIELDS = {"model_id", "dims", "type"}


def test_all_baselines_have_required_fields():
    for name, config in BASELINES.items():
        for field in REQUIRED_FIELDS:
            assert field in config, f"Baseline '{name}' missing required field '{field}'"


def test_model_id_is_string():
    for name, config in BASELINES.items():
        assert isinstance(config["model_id"], str), f"'{name}' model_id must be str"


def test_dims_is_int_or_none():
    for name, config in BASELINES.items():
        assert config["dims"] is None or isinstance(config["dims"], int), (
            f"'{name}' dims must be int or None"
        )


def test_type_is_valid():
    valid_types = {"open", "lexical"}
    for name, config in BASELINES.items():
        assert config["type"] in valid_types, f"'{name}' type must be one of {valid_types}"


def test_query_instruction_is_string_when_present():
    for name, config in BASELINES.items():
        if "query_instruction" in config:
            assert isinstance(config["query_instruction"], str), (
                f"'{name}' query_instruction must be str"
            )


def test_doc_instruction_is_string_when_present():
    for name, config in BASELINES.items():
        if "doc_instruction" in config:
            assert isinstance(config["doc_instruction"], str), (
                f"'{name}' doc_instruction must be str"
            )


def test_trust_remote_code_is_bool_when_present():
    for name, config in BASELINES.items():
        if "trust_remote_code" in config:
            assert isinstance(config["trust_remote_code"], bool), (
                f"'{name}' trust_remote_code must be bool"
            )


def test_get_baseline_config_found():
    config = get_baseline_config("BAAI/bge-base-en-v1.5")
    assert config is not None
    assert config["model_id"] == "BAAI/bge-base-en-v1.5"
    assert config["dims"] == 768


def test_get_baseline_config_not_found():
    config = get_baseline_config("nonexistent/model")
    assert config is None


def test_get_baseline_config_with_instructions():
    config = get_baseline_config("nomic-ai/nomic-embed-text-v1.5")
    assert config is not None
    assert "query_instruction" in config
    assert "doc_instruction" in config
    assert config["trust_remote_code"] is True


def test_baseline_count():
    # 8 original + 6 new = 14 total
    assert len(BASELINES) == 14


def test_new_baselines_present():
    expected_new = {
        "arctic-l-v2", "mxbai-large", "bge-m3",
        "modernbert-large", "nomic-v2-moe", "e5-mistral",
    }
    assert expected_new.issubset(set(BASELINES.keys()))
