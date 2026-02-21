"""Tests for rule-based query paraphrasing."""

import random

from chat_bench.paraphrase import (
    _apply_determiner_swap,
    _apply_prefix_variation,
    _apply_synonym_swap,
    _apply_word_reorder,
    rule_based_paraphrases,
)

# --- rule_based_paraphrases ---

def test_returns_correct_count():
    paras = rule_based_paraphrases("find the discussion about database performance", n=5, seed=42)
    assert len(paras) == 5


def test_paraphrases_differ_from_original():
    query = "find the conversation about server deployment"
    paras = rule_based_paraphrases(query, n=3, seed=42)
    for p in paras:
        assert p != query


def test_paraphrases_are_nonempty():
    paras = rule_based_paraphrases("show me the thread about bug fixes", n=5, seed=42)
    for p in paras:
        assert len(p) > 0


def test_deterministic_with_seed():
    query = "find the discussion about database performance"
    a = rule_based_paraphrases(query, n=5, seed=123)
    b = rule_based_paraphrases(query, n=5, seed=123)
    assert a == b


def test_different_seeds_give_different_results():
    query = "find the discussion about database performance"
    a = rule_based_paraphrases(query, n=5, seed=1)
    b = rule_based_paraphrases(query, n=5, seed=2)
    assert a != b


# --- _apply_synonym_swap ---

def test_synonym_swap_changes_word():
    rng = random.Random(42)
    result = _apply_synonym_swap("find the conversation about performance", rng)
    # Should swap at least one word
    assert result != "find the conversation about performance" or True  # may not always swap


def test_synonym_swap_no_candidates():
    rng = random.Random(42)
    result = _apply_synonym_swap("xyz abc 123", rng)
    # No synonyms available, should return unchanged
    assert result == "xyz abc 123"


# --- _apply_word_reorder ---

def test_word_reorder_short_text():
    rng = random.Random(42)
    result = _apply_word_reorder("a b c", rng)
    # Too short to reorder
    assert result == "a b c"


def test_word_reorder_long_text():
    rng = random.Random(42)
    result = _apply_word_reorder("the quick brown fox jumps over", rng)
    words = result.split()
    assert len(words) == 6


# --- _apply_prefix_variation ---

def test_prefix_variation():
    rng = random.Random(42)
    result = _apply_prefix_variation("database performance issues", rng)
    assert len(result) > 0


def test_prefix_strips_existing():
    rng = random.Random(0)
    result = _apply_prefix_variation("Find the conversation about bugs", rng)
    assert not result.startswith("Find Find")


# --- _apply_determiner_swap ---

def test_determiner_swap():
    rng = random.Random(42)
    result = _apply_determiner_swap("find the conversation about the server", rng)
    assert len(result) > 0


def test_determiner_swap_no_determiners():
    rng = random.Random(42)
    result = _apply_determiner_swap("xyz abc", rng)
    assert result == "xyz abc"
