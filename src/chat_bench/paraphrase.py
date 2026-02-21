"""Query paraphrasing for robustness evaluation.

Two modes:
- rule_based_paraphrases: deterministic synonym/reorder transforms (no API needed)
- llm_paraphrases: batched Claude API calls for higher-quality paraphrases
"""

from __future__ import annotations

import random
import re

# -- Synonym tables for rule-based paraphrasing --

_SYNONYMS: dict[str, list[str]] = {
    "find": ["locate", "search for", "look up", "retrieve"],
    "show": ["display", "list", "present", "reveal"],
    "get": ["fetch", "obtain", "retrieve", "pull"],
    "about": ["regarding", "concerning", "related to", "on the topic of"],
    "discussion": ["conversation", "thread", "exchange", "dialogue"],
    "conversation": ["discussion", "thread", "exchange", "dialogue"],
    "thread": ["conversation", "discussion", "exchange"],
    "issue": ["problem", "bug", "defect", "concern"],
    "problem": ["issue", "challenge", "difficulty", "concern"],
    "fix": ["resolve", "repair", "patch", "address"],
    "bug": ["defect", "issue", "error", "glitch"],
    "error": ["mistake", "fault", "bug", "failure"],
    "change": ["modify", "update", "alter", "adjust"],
    "update": ["change", "modify", "revise", "refresh"],
    "performance": ["speed", "efficiency", "throughput", "responsiveness"],
    "implement": ["build", "create", "develop", "add"],
    "design": ["plan", "architect", "blueprint", "layout"],
    "team": ["group", "crew", "squad", "developers"],
    "server": ["backend", "service", "node", "instance"],
    "database": ["DB", "datastore", "data layer", "storage"],
    "deploy": ["release", "ship", "push", "roll out"],
    "config": ["configuration", "settings", "setup", "parameters"],
    "configuration": ["config", "settings", "setup", "parameters"],
    "how": ["in what way", "by what means", "what approach"],
    "where": ["in which location", "at what point", "in what place"],
    "when": ["at what time", "on what occasion", "at which point"],
}

_QUERY_PREFIXES = [
    "",
    "Find ",
    "Show me ",
    "I'm looking for ",
    "Search for ",
    "Where is the ",
    "What about ",
    "Can you find ",
    "Looking for ",
    "Help me find ",
]

_DETERMINERS = {
    "the": ["a", "that", "this"],
    "a": ["the", "one", "some"],
    "an": ["the", "one", "some"],
    "this": ["the", "that"],
    "that": ["the", "this"],
}


def rule_based_paraphrases(query: str, n: int = 5, seed: int | None = None) -> list[str]:
    """Generate N rule-based paraphrases of a query.

    Applies random combinations of:
    1. Synonym substitution
    2. Word reordering (swap adjacent pairs)
    3. Query prefix variation
    4. Determiner swaps

    Args:
        query: The original query text.
        n: Number of paraphrases to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        List of N paraphrased queries (may include near-duplicates).
    """
    rng = random.Random(seed)
    paraphrases: list[str] = []

    for _ in range(n):
        text = query
        transforms = rng.sample(
            ["synonym", "reorder", "prefix", "determiner"],
            k=rng.randint(1, 3),
        )

        for transform in transforms:
            if transform == "synonym":
                text = _apply_synonym_swap(text, rng)
            elif transform == "reorder":
                text = _apply_word_reorder(text, rng)
            elif transform == "prefix":
                text = _apply_prefix_variation(text, rng)
            elif transform == "determiner":
                text = _apply_determiner_swap(text, rng)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if text and text != query:
            paraphrases.append(text)

    # Fill up to n if some transforms produced identical text
    attempts = 0
    while len(paraphrases) < n and attempts < n * 3:
        attempts += 1
        text = _apply_synonym_swap(query, rng)
        text = _apply_prefix_variation(text, rng)
        text = re.sub(r"\s+", " ", text).strip()
        if text and text != query and text not in paraphrases:
            paraphrases.append(text)

    return paraphrases[:n]


def _apply_synonym_swap(text: str, rng: random.Random) -> str:
    """Replace one random word with a synonym."""
    words = text.split()
    candidates = [
        (i, w) for i, w in enumerate(words)
        if w.lower().rstrip(".,!?;:") in _SYNONYMS
    ]
    if not candidates:
        return text
    idx, word = rng.choice(candidates)
    clean = word.lower().rstrip(".,!?;:")
    suffix = word[len(clean):]
    replacement = rng.choice(_SYNONYMS[clean])
    # Preserve original capitalization
    if word[0].isupper():
        replacement = replacement[0].upper() + replacement[1:]
    words[idx] = replacement + suffix
    return " ".join(words)


def _apply_word_reorder(text: str, rng: random.Random) -> str:
    """Swap two adjacent words (avoiding first/last)."""
    words = text.split()
    if len(words) < 4:
        return text
    idx = rng.randint(1, len(words) - 3)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)


def _apply_prefix_variation(text: str, rng: random.Random) -> str:
    """Add or replace a query prefix."""
    # Strip existing prefixes
    lower = text.lower()
    for prefix in _QUERY_PREFIXES:
        if prefix and lower.startswith(prefix.lower()):
            text = text[len(prefix):]
            break

    new_prefix = rng.choice(_QUERY_PREFIXES)
    if new_prefix:
        # Lowercase the first char of the remaining text
        if text:
            text = text[0].lower() + text[1:]
        return new_prefix + text
    return text[0].upper() + text[1:] if text else text


def _apply_determiner_swap(text: str, rng: random.Random) -> str:
    """Swap a random determiner."""
    words = text.split()
    candidates = [
        (i, w) for i, w in enumerate(words)
        if w.lower() in _DETERMINERS
    ]
    if not candidates:
        return text
    idx, word = rng.choice(candidates)
    replacement = rng.choice(_DETERMINERS[word.lower()])
    if word[0].isupper():
        replacement = replacement[0].upper() + replacement[1:]
    words[idx] = replacement
    return " ".join(words)


def llm_paraphrases(
    queries: list[str],
    n: int = 5,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, list[str]]:
    """Generate N paraphrases per query using Claude API.

    Args:
        queries: List of original query texts.
        n: Number of paraphrases per query.
        model: Anthropic model ID.

    Returns:
        Dict mapping original query → list of paraphrased queries.
    """
    import json

    import anthropic

    client = anthropic.Anthropic()
    results: dict[str, list[str]] = {}

    # Process in batches of 10 queries
    for batch_start in range(0, len(queries), 10):
        batch = queries[batch_start:batch_start + 10]
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(batch))

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""\
Generate {n} paraphrases for each query below. Each paraphrase must:
- Preserve the original search intent
- Use different vocabulary and phrasing
- Be a natural information-need query

Queries:
{numbered}

Return JSON: {{"paraphrases": {{"1": ["p1", "p2", ...], "2": [...], ...}}}}
Only return JSON, no other text.""",
            }],
        )

        text = response.content[0].text
        data = json.loads(text)
        paraphrases = data.get("paraphrases", {})
        for i, q in enumerate(batch):
            key = str(i + 1)
            results[q] = paraphrases.get(key, [])[:n]

    return results
