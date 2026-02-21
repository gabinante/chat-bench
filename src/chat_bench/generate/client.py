"""Anthropic API wrapper with retry, validation, and usage tracking."""

from __future__ import annotations

import json
import logging
import re
import time

from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_OUTPUT_TOKENS = 16384


class UsageStats(BaseModel):
    """Track token usage and estimated cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        # Sonnet pricing: $3/M input, $15/M output
        return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000


class GenerationClient:
    """Wraps Anthropic API with retry, JSON extraction, and Pydantic validation."""

    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 3):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for corpus generation. "
                "Install it with: pip install 'chat-bench[generate]'"
            )
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries
        self.usage = UsageStats()

    def generate(
        self,
        system: str,
        user_prompt: str,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = 1.0,
    ) -> tuple[str, str]:
        """Send a message to Claude and return (text, stop_reason).

        Retries on rate limits and transient errors with exponential backoff.
        """
        import anthropic

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                self.usage.api_calls += 1
                self.usage.input_tokens += response.usage.input_tokens
                self.usage.output_tokens += response.usage.output_tokens

                return response.content[0].text, response.stop_reason

            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited. Waiting {wait}s before retry {attempt + 1}/{self.max_retries}")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Server error {e.status_code}. Waiting {wait}s before retry")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Failed after {self.max_retries} retries")

    def generate_json(
        self,
        system: str,
        user_prompt: str,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = 1.0,
    ) -> dict | list:
        """Generate a response and parse as JSON.

        Handles markdown code blocks and truncated output (stop_reason=max_tokens).
        """
        text, stop_reason = self.generate(system, user_prompt, max_tokens, temperature)

        if stop_reason == "max_tokens":
            logger.warning("Response truncated (max_tokens). Attempting JSON repair.")

        return self._extract_json(text, truncated=(stop_reason == "max_tokens"))

    def generate_validated(
        self,
        system: str,
        user_prompt: str,
        model_class: type[BaseModel],
        *,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = 1.0,
        wrap_key: str | None = None,
    ) -> BaseModel | list[BaseModel]:
        """Generate, parse JSON, and validate against a Pydantic model.

        Args:
            wrap_key: If set, expects the JSON to have this key containing the data.
                      e.g., wrap_key="conversations" expects {"conversations": [...]}
        """
        data = self.generate_json(system, user_prompt, max_tokens, temperature)

        if wrap_key and isinstance(data, dict):
            data = data[wrap_key]

        if isinstance(data, list):
            return [model_class.model_validate(item) for item in data]
        return model_class.model_validate(data)

    @staticmethod
    def _extract_json(text: str, truncated: bool = False) -> dict | list:
        """Extract JSON from a response that may contain markdown code blocks."""
        stripped = text.strip()

        # Remove markdown code block wrapper if present
        cleaned = stripped
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        # Remove trailing ``` if present
        if "```" in cleaned:
            cleaned = cleaned[:cleaned.rindex("```")]
        cleaned = cleaned.strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # If truncated, try to repair by finding complete items in an array
        if truncated:
            repaired = _repair_truncated_json(cleaned)
            if repaired is not None:
                return repaired

        # Try finding first [ or { and attempt parse with repair
        for i, ch in enumerate(cleaned):
            if ch in "[{":
                fragment = cleaned[i:]
                try:
                    return json.loads(fragment)
                except json.JSONDecodeError:
                    if truncated:
                        repaired = _repair_truncated_json(fragment)
                        if repaired is not None:
                            return repaired

        raise ValueError(f"Could not extract JSON from response:\n{text[:500]}")


def _repair_truncated_json(text: str) -> dict | list | None:
    """Attempt to repair truncated JSON by extracting complete objects from an array.

    When a response hits max_tokens, we often get a valid JSON array with the last
    object cut off. This function extracts all complete objects.
    """
    # Strategy: find the wrapper structure and extract complete items

    # Case 1: {"conversations": [{...}, {..TRUNCATED
    # Find the array start and extract complete objects
    wrapper_match = re.match(r'\s*\{\s*"(\w+)"\s*:\s*\[', text)
    if wrapper_match:
        key = wrapper_match.group(1)
        array_start = wrapper_match.end() - 1  # position of [
        items = _extract_complete_objects(text[array_start:])
        if items:
            return {key: items}

    # Case 2: [{...}, {..TRUNCATED
    if text.lstrip().startswith("["):
        items = _extract_complete_objects(text.lstrip())
        if items:
            return items

    return None


def _extract_complete_objects(text: str) -> list | None:
    """Extract complete JSON objects from a potentially truncated array."""
    if not text.startswith("["):
        return None

    items = []
    depth = 0
    in_string = False
    escape = False
    obj_start = None

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 1 and obj_start is None:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 1 and obj_start is not None:
                obj_text = text[obj_start:i + 1]
                try:
                    items.append(json.loads(obj_text))
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == "[" and depth == 0:
            depth = 1

    return items if items else None
