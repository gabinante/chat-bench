"""Baseline models for ChatBench comparison."""

from __future__ import annotations

# Models to evaluate against
BASELINES = {
    "bge-base": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "dims": 768,
        "type": "open",
    },
    "bge-large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "dims": 1024,
        "type": "open",
    },
    "gte-base": {
        "model_id": "thenlper/gte-base-en-v1.5",
        "dims": 768,
        "type": "open",
    },
    "minilm": {
        "model_id": "all-MiniLM-L6-v2",
        "dims": 384,
        "type": "open",
    },
}
