"""Model registry for ChatBench evaluation."""

from __future__ import annotations

# Models to evaluate against
# Optional fields per model:
#   query_instruction: prefix prepended to queries before encoding
#   doc_instruction: prefix prepended to documents before encoding
#   trust_remote_code: passed to SentenceTransformer() constructor
#   model_kwargs: additional kwargs for model loading
MODELS = {
    # --- Classic / lightweight ---
    "bge-base": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "dims": 768,
        "type": "open",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
    },
    "bge-large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "dims": 1024,
        "type": "open",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
    },
    "gte-base": {
        "model_id": "Alibaba-NLP/gte-base-en-v1.5",
        "dims": 768,
        "type": "open",
        "trust_remote_code": True,
    },
    "minilm": {
        "model_id": "all-MiniLM-L6-v2",
        "dims": 384,
        "type": "open",
    },
    # --- Instruction-following models ---
    "gte-qwen2": {
        "model_id": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "dims": 1536,
        "type": "open",
        "query_instruction": (
            "Instruct: Given a web search query, retrieve relevant"
            " passages that answer the query\nQuery: "
        ),
        "trust_remote_code": True,
    },
    "stella-1.5b": {
        "model_id": "dunzhang/stella_en_1.5B_v5",
        "dims": 1024,
        "type": "open",
        "query_instruction": "s2p_query: ",
        "trust_remote_code": True,
    },
    "nomic-v1.5": {
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "dims": 768,
        "type": "open",
        "query_instruction": "search_query: ",
        "doc_instruction": "search_document: ",
        "trust_remote_code": True,
    },
    # --- Modern (2024-2025) ---
    "arctic-l-v2": {
        "model_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "dims": 1024,
        "type": "open",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "trust_remote_code": True,
    },
    "mxbai-large": {
        "model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "dims": 1024,
        "type": "open",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
    },
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dims": 1024,
        "type": "open",
    },
    "modernbert-large": {
        "model_id": "lightonai/modernbert-embed-large",
        "dims": 1024,
        "type": "open",
    },
    "nomic-v2-moe": {
        "model_id": "nomic-ai/nomic-embed-text-v2-moe",
        "dims": 768,
        "type": "open",
        "query_instruction": "search_query: ",
        "doc_instruction": "search_document: ",
        "trust_remote_code": True,
    },
    "e5-mistral": {
        "model_id": "intfloat/e5-mistral-7b-instruct",
        "dims": 4096,
        "type": "open",
        "query_instruction": (
            "Instruct: Given a web search query, retrieve relevant"
            " passages that answer the query\nQuery: "
        ),
    },
    # --- Lexical baseline ---
    "bm25": {
        "model_id": "bm25",
        "dims": None,
        "type": "lexical",
    },
}


def get_model_config(model_id: str) -> dict | None:
    """Look up model config by model_id. Returns the config dict or None."""
    for config in MODELS.values():
        if config["model_id"] == model_id:
            return config
    return None
