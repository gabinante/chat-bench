"""Download and cache benchmark data from HuggingFace Hub."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO_ID = "GabeA/chat-bench"
TASK_FILES = [
    "thread_retrieval.json",
    "response_retrieval.json",
    "conversation_similarity.json",
    "cross_platform_transfer.json",
    "topic_retrieval.json",
    "specific_detail.json",
    "cross_channel.json",
    "thread_discrimination.json",
]


def get_tasks_dir() -> Path:
    """Download task files from HF Hub and return the local cache directory."""
    # Download first file to determine cache dir
    first = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=f"tasks/{TASK_FILES[0]}",
        repo_type="dataset",
    )
    cache_dir = Path(first).parent

    # Download remaining files
    for fname in TASK_FILES[1:]:
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"tasks/{fname}",
            repo_type="dataset",
        )

    return cache_dir
