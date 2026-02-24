# ChatBench

## Project Goal
Standalone benchmark for evaluating embedding models on chat/conversational retrieval tasks. Companion to the [chat-embed](https://github.com/gabinante/chat-embed) model.

## Repository
- **This repo:** `https://github.com/gabinante/chat-bench`
- **Embedding model repo:** `git@github.com:gabinante/chat-embed.git`

## Architecture
- **Package:** `chat_bench` (src/chat_bench/)
- **CLI:** `chat-bench evaluate`, `chat-bench compare`, `chat-bench list`
- **Tasks:** Thread Retrieval, Response Retrieval, Conversation Similarity, Cross-Platform Transfer
- **Metrics:** MRR@10, Recall@1/5/10, NDCG@10

## Key Commands
```bash
chat-bench evaluate BAAI/bge-base-en-v1.5
chat-bench compare --models model-a --models model-b --include-registry
```

## Package Management
Using `uv` with `pyproject.toml`. Install: `uv sync`
