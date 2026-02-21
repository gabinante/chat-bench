# ChatBench

A benchmark for evaluating embedding models on chat/conversational retrieval tasks. Built to measure how well models handle threaded, multi-turn conversations from platforms like Slack, Discord, and IRC.

## Why ChatBench?

Chat-format data is rapidly becoming one of the most common formats for retrieval. The rise of AI agents, multi-turn tool use, and conversational interfaces means more and more production data lives in threaded, multi-party conversations — not just in Slack and Discord, but in agent logs, support transcripts, and collaborative workflows. Embedding models need to handle this data well, but existing benchmarks don't test for it.

General-purpose embedding benchmarks (MTEB, BEIR) don't test the unique challenges of chat data:
- **Thread coherence** — messages in the same thread are related even when topically diverse
- **Temporal adjacency** — nearby messages in a conversation are contextually linked
- **Multi-party dynamics** — conversations involve multiple speakers with different roles
- **Informal language** — abbreviations, code snippets, emoji, platform-specific formatting

ChatBench fills this gap with four retrieval tasks designed for chat data.

## Tasks

| Task | Description | Metric |
|------|-------------|--------|
| **Thread Retrieval** | Given a message, find the correct conversation thread | MRR@10, R@1/5/10 |
| **Response Retrieval** | Given a conversation prefix, find the continuation | MRR@10, R@5 |
| **Summary-to-Thread** | Given a natural language description, find the matching thread | MRR@10, NDCG@10 |
| **Cross-Platform Transfer** | Thread retrieval on held-out platform (e.g., Slack) | MRR@10, R@1/5/10 |

## Quick Start

```bash
# Install
uv sync

# Evaluate a model
chat-bench evaluate BAAI/bge-base-en-v1.5

# Compare multiple models
chat-bench compare --models your-model --models BAAI/bge-base-en-v1.5 --include-baselines

# List available tasks
chat-bench list
```

## Data Sources

Benchmark data is derived from publicly available chat datasets:
- Discord conversations (Discord-Dialogues)
- IRC conversations (IRC Disentanglement corpus)
- Slack conversations (open source community exports)

## License

Apache-2.0
