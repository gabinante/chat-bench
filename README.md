# ChatBench

A benchmark for evaluating embedding models on chat/conversational retrieval tasks. Built to measure how well models handle threaded, multi-turn conversations from platforms like Slack, Discord, and IRC.

## Why ChatBench?

Chat-format data is rapidly becoming one of the most common formats for retrieval. The rise of AI agents, multi-turn tool use, and conversational interfaces means more and more production data lives in threaded, multi-party conversations — not just in Slack and Discord, but in agent logs, support transcripts, and collaborative workflows. Embedding models need to handle this data well, but existing benchmarks don't test for it.

General-purpose embedding benchmarks (MTEB, BEIR) don't test the unique challenges of chat data:
- **Thread coherence** — messages in the same thread are related even when topically diverse
- **Temporal adjacency** — nearby messages in a conversation are contextually linked
- **Multi-party dynamics** — conversations involve multiple speakers with different roles
- **Informal language** — abbreviations, code snippets, emoji, platform-specific formatting

ChatBench fills this gap with eight retrieval tasks designed for chat data.

## Tasks

ChatBench includes four programmatic tasks derived directly from conversation structure, plus four LLM-generated query tasks that test semantic understanding.

**Programmatic tasks** (ground truth from conversation structure):

| Task | Description | Metric |
|------|-------------|--------|
| **Thread Retrieval** | Given a message, find the correct conversation thread | MRR@10, R@1/5/10 |
| **Response Retrieval** | Given a conversation prefix, find the continuation | MRR@10, R@5 |
| **Summary Matching** | Given a natural language summary, find the matching thread | MRR@10, NDCG@10 |
| **Cross-Platform Transfer** | Thread retrieval on a held-out platform (e.g., Discord) | MRR@10, R@1/5/10 |

**Generated query tasks** (BM25-filtered, deduplicated):

| Task | Description | Metric |
|------|-------------|--------|
| **Topic Retrieval** | Find conversations by topic description | MRR@10, NDCG@10 |
| **Specific Detail** | Find conversations containing a specific detail | MRR@10, NDCG@10 |
| **Cross-Channel** | Find related conversations across channels | MRR@10, NDCG@10 |
| **Thread Discrimination** | Distinguish between semantically similar conversations | MRR@10, NDCG@10 |

## Corpus

- **512 conversations** across 6 channels and 3 platforms (Slack, Discord, IRC)
- **90 seed** conversations, **~250 confounders** (topically similar hard negatives), **180 noise**
- **72 generated queries** (18 per scenario) + programmatic queries from conversation structure
- Average 8.8 messages per conversation

The corpus is synthetically generated using Claude to ensure controlled difficulty and reproducibility. Confounder conversations are purpose-built hard negatives: topically similar to their seed but factually distinct, ensuring the benchmark tests semantic precision rather than surface-level matching. Generated queries are filtered to remove those solvable by BM25, so the benchmark specifically measures what embedding models add beyond lexical retrieval.

## Metrics

All tasks report **MRR@10** as the primary metric. Additional metrics per task:

- **Recall@1, @5, @10** — retrieval coverage at different cutoffs
- **NDCG@10** — ranking quality with graded relevance
- **MAP@10** — mean average precision (BEIR-compatible)
- **Bootstrap 95% confidence intervals** (1000 resamples)
- **Per-difficulty breakdowns** — easy/medium/hard splits for generated query tasks
- **Hard-negative metrics** — how often models rank confounders above the correct conversation
- **Robustness score** — MRR stability across paraphrased queries (PTEB-style)

## Quick Start

```bash
# Install
uv sync

# Evaluate a model
chat-bench evaluate BAAI/bge-base-en-v1.5

# Compare multiple models with built-in baselines
chat-bench compare --models your-model --models BAAI/bge-base-en-v1.5 --include-baselines

# List available tasks
chat-bench list
```

### Python API

```python
from chat_bench.runner import run_evaluation

results = run_evaluation("BAAI/bge-base-en-v1.5")
for r in results:
    print(f"{r.task}: MRR@10={r.mrr_at_10:.3f}")
```

## Baselines

ChatBench ships with 13 built-in baselines spanning classic and modern architectures:

| Model | Dims | Year | Notes |
|-------|------|------|-------|
| all-MiniLM-L6-v2 | 384 | 2021 | Lightweight baseline |
| BGE-base-en-v1.5 | 768 | 2023 | Mid-tier classic |
| BGE-large-en-v1.5 | 1024 | 2023 | Mid-tier classic |
| GTE-base-en-v1.5 | 768 | 2023 | Mid-tier classic |
| Nomic-embed-text-v1.5 | 768 | 2024 | Instruction-following |
| Stella-en-1.5B-v5 | 1024 | 2024 | Instruction-following |
| Snowflake Arctic-embed-l-v2 | 1024 | 2024 | Modern |
| mxbai-embed-large-v1 | 1024 | 2024 | Modern |
| BGE-M3 | 1024 | 2024 | Modern, multilingual |
| ModernBERT-embed-large | 1024 | 2025 | Modern |
| Nomic-embed-text-v2-moe | 768 | 2025 | MoE architecture |
| GTE-Qwen2-1.5B-instruct | 1536 | 2024 | LLM-based |
| E5-Mistral-7B-instruct | 4096 | 2024 | LLM-based |

Plus a **BM25 lexical baseline** for reference.

## Data

The benchmark corpus is synthetically generated. The generation pipeline (Phases A-F) creates seed conversations, confounder hard negatives, noise, cross-references, and retrieval queries using Claude with structured output validation. The pipeline is reproducible and resumable:

```bash
# Regenerate the full corpus (~$15-20, ~150 API calls)
chat-bench generate --no-resume

# Rebuild task files from the corpus
chat-bench build
```

## Contributing

Contributions are welcome. To evaluate your own model, run `chat-bench evaluate your-org/your-model` and open a PR adding your results.

To add a new baseline, edit `src/chat_bench/baselines.py`.

## Citing

If you use ChatBench in your research, please cite:

```bibtex
@software{chatbench2025,
  author = {Abinante, Gabriel},
  title = {ChatBench: A Benchmark for Chat/Conversational Retrieval},
  url = {https://github.com/gabinante/chat-bench},
  year = {2025}
}
```

## License

Apache-2.0
