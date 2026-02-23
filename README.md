[![PyPI](https://img.shields.io/pypi/v/chat-bench)](https://pypi.org/project/chat-bench/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

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
| **Conversation Similarity** | Given a conversation, find the most similar conversation | MRR@10, NDCG@10 |
| **Cross-Platform Transfer** | Thread retrieval on a held-out platform (e.g., Discord) | MRR@10, R@1/5/10 |

**Generated query tasks** (BM25-filtered, deduplicated):

| Task | Description | Metric |
|------|-------------|--------|
| **Topic Retrieval** | Find conversations by topic description | MRR@10, NDCG@10 |
| **Specific Detail** | Find conversations containing a specific detail | MRR@10, NDCG@10 |
| **Cross-Channel** | Find related conversations across channels | MRR@10, NDCG@10 |
| **Thread Discrimination** | Distinguish between semantically similar conversations | MRR@10, NDCG@10 |

## Corpus

- **1,595 conversations** — 512 synthetic + 1,083 real Discord conversations (from [DISCO](https://github.com/google-research/disco))
- **6 channels**, **3 platforms** (Slack, Discord, IRC)
- **90 seed** conversations, **~250 confounders** (topically similar hard negatives), **180 noise** (synthetic), plus real Discord conversations at 50/channel
- Generated queries (200+ per scenario) + programmatic queries from conversation structure
- Average 8.8 messages per conversation

The corpus combines synthetic and real conversations. The synthetic portion is generated using Claude to ensure controlled difficulty and reproducibility. Confounder conversations are purpose-built hard negatives: topically similar to their seed but factually distinct, ensuring the benchmark tests semantic precision rather than surface-level matching. The real portion comes from the [DISCO dataset](https://github.com/google-research/disco) (Ekstedt & Malmqvist, 2023), a curated collection of Discord server conversations that adds authentic conversational patterns, informal language, and natural topic drift. Generated queries are filtered to remove those solvable by BM25, so the benchmark specifically measures what embedding models add beyond lexical retrieval.

## Metrics

All tasks report **MRR@10** as the primary metric. Additional metrics per task:

- **Recall@1, @5, @10** — retrieval coverage at different cutoffs
- **NDCG@10** — ranking quality with graded relevance
- **MAP@10** — mean average precision (BEIR-compatible)
- **Bootstrap 95% confidence intervals** (1000 resamples)
- **Per-difficulty breakdowns** — easy/medium/hard splits for generated query tasks
- **Hard-negative metrics** — how often models rank confounders above the correct conversation
- **Robustness score** — MRR stability across paraphrased queries (PTEB-style)

## Installation

```bash
# From PyPI
pip install chat-bench

# With optional dependencies
pip install 'chat-bench[viz]'       # matplotlib, seaborn, tabulate
pip install 'chat-bench[mteb]'      # MTEB integration
pip install 'chat-bench[generate]'  # corpus generation (requires Anthropic API key)

# Development install
git clone https://github.com/gabinante/chat-bench.git
cd chat-bench
uv sync --dev
```

Requires **Python >= 3.11**.

## Quick Start

```bash
# Evaluate a model
chat-bench evaluate BAAI/bge-base-en-v1.5

# Compare multiple models with built-in baselines
chat-bench compare --models your-model --models BAAI/bge-base-en-v1.5 --include-baselines

# BM25 lexical baseline
chat-bench evaluate --bm25

# List available tasks
chat-bench list
```

### Python API

```python
from pathlib import Path
from chat_bench.runner import evaluate_task, load_task, print_results_table
from chat_bench.data import get_tasks_dir

# Load tasks (auto-downloads from HuggingFace Hub)
tasks_dir = get_tasks_dir()

results = []
for task_path in sorted(tasks_dir.glob("*.json")):
    task = load_task(task_path)
    result = evaluate_task(None, task, model_name="bm25", use_bm25=True)
    results.append(result)

print_results_table(results)
```

### HuggingFace Hub

All task datasets are published on HuggingFace Hub in MTEB-compatible format:

```python
from datasets import load_dataset

# Load a specific task dataset
corpus = load_dataset("GabeA/chatbench-thread-retrieval", "corpus", split="test")
queries = load_dataset("GabeA/chatbench-thread-retrieval", "queries", split="test")
qrels = load_dataset("GabeA/chatbench-thread-retrieval", split="test")
```

Available datasets:
- `GabeA/chatbench-thread-retrieval`
- `GabeA/chatbench-response-retrieval`
- `GabeA/chatbench-conversation-similarity`
- `GabeA/chatbench-cross-platform-transfer`
- `GabeA/chatbench-topic-retrieval`
- `GabeA/chatbench-specific-detail`
- `GabeA/chatbench-cross-channel`
- `GabeA/chatbench-thread-discrimination`

### MTEB Integration

ChatBench tasks integrate with [MTEB](https://github.com/embeddings-benchmark/mteb) for standardized evaluation:

```python
import mteb
from chat_bench.mteb_tasks import CHATBENCH_TASKS

# Evaluate with MTEB
evaluation = mteb.MTEB(tasks=[cls() for cls in CHATBENCH_TASKS])
results = evaluation.run(model)
```

Requires `pip install 'chat-bench[mteb]'`.

## Baselines

ChatBench ships with 13 built-in baselines spanning classic and modern architectures:

| Model | Dims | Year | Notes |
|-------|------|------|-------|
| all-MiniLM-L6-v2 | 384 | 2021 | Lightweight baseline |
| BGE-base-en-v1.5 | 768 | 2023 | Mid-tier classic |
| BGE-large-en-v1.5 | 1024 | 2023 | Mid-tier classic |
| GTE-base-en-v1.5 | 768 | 2024 | Mid-tier classic |
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

## Data Generation

The benchmark corpus is a mix of synthetic conversations and real Discord conversations from the [DISCO dataset](https://github.com/google-research/disco). The synthetic generation pipeline (Phases A-F) creates seed conversations, confounder hard negatives, noise, cross-references, and retrieval queries using Claude with structured output validation. The pipeline is reproducible and resumable:

```bash
# Regenerate the full corpus (~$15-20, ~150 API calls)
chat-bench generate --no-resume

# Rebuild task files from the corpus (with DISCO real conversations)
chat-bench build --include-disco --disco-max-per-channel 50
```

## Data Contamination

ChatBench is designed to minimize training data contamination concerns:

- **Synthetic corpus is unique.** The 512 synthetic conversations were generated specifically for this benchmark using Claude with controlled prompts. These conversations do not exist anywhere on the public internet and cannot appear in any model's pretraining data. Each regeneration produces different conversations, so researchers can regenerate a fresh corpus if contamination is ever suspected.

- **DISCO conversations are from a research dataset.** The 1,083 real Discord conversations come from the [DISCO dataset](https://github.com/google-research/disco) (Ekstedt & Malmqvist, 2023), a curated research corpus. While these are real conversations, their inclusion in a retrieval benchmark with specific query-document pairings is novel — the retrieval task framing is unique to ChatBench.

- **Queries are generated and filtered.** The 2,000+ retrieval queries are LLM-generated with BM25 filtering, making them unlikely to appear verbatim in training corpora. Programmatic queries (thread retrieval, response retrieval) are derived from conversation structure and don't exist as standalone text.

- **Reproducible regeneration.** The full corpus can be regenerated (`chat-bench generate --no-resume`) to produce an entirely fresh set of conversations and queries, providing a straightforward mitigation if contamination is ever a concern.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

To evaluate your own model, run `chat-bench evaluate your-org/your-model` and open a PR adding your results.

To add a new baseline, edit `src/chat_bench/baselines.py`.

## Citing

If you use ChatBench in your research, please cite:

```bibtex
@software{chatbench2026,
  author = {Abinante, Gabriel},
  title = {ChatBench: A Benchmark for Chat/Conversational Retrieval},
  url = {https://github.com/gabinante/chat-bench},
  year = {2026}
}
```

## License

Apache-2.0
