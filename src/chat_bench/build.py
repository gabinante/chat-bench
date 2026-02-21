"""Transform raw corpus into BenchmarkTask JSON files for evaluation."""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from rich.console import Console

from .generate.schemas import Conversation, RetrievalQuery
from .schemas import BenchmarkCorpusDoc, BenchmarkQuery, BenchmarkTask
from .tasks.cross_platform import build_cross_platform_task
from .tasks.response_retrieval import build_response_retrieval_task
from .tasks.summary_matching import build_summary_matching_task
from .tasks.thread_retrieval import build_thread_retrieval_task

console = Console()
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def build_all_tasks(
    corpus_dir: str | Path | None = None,
    queries_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> None:
    """Build all benchmark tasks from the generated corpus.

    Creates 6 task JSON files + dev/test splits.
    """
    corpus_path = Path(corpus_dir) if corpus_dir else _DATA_DIR / "corpus"
    queries_path = Path(queries_dir) if queries_dir else _DATA_DIR / "queries"
    output_path = Path(output_dir) if output_dir else _DATA_DIR / "tasks"
    splits_path = _DATA_DIR / "splits"

    output_path.mkdir(parents=True, exist_ok=True)
    splits_path.mkdir(parents=True, exist_ok=True)

    # Load corpus
    conversations = _load_conversations(corpus_path / "conversations.jsonl")
    if not conversations:
        console.print("[red]No conversations found. Run 'chat-bench generate' first.[/]")
        return

    console.print(f"Loaded {len(conversations)} conversations")

    # Load queries
    all_queries: dict[str, list[RetrievalQuery]] = {}
    for scenario_file in queries_path.glob("*.jsonl"):
        scenario = scenario_file.stem
        queries = _load_queries(scenario_file)
        all_queries[scenario] = queries
        console.print(f"Loaded {len(queries)} queries for {scenario}")

    # Convert conversations to the format expected by existing task builders
    conv_dicts = _conversations_to_dicts(conversations)

    # Build query-based tasks (4 scenarios)
    corpus_docs = _build_corpus_docs(conversations)
    scenarios = [
        "topic_retrieval", "specific_detail",
        "cross_channel", "thread_discrimination",
    ]
    for scenario in scenarios:
        queries = all_queries.get(scenario, [])
        if not queries:
            console.print(f"[yellow]No queries for {scenario}, skipping.[/]")
            continue

        task = _build_query_task(scenario, queries, corpus_docs)
        task_path = output_path / f"{scenario}.json"
        task_path.write_text(task.model_dump_json(indent=2))
        _log_built_task(task, scenario)
        _log_bm25_baseline(task, scenario)

    # Build thread_retrieval task (programmatic)
    if conv_dicts:
        n_queries = min(500, len(conv_dicts))
        thread_task = build_thread_retrieval_task(
            conv_dicts, num_queries=n_queries, seed=seed,
        )
        _write_task(output_path, "thread_retrieval", thread_task)

    # Build response_retrieval task (programmatic)
    if conv_dicts:
        n_queries = min(500, len(conv_dicts))
        resp_task = build_response_retrieval_task(
            conv_dicts, num_queries=n_queries, seed=seed,
        )
        _write_task(output_path, "response_retrieval", resp_task)

    # Build summary_matching task (using transformed summaries as queries)
    summaries = _transform_summaries_for_retrieval(conversations)
    if summaries:
        summary_task = build_summary_matching_task(conv_dicts, summaries)
        _write_task(output_path, "summary_matching", summary_task)

    # Build cross_platform task (programmatic)
    if conv_dicts:
        # Hold out a minority platform for meaningful cross-platform evaluation
        from collections import Counter as _Counter
        source_counts = _Counter(c.get("source", "slack") for c in conv_dicts)
        held_out = source_counts.most_common()[-1][0] if source_counts else "slack"
        n_queries = min(500, len(conv_dicts))
        cp_task = build_cross_platform_task(
            conv_dicts, held_out_platform=held_out,
            num_queries=n_queries, seed=seed,
        )
        _write_task(output_path, "cross_platform_transfer", cp_task)

    # Generate dev/test splits
    _generate_splits(all_queries, splits_path, seed=seed)

    console.print(f"\n[bold green]All tasks built in {output_path}[/]")


def _load_conversations(path: Path) -> list[Conversation]:
    """Load conversations from JSONL file."""
    if not path.exists():
        return []
    conversations = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(Conversation.model_validate_json(line))
    return conversations


def _load_queries(path: Path) -> list[RetrievalQuery]:
    """Load queries from JSONL file."""
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(RetrievalQuery.model_validate_json(line))
    return queries


def _conversations_to_dicts(conversations: list[Conversation]) -> list[dict]:
    """Convert Conversation objects to the dict format expected by existing task builders."""
    result = []
    for conv in conversations:
        messages = [f"{msg.author}: {msg.content}" for msg in conv.messages]
        result.append({
            "id": conv.conversation_id,
            "source": conv.platform,
            "messages": messages,
        })
    return result


def _build_corpus_docs(conversations: list[Conversation]) -> list[BenchmarkCorpusDoc]:
    """Build corpus documents from conversations (messages joined with authorship)."""
    docs = []
    for conv in conversations:
        text = "\n".join(f"{msg.author}: {msg.content}" for msg in conv.messages)
        docs.append(BenchmarkCorpusDoc(
            doc_id=conv.conversation_id,
            text=text,
            source=conv.platform,
            metadata={
                "channel": conv.channel,
                "title": conv.title,
                "topic_tags": conv.topic_tags,
            },
        ))
    return docs


def _build_query_task(
    scenario: str,
    queries: list[RetrievalQuery],
    corpus_docs: list[BenchmarkCorpusDoc],
) -> BenchmarkTask:
    """Build a BenchmarkTask from queries and corpus for a given scenario."""
    descriptions = {
        "topic_retrieval": "Find conversations by general topic description",
        "specific_detail": "Find conversations containing a specific detail or decision",
        "cross_channel": "Find related conversations spanning multiple channels",
        "thread_discrimination": "Distinguish between topically similar conversations",
    }

    benchmark_queries = []
    for q in queries:
        benchmark_queries.append(BenchmarkQuery(
            query_id=q.query_id,
            query_text=q.query_text,
            relevant_doc_ids=q.relevant_conversation_ids,
            metadata={
                "scenario": q.scenario,
                "difficulty": q.difficulty,
                "hard_negative_ids": q.hard_negative_ids,
                "notes": q.notes,
            },
        ))

    return BenchmarkTask(
        task_id=scenario,
        task_name=scenario.replace("_", " ").title(),
        description=descriptions.get(scenario, scenario),
        queries=benchmark_queries,
        corpus=corpus_docs,
    )


def _transform_summaries_for_retrieval(
    conversations: list[Conversation],
) -> list[dict]:
    """Convert factual summaries to indirect information-need queries.

    Strips leading "Team discusses/debugs/designs..." patterns and converts
    to a shorter, less lexically overlapping form using channel name,
    participants, and a topic phrase.
    """
    # Patterns to strip from beginning of summaries
    strip_patterns = re.compile(
        r"^(The\s+)?team\s+(discusses?|debugs?|designs?|reviews?|brainstorms?|investigates?|works?\s+on|talks?\s+about|explores?)\s+",
        re.IGNORECASE,
    )

    summaries = []
    for conv in conversations:
        if not conv.summary or len(conv.messages) <= 5:
            continue

        # Start with the summary, strip leading team action phrases
        text = strip_patterns.sub("", conv.summary).strip()

        # Capitalize first letter after stripping
        if text:
            text = text[0].upper() + text[1:]

        # Build an information-need style query
        # Use channel + shortened topic rather than the full factual summary
        channel_label = conv.channel.replace("_", " ")
        participants_str = ""
        if conv.participants:
            participants_str = f" involving {', '.join(conv.participants[:2])}"

        # Truncate to first sentence or 80 chars, whichever is shorter
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77] + "..."

        query_text = f"{first_sentence} in {channel_label}{participants_str}"

        summaries.append({
            "conversation_id": conv.conversation_id,
            "summary_text": query_text,
        })

    return summaries


def _write_task(
    output_path: Path, name: str, task: BenchmarkTask,
) -> None:
    """Write a task to JSON and log its size."""
    path = output_path / f"{name}.json"
    path.write_text(task.model_dump_json(indent=2))
    _log_built_task(task, name)
    _log_bm25_baseline(task, name)


def _log_built_task(task: BenchmarkTask, name: str) -> None:
    """Log task build info."""
    n_q = len(task.queries)
    n_d = len(task.corpus)
    console.print(f"  Built {name}: {n_q} queries, {n_d} docs")


def _log_bm25_baseline(task: BenchmarkTask, scenario: str) -> None:
    """Run BM25 R@1 on a built task and log the result."""
    if not task.queries or not task.corpus:
        return

    corpus_texts = [d.text for d in task.corpus]
    doc_ids = [d.doc_id for d in task.corpus]
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    hits = 0
    for q in task.queries:
        tokenized_query = q.query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_idx = int(np.argmax(scores))
        if doc_ids[top_idx] in set(q.relevant_doc_ids):
            hits += 1

    total = len(task.queries)
    r1 = hits / total if total > 0 else 0.0
    msg = f"    BM25 R@1 for {scenario}: {r1:.0%} ({hits}/{total})"

    if r1 > 0.40:
        console.print(f"[yellow]{msg} — WARNING: > 40% threshold[/]")
        logger.warning(msg)
    else:
        console.print(f"[dim]{msg}[/]")


def _generate_splits(
    all_queries: dict[str, list[RetrievalQuery]],
    splits_path: Path,
    seed: int = 42,
) -> None:
    """Generate stratified dev/test splits (30/70)."""
    rng = random.Random(seed)
    dev_queries = []
    test_queries = []

    for scenario, queries in all_queries.items():
        shuffled = list(queries)
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * 0.3))
        dev_queries.extend(q.model_dump() for q in shuffled[:split_idx])
        test_queries.extend(q.model_dump() for q in shuffled[split_idx:])

    (splits_path / "dev.json").write_text(json.dumps(dev_queries, indent=2))
    (splits_path / "test.json").write_text(json.dumps(test_queries, indent=2))
    console.print(f"  Splits: {len(dev_queries)} dev, {len(test_queries)} test")
