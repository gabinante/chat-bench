"""Main orchestrator — runs generation phases A-F."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console

from .client import GenerationClient
from .prompts import (
    phase_a_prompt,
    phase_b_prompt,
    phase_c_prompt,
    phase_d_prompt,
    phase_e_prompt,
    system_prompt,
)
from .reference_data import get_channel_config
from .schemas import Conversation, RetrievalQuery
from .state import is_phase_complete, load_state, mark_phase_complete, save_state
from .validate import validate_corpus

logger = logging.getLogger(__name__)
console = Console()

# -- Corpus scaling constants --
SEEDS_PER_CHANNEL = 15          # 5 batches × 3 per batch
SEED_BATCH_SIZE = 3             # conversations per seed batch
SEED_BATCHES_PER_CHANNEL = 5   # batches per channel in Phase A

CONFOUNDERS_PER_SEED = 3        # confounders generated per seed conversation
CONFOUNDER_BATCH_SIZE = 3       # seeds processed per Phase B call

NOISE_PER_CHANNEL = 30          # total noise conversations per channel
NOISE_BATCH_SIZE = 10           # conversations per Phase C call
NOISE_BATCHES_PER_CHANNEL = 3   # batches per channel in Phase C

QUERY_BATCHES_PER_SCENARIO = 16 # batches of queries per scenario in Phase E
QUERY_BATCH_SIZE = 18           # queries per batch

_CORPUS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "corpus" / "conversations.jsonl"
_QUERIES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "queries"
_STATS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "metadata" / "statistics.json"

QUERY_SCENARIOS = ["topic_retrieval", "specific_detail", "cross_channel", "thread_discrimination"]


def run_pipeline(
    phases: list[str] | None = None,
    resume: bool = True,
    model: str | None = None,
) -> None:
    """Run the generation pipeline.

    Args:
        phases: List of phase letters to run (e.g., ["A", "B"]). None = all.
        resume: If True, skip already-completed phases.
        model: Model ID override (default: claude-sonnet-4-20250514).
    """
    all_phases = ["A", "B", "C", "D", "E", "F"]
    phases = phases or all_phases

    state = load_state() if resume else load_state.__wrapped__() if hasattr(load_state, '__wrapped__') else load_state()
    if not resume:
        from .schemas import GenerationState
        state = GenerationState()
        _clear_generated_data()

    client_kwargs = {}
    if model:
        client_kwargs["model"] = model
    client = GenerationClient(**client_kwargs)

    _CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    for phase in phases:
        phase = phase.upper()
        if phase not in all_phases:
            console.print(f"[red]Unknown phase: {phase}[/]")
            continue

        if resume and is_phase_complete(state, phase):
            console.print(f"[dim]Phase {phase} already complete, skipping.[/]")
            continue

        console.print(f"\n[bold cyan]{'='*60}[/]")
        console.print(f"[bold cyan]Phase {phase}[/]")
        console.print(f"[bold cyan]{'='*60}[/]")

        if phase == "A":
            _run_phase_a(client, state)
        elif phase == "B":
            _run_phase_b(client, state)
        elif phase == "C":
            _run_phase_c(client, state)
        elif phase == "D":
            _run_phase_d(client, state)
        elif phase == "E":
            _run_phase_e(client, state)
        elif phase == "F":
            _run_phase_f(state)

        mark_phase_complete(state, phase)
        save_state(state)

    console.print(f"\n[bold green]Pipeline complete.[/]")
    console.print(f"API usage: {client.usage.api_calls} calls, "
                  f"{client.usage.input_tokens:,} input tokens, "
                  f"{client.usage.output_tokens:,} output tokens")
    console.print(f"Estimated cost: ${client.usage.estimated_cost_usd:.2f}")


def _run_phase_a(client: GenerationClient, state) -> None:
    """Phase A: Generate seed conversations per channel."""
    channels = get_channel_config()

    for ch in channels:
        platform = ch.get("platform", "slack")
        sys = system_prompt(platform=platform)
        for batch_num in range(SEED_BATCHES_PER_CHANNEL):
            console.print(
                f"  Generating seeds for [bold]{ch['name']}[/] "
                f"(batch {batch_num + 1}/{SEED_BATCHES_PER_CHANNEL})..."
            )
            prompt = phase_a_prompt(ch["id"], batch_size=SEED_BATCH_SIZE)

            conversations = client.generate_validated(
                sys, prompt, Conversation, wrap_key="conversations"
            )
            if not isinstance(conversations, list):
                conversations = [conversations]

            _append_conversations(conversations, platform=platform, phase="seed")
            count = len(conversations)
            state.phases["A"].conversations_generated += count
            state.phases["A"].batches_completed += 1
            state.total_conversations += count
            state.conversations_by_channel[ch["id"]] = (
                state.conversations_by_channel.get(ch["id"], 0) + count
            )
            save_state(state)
            console.print(f"    Generated {count} seed conversations")

    state.phases["A"].total_batches = len(channels) * SEED_BATCHES_PER_CHANNEL


def _run_phase_b(client: GenerationClient, state) -> None:
    """Phase B: Generate confounders for each seed.

    Processes seeds in small batches to keep output manageable.
    """
    conversations = _load_conversations()
    channels = get_channel_config()
    channel_map = {ch["id"]: ch for ch in channels}

    channel_ids = [ch["id"] for ch in channels]
    for channel_id in channel_ids:
        platform = channel_map[channel_id].get("platform", "slack")
        sys = system_prompt(platform=platform)
        seeds = [c for c in conversations if c.channel == channel_id]
        if not seeds:
            continue

        # Process seeds in batches
        for batch_start in range(0, len(seeds), CONFOUNDER_BATCH_SIZE):
            batch_seeds = seeds[batch_start:batch_start + CONFOUNDER_BATCH_SIZE]
            console.print(
                f"  Generating confounders for [bold]#{channel_id}[/] "
                f"(seeds {batch_start + 1}-{batch_start + len(batch_seeds)}/{len(seeds)})..."
            )
            seed_dicts = [s.model_dump() for s in batch_seeds]
            prompt = phase_b_prompt(
                channel_id, seed_dicts,
                confounders_per_seed=CONFOUNDERS_PER_SEED,
            )

            confounders = client.generate_validated(
                sys, prompt, Conversation, wrap_key="conversations"
            )
            if not isinstance(confounders, list):
                confounders = [confounders]

            _append_conversations(confounders, platform=platform, phase="confounder")
            count = len(confounders)
            state.phases["B"].conversations_generated += count
            state.phases["B"].batches_completed += 1
            state.total_conversations += count
            state.conversations_by_channel[channel_id] = (
                state.conversations_by_channel.get(channel_id, 0) + count
            )
            save_state(state)
            console.print(f"    Generated {count} confounders")


def _run_phase_c(client: GenerationClient, state) -> None:
    """Phase C: Generate noise conversations (short, 3-8 messages each)."""
    channels = get_channel_config()

    for ch in channels:
        platform = ch.get("platform", "slack")
        sys = system_prompt(platform=platform)
        for batch_num in range(NOISE_BATCHES_PER_CHANNEL):
            console.print(
                f"  Generating noise for [bold]{ch['name']}[/] "
                f"(batch {batch_num + 1}/{NOISE_BATCHES_PER_CHANNEL})..."
            )
            prompt = phase_c_prompt(ch["id"], batch_size=NOISE_BATCH_SIZE)

            noise = client.generate_validated(
                sys, prompt, Conversation, wrap_key="conversations"
            )
            if not isinstance(noise, list):
                noise = [noise]

            _append_conversations(noise, platform=platform, phase="noise")
            count = len(noise)
            state.phases["C"].conversations_generated += count
            state.phases["C"].batches_completed += 1
            state.total_conversations += count
            state.conversations_by_channel[ch["id"]] = (
                state.conversations_by_channel.get(ch["id"], 0) + count
            )
            save_state(state)
            console.print(f"    Generated {count} noise conversations")

    state.phases["C"].total_batches = len(channels) * NOISE_BATCHES_PER_CHANNEL


def _run_phase_d(client: GenerationClient, state) -> None:
    """Phase D: Add cross-references between conversations."""
    conversations = _load_conversations()
    sys = system_prompt()

    # Process in batches to fit context window
    batch_size = 50
    conv_dicts = [c.model_dump() for c in conversations]
    total_refs_added = 0

    for i in range(0, len(conv_dicts), batch_size):
        batch = conv_dicts[i:i + batch_size]
        console.print(f"  Processing cross-references for batch {i // batch_size + 1}...")

        prompt = phase_d_prompt(batch)
        result = client.generate_json(sys, prompt, max_tokens=4096)

        xrefs = result.get("cross_references", [])
        # Build lookup from conversation_id → Conversation
        conv_map = {c.conversation_id: c for c in conversations}
        for xref in xrefs:
            cid = xref.get("conversation_id", "")
            refs = xref.get("add_references", [])
            if cid in conv_map:
                existing = set(conv_map[cid].cross_references)
                for ref in refs:
                    if ref in conv_map and ref not in existing:
                        conv_map[cid].cross_references.append(ref)
                        total_refs_added += 1

        state.phases["D"].batches_completed += 1
        save_state(state)

    # Rewrite corpus with updated cross-references
    _rewrite_corpus(conversations)
    console.print(f"    Added {total_refs_added} cross-references")


def _run_phase_e(client: GenerationClient, state) -> None:
    """Phase E: Generate retrieval queries (8 batches per scenario)."""
    conversations = _load_conversations()

    # Backfill phase metadata so prompt can show confounder relationships
    _backfill_phase_metadata(conversations)

    # Build confounder map for prompt context
    confounder_map = _build_confounder_map(conversations)

    # Only pass summaries to prompt (not full messages) to fit context
    conv_summaries = [
        {
            "conversation_id": c.conversation_id,
            "channel": c.channel,
            "title": c.title,
            "topic_tags": c.topic_tags,
            "phase": c.phase,
            "confounder_for": c.confounder_for,
        }
        for c in conversations
    ]

    # Build corpus texts for BM25 filtering
    corpus_texts = {
        c.conversation_id: "\n".join(f"{msg.author}: {msg.content}" for msg in c.messages)
        for c in conversations
    }

    sys = system_prompt()

    for scenario in QUERY_SCENARIOS:
        console.print(f"  Generating queries for [bold]{scenario}[/]...")

        all_queries: list[RetrievalQuery] = []
        for batch_num in range(QUERY_BATCHES_PER_SCENARIO):
            console.print(f"    Batch {batch_num + 1}/{QUERY_BATCHES_PER_SCENARIO}...")
            prompt = phase_e_prompt(
                conv_summaries, scenario, batch_size=QUERY_BATCH_SIZE,
                confounder_map=confounder_map,
            )
            queries = client.generate_validated(
                sys, prompt, RetrievalQuery, max_tokens=8192, wrap_key="queries"
            )
            if not isinstance(queries, list):
                queries = [queries]
            all_queries.extend(queries)
            state.phases["E"].batches_completed += 1
            save_state(state)

        console.print(f"    Raw: {len(all_queries)} queries")

        # Assign unique IDs before dedup (LLM reuses IDs across batches)
        _assign_query_ids(all_queries, scenario)

        # Deduplicate by text similarity (IDs are now unique)
        unique = _deduplicate_queries(all_queries)
        console.print(f"    After dedup: {len(unique)} queries")

        # BM25 filter: prefer queries that BM25 can't solve
        filtered = _bm25_filter_queries(unique, corpus_texts)
        console.print(f"    After BM25 filter: {len(filtered)} queries")

        # Write queries to file
        query_path = _QUERIES_DIR / f"{scenario}.jsonl"
        with open(query_path, "w") as f:
            for q in filtered:
                f.write(q.model_dump_json() + "\n")

        state.phases["E"].queries_generated += len(filtered)
        state.total_queries += len(filtered)
        console.print(f"    [green]Final: {len(filtered)} queries for {scenario}[/]")

    state.phases["E"].total_batches = len(QUERY_SCENARIOS) * QUERY_BATCHES_PER_SCENARIO


def _assign_query_ids(queries: list[RetrievalQuery], scenario: str) -> None:
    """Assign unique sequential query IDs within a scenario.

    Replaces whatever IDs the LLM generated with deterministic sequential IDs.
    """
    for i, q in enumerate(queries, 1):
        q.query_id = f"{scenario}_{i:03d}"


def _deduplicate_queries(queries: list[RetrievalQuery]) -> list[RetrievalQuery]:
    """Deduplicate queries by query_id and by text similarity (Jaccard > 0.7)."""
    # First pass: deduplicate by query_id
    seen_ids: set[str] = set()
    id_unique: list[RetrievalQuery] = []
    for q in queries:
        if q.query_id not in seen_ids:
            seen_ids.add(q.query_id)
            id_unique.append(q)

    # Second pass: deduplicate by text similarity
    result: list[RetrievalQuery] = []
    seen_word_sets: list[set[str]] = []

    for q in id_unique:
        words = set(q.query_text.lower().split())
        is_dup = False
        for existing_words in seen_word_sets:
            if not words or not existing_words:
                continue
            intersection = len(words & existing_words)
            union = len(words | existing_words)
            if union > 0 and intersection / union > 0.7:
                is_dup = True
                break
        if not is_dup:
            result.append(q)
            seen_word_sets.append(words)

    return result


def _bm25_filter_queries(
    queries: list[RetrievalQuery],
    corpus_texts: dict[str, str],
) -> list[RetrievalQuery]:
    """Filter queries based on BM25 solvability.

    Keep all BM25-resistant queries. Allow up to 50% of remaining budget
    as BM25-solvable (preferring 'hard' difficulty).
    """
    from rank_bm25 import BM25Okapi

    if not queries or not corpus_texts:
        return queries

    doc_ids = list(corpus_texts.keys())
    tokenized_corpus = [corpus_texts[did].lower().split() for did in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_resistant: list[RetrievalQuery] = []
    bm25_solvable: list[RetrievalQuery] = []

    for q in queries:
        tokenized_query = q.query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_idx = scores.argmax()
        top_doc_id = doc_ids[top_idx]

        relevant_set = set(q.relevant_conversation_ids)
        if top_doc_id in relevant_set:
            q.bm25_rank_1 = True
            bm25_solvable.append(q)
        else:
            q.bm25_rank_1 = False
            bm25_resistant.append(q)

    # Budget: allow up to 50% of remaining slots as BM25-solvable
    target_total = max(len(queries), 100)
    remaining_budget = max(0, target_total - len(bm25_resistant))
    allowed_solvable = remaining_budget // 2

    # Prefer hard difficulty among BM25-solvable
    difficulty_order = {"hard": 0, "medium": 1, "easy": 2}
    bm25_solvable.sort(key=lambda q: difficulty_order.get(q.difficulty, 1))
    kept_solvable = bm25_solvable[:allowed_solvable]

    return bm25_resistant + kept_solvable


def _run_phase_f(state) -> None:
    """Phase F: Run validation and report."""
    conversations = _load_conversations()
    queries = _load_all_queries()

    # Build channel participants map
    from .reference_data import get_channel_config
    channel_participants = {}
    for ch in get_channel_config():
        channel_participants[ch["id"]] = ch["participants"]

    report = validate_corpus(conversations, queries, channel_participants)
    console.print(f"\n{report.summary()}")

    state.validation_passed = report.passed

    # Write statistics
    stats = {
        "total_conversations": len(conversations),
        "total_queries": len(queries),
        "conversations_by_channel": {},
        "queries_by_scenario": {},
        "avg_messages_per_conversation": 0,
        "validation_passed": report.passed,
        "validation_errors": len(report.errors),
        "validation_warnings": len(report.warnings),
    }

    from collections import Counter
    ch_counts = Counter(c.channel for c in conversations)
    stats["conversations_by_channel"] = dict(ch_counts)

    sc_counts = Counter(q.scenario for q in queries)
    stats["queries_by_scenario"] = dict(sc_counts)

    if conversations:
        total_msgs = sum(len(c.messages) for c in conversations)
        stats["avg_messages_per_conversation"] = round(total_msgs / len(conversations), 1)

    _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATS_PATH.write_text(json.dumps(stats, indent=2))
    console.print(f"\n[green]Statistics written to {_STATS_PATH}[/]")


# -- Confounder map and phase metadata --

def _build_confounder_map(conversations: list[Conversation]) -> dict[str, list[str]]:
    """Build seed→confounder mapping using the phase field and ID ordering.

    Seeds and confounders are tagged at write time (phase="seed" / "confounder").
    Within each channel, confounders are assigned round-robin to seeds by ID order:
    the first CONFOUNDERS_PER_SEED confounders map to the first seed, etc.

    Returns:
        dict mapping seed conversation_id → list of confounder conversation_ids
    """
    n_conf = CONFOUNDERS_PER_SEED

    def _sort_key(c: Conversation) -> int:
        parts = c.conversation_id.rsplit("_", 1)
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return 999

    # Group by channel
    by_channel: dict[str, list[Conversation]] = {}
    for c in conversations:
        by_channel.setdefault(c.channel, []).append(c)

    confounder_map: dict[str, list[str]] = {}

    for channel_id, convs in by_channel.items():
        seeds = sorted(
            [c for c in convs if c.phase == "seed"], key=_sort_key
        )
        confounders = sorted(
            [c for c in convs if c.phase == "confounder"], key=_sort_key
        )

        for i, seed in enumerate(seeds):
            conf_start = i * n_conf
            conf_end = conf_start + n_conf
            seed_confounders = confounders[conf_start:conf_end]
            confounder_map[seed.conversation_id] = [
                c.conversation_id for c in seed_confounders
            ]

    return confounder_map


def _backfill_phase_metadata(conversations: list[Conversation]) -> None:
    """Set confounder_for on confounder conversations using the confounder map.

    Phase tags (seed/confounder/noise) are set at write time by _append_conversations.
    This only fills in the confounder_for field which requires the full map.
    """
    confounder_map = _build_confounder_map(conversations)

    # Build reverse map: confounder_id → seed_id
    reverse_map: dict[str, str] = {}
    for seed_id, conf_ids in confounder_map.items():
        for cid in conf_ids:
            reverse_map[cid] = seed_id

    for conv in conversations:
        if conv.phase == "confounder" and conv.conversation_id in reverse_map:
            conv.confounder_for = reverse_map[conv.conversation_id]


# -- Channel ID normalization --

# Map common variants to canonical channel IDs
_CHANNEL_ALIASES = {
    "game-design": "game_design",
    "#game-design": "game_design",
    "game design": "game_design",
    "art-direction": "art_direction",
    "#art-direction": "art_direction",
    "art direction": "art_direction",
    "lore-narrative": "lore_narrative",
    "#lore-narrative": "lore_narrative",
    "lore narrative": "lore_narrative",
    "devops-infra": "devops_infra",
    "#devops-infra": "devops_infra",
    "devops infra": "devops_infra",
    "#engineering": "engineering",
    "#general": "general",
}


def _normalize_channel(channel: str) -> str:
    """Normalize a channel string to its canonical ID."""
    return _CHANNEL_ALIASES.get(channel, channel)


def _normalize_conversations(conversations: list[Conversation]) -> list[Conversation]:
    """Normalize channel IDs and conversation_ids in a list of conversations."""
    for conv in conversations:
        canonical = _normalize_channel(conv.channel)
        if canonical != conv.channel:
            # Also fix conversation_id prefix if it used the wrong channel form
            old_prefix = conv.channel.lstrip("#").replace("-", "_").replace(" ", "_")
            conv.conversation_id = conv.conversation_id.replace(
                conv.channel.lstrip("#").replace("_", "-"),
                canonical,
            ).replace(
                conv.channel.lstrip("#"),
                canonical,
            )
            conv.channel = canonical
    return conversations


def _clear_generated_data() -> None:
    """Remove generated corpus, query, and statistics files for a fresh run."""
    if _CORPUS_PATH.exists():
        _CORPUS_PATH.unlink()
    for qf in _QUERIES_DIR.glob("*.jsonl"):
        qf.unlink()
    if _STATS_PATH.exists():
        _STATS_PATH.unlink()


# -- Corpus I/O helpers --


def _get_channel_counters() -> dict[str, int]:
    """Scan existing corpus and return the next available number per channel.

    Returns a dict mapping channel_id to the next unused integer suffix.
    For example, if engineering_001 through engineering_015 exist, returns
    {"engineering": 16}.
    """
    import re

    counters: dict[str, int] = {}
    if not _CORPUS_PATH.exists():
        return counters

    with open(_CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cid = data.get("conversation_id", "")
            m = re.match(r"^(.+)_(\d+)$", cid)
            if m:
                channel, num = m.group(1), int(m.group(2))
                counters[channel] = max(counters.get(channel, 0), num + 1)
    return counters


def _assign_ids(conversations: list[Conversation]) -> None:
    """Assign unique, deterministic conversation_id and message_id values.

    Reads the existing corpus to find the next available number per channel,
    then assigns sequential IDs regardless of what the LLM generated.
    """
    counters = _get_channel_counters()

    for conv in conversations:
        channel = conv.channel
        num = counters.get(channel, 1)
        counters[channel] = num + 1

        new_conv_id = f"{channel}_{num:03d}"
        conv.conversation_id = new_conv_id

        # Reassign message IDs
        for i, msg in enumerate(conv.messages, 1):
            msg.message_id = f"{new_conv_id}_msg_{i:03d}"

        # Fix reply_to references within the same conversation
        # (these reference old message IDs the LLM generated — remap them)
        old_msg_ids = {}
        for i, msg in enumerate(conv.messages, 1):
            # Build mapping from position to new ID (already assigned above)
            old_msg_ids[i] = msg.message_id

        # reply_to can only reference messages within the same conversation,
        # so remap by finding the referenced position
        # Since we don't have a reliable old->new mapping, clear invalid reply_to
        valid_ids = {msg.message_id for msg in conv.messages}
        for msg in conv.messages:
            if msg.reply_to and msg.reply_to not in valid_ids:
                msg.reply_to = None


def _append_conversations(
    conversations: list[Conversation],
    platform: str = "slack",
    phase: str = "",
) -> None:
    """Append conversations to the JSONL corpus file.

    Normalizes channel IDs, assigns unique conversation/message IDs, tags
    with generation phase, and sets platform as a fallback.
    """
    _normalize_conversations(conversations)
    _assign_ids(conversations)
    for conv in conversations:
        if phase:
            conv.phase = phase
        if not conv.platform or conv.platform == "slack":
            conv.platform = platform
    with open(_CORPUS_PATH, "a") as f:
        for conv in conversations:
            f.write(conv.model_dump_json() + "\n")


def _rewrite_corpus(conversations: list[Conversation]) -> None:
    """Rewrite the entire corpus file (used after cross-reference updates)."""
    with open(_CORPUS_PATH, "w") as f:
        for conv in conversations:
            f.write(conv.model_dump_json() + "\n")


def _load_conversations() -> list[Conversation]:
    """Load all conversations from the JSONL corpus file."""
    if not _CORPUS_PATH.exists():
        return []
    conversations = []
    with open(_CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(Conversation.model_validate_json(line))
    return conversations


def _load_all_queries() -> list[RetrievalQuery]:
    """Load all queries from all scenario files."""
    queries = []
    if not _QUERIES_DIR.exists():
        return queries
    for path in _QUERIES_DIR.glob("*.jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    queries.append(RetrievalQuery.model_validate_json(line))
    return queries
