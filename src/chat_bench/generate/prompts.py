"""Prompt templates for each generation phase."""

from __future__ import annotations

from .reference_data import format_channel_context, get_mesh_context

# -- Platform style guides --

PLATFORM_STYLES = {
    "slack": {
        "label": "Slack",
        "conventions": (
            "Threaded replies (reply_to field), emoji reactions, @mentions, "
            "code blocks with triple backticks, channel cross-links (#channel), "
            "integration bot messages (e.g., GitHub, Jira), formatted text "
            "(bold, italic, lists). Professional but friendly tone."
        ),
    },
    "discord": {
        "label": "Discord",
        "conventions": (
            "Casual tone with internet slang, heavy emoji/reaction use, "
            "embedded images and links, role @mentions, reply chains, "
            "occasional ALL CAPS for emphasis, GIF reactions, "
            "channel categories. More playful and informal than Slack."
        ),
    },
    "irc": {
        "label": "IRC",
        "conventions": (
            "Terse, no rich formatting (no bold/italic/embeds). "
            "Nick-based addressing (nick: message), action commands (/me does something), "
            "no reactions or threading — flat linear chat. "
            "Short messages, abbreviations, minimal punctuation. Old-school internet tone."
        ),
    },
}


# -- Shared JSON schema descriptions embedded in prompts --

CONVERSATION_SCHEMA = """\
Each conversation must be a JSON object with these fields:
{
  "conversation_id": "string (unique, format: {channel_id}_{number}, e.g. engineering_001)",
  "channel": "string (channel id using underscores, one of: engineering, game_design, art_direction, lore_narrative, devops_infra, general)",
  "platform": "string (one of: slack, discord, irc)",
  "title": "string (short descriptive title)",
  "topic_tags": ["string", ...],
  "participants": ["author_id", ...],
  "messages": [
    {
      "message_id": "string (unique, format: {conversation_id}_msg_{number})",
      "author": "string (participant id, e.g. alex_chen)",
      "timestamp": "string (ISO 8601, within a realistic work day)",
      "content": "string (the message text)",
      "reply_to": "string or null (message_id being replied to)",
      "reactions": [{"emoji": "string", "users": ["author_id"]}]
    }
  ],
  "cross_references": [],
  "summary": "string (1-2 sentence summary of the conversation)"
}"""

QUERY_SCHEMA = """\
Each query must be a JSON object with these fields:
{
  "query_id": "string (unique, format: {scenario}_{number})",
  "query_text": "string (natural language search query)",
  "scenario": "string (one of: topic_retrieval, specific_detail, cross_channel, thread_discrimination)",
  "relevant_conversation_ids": ["string", ...],
  "hard_negative_ids": ["string", ...],
  "difficulty": "string (easy, medium, or hard)",
  "notes": "string (why this query is interesting/challenging)"
}"""


def system_prompt(platform: str = "slack") -> str:
    """Shared system prompt with world context and output format."""
    mesh_context = get_mesh_context()
    style = PLATFORM_STYLES.get(platform, PLATFORM_STYLES["slack"])
    platform_label = style["label"]
    platform_conventions = style["conventions"]

    return f"""\
You are a data generation assistant for a benchmark dataset. You generate realistic {platform_label} \
conversations from "Prismatic Studios," a fictional game development studio building \
"The Mesh" — a cyberpunk MMORPG set in a post-apocalyptic Bay Area.

Your output must be valid JSON. Do not include any text outside the JSON.

## World Reference
{mesh_context}

## Output Format
{CONVERSATION_SCHEMA}

## Platform: {platform_label}
Conventions: {platform_conventions}
All conversations in this batch MUST set "platform": "{platform}".

## Quality Requirements
- Messages should feel natural — varied lengths, tone appropriate to {platform_label} culture
- Include realistic {platform_label} behaviors: {platform_conventions}
- Reference specific game content by name (abilities, zones, NPCs, items) — not generic placeholders
- Each conversation should have 8-25 messages with a clear topic arc
- Timestamps should span realistic durations (5 minutes to 2 hours)
- Participants should stay in character per their personality profiles"""


def phase_a_prompt(channel_id: str, batch_size: int = 5) -> str:
    """Phase A: Generate seed conversations with backbone storylines."""
    channel_context = format_channel_context(channel_id)
    return f"""\
Generate {batch_size} seed conversations for the following Slack channel. These are the \
"backbone" conversations that establish key topics and ongoing storylines.

{channel_context}

## Guidelines
- Each conversation should cover a distinct, substantial topic relevant to this channel
- Include specific Mesh game details (named abilities, zones, NPCs, tech components)
- Vary conversation length (8-25 messages) and intensity
- Some conversations should have clear resolutions, others should end with open questions
- Include at least one conversation with a problem/debugging thread
- Include at least one with a creative brainstorming discussion
- Make conversations reference specific ongoing work (e.g., "the Channeler rework", "the rift scaling PR")

Return a JSON object: {{"conversations": [...]}}"""


def phase_b_prompt(
    channel_id: str,
    seed_conversations: list[dict],
    confounders_per_seed: int = 2,
) -> str:
    """Phase B: Generate topically similar confounders for each seed."""
    channel_context = format_channel_context(channel_id)
    seed_summaries = []
    for conv in seed_conversations:
        seed_summaries.append(
            f"- [{conv['conversation_id']}] \"{conv['title']}\": {conv['summary']}"
        )
    seeds_text = "\n".join(seed_summaries)

    return f"""\
Generate confounding conversations — topically similar but semantically distinct from the \
seed conversations below. These serve as hard negatives for retrieval tasks.

{channel_context}

## Seed Conversations (generate confounders for each)
{seeds_text}

## Guidelines
- For each seed, generate {confounders_per_seed} conversations that discuss related but \
different aspects of the same topic area
- A confounder for a "CockroachDB query optimization" seed might discuss "CockroachDB schema migration"
- A confounder for a "Channeler ability balance" seed might discuss "Shaman ability balance"
- Confounders should share vocabulary and themes but have clearly different specific content
- Use overlapping participants where natural
- Maintain the same channel style and tone
- Give confounders IDs that continue the channel's numbering sequence

Return a JSON object: {{"conversations": [...]}}"""


def phase_c_prompt(channel_id: str, batch_size: int = 12) -> str:
    """Phase C: Generate noise conversations (short, low-signal)."""
    channel_context = format_channel_context(channel_id)
    return f"""\
Generate {batch_size} short, low-signal noise conversations for this channel. These represent \
the everyday chatter that pads a real Slack workspace.

{channel_context}

## Guidelines
- Keep conversations SHORT: 3-8 messages each
- Topics should be mundane/routine: quick questions, status updates, scheduling, social chat
- Examples: "anyone know the wifi password?", "standup in 5", "nice PR!", "lunch?", \
"build is broken again", "who's reviewing the rift PR?", "happy friday"
- Some can be just 2-3 messages with an answer
- These should be clearly less substantive than seed conversations
- Mix of channel-relevant quick questions and general chatter

Return a JSON object: {{"conversations": [...]}}"""


def phase_d_prompt(conversations: list[dict]) -> str:
    """Phase D: Add cross-references between existing conversations."""
    conv_summaries = []
    for conv in conversations:
        conv_summaries.append(
            f"- [{conv['conversation_id']}] #{conv['channel']} \"{conv['title']}\": {conv['summary']}"
        )
    summaries_text = "\n".join(conv_summaries)

    return f"""\
Review these conversations and identify natural cross-references between them. A cross-reference \
means one conversation's topic is directly related to or builds on another's.

## Conversations
{summaries_text}

## Guidelines
- Identify 20-40 cross-reference pairs where conversations naturally relate
- Cross-channel references are especially valuable (e.g., engineering discussing an art pipeline issue)
- Only mark genuine topical connections, not superficial keyword overlap
- Each reference should be bidirectional (if A references B, B references A)

Return a JSON object:
{{
  "cross_references": [
    {{
      "conversation_id": "string (the conversation to update)",
      "add_references": ["string (conversation_ids to add as cross-references)"]
    }}
  ]
}}"""


def phase_e_prompt(
    conversations: list[dict],
    scenario: str,
    batch_size: int = 18,
    confounder_map: dict[str, list[str]] | None = None,
) -> str:
    """Phase E: Generate retrieval queries with hard negatives.

    Args:
        conversations: List of conversation summary dicts.
        scenario: Query scenario type.
        batch_size: Number of queries to generate per call.
        confounder_map: Optional mapping of seed_id → list of confounder_ids.
    """
    conv_summaries = []
    for conv in conversations:
        label = ""
        if conv.get("phase") == "seed":
            label = " [SEED]"
        elif conv.get("phase") == "confounder":
            label = f" [CONFOUNDER for {conv.get('confounder_for', '?')}]"
        conv_summaries.append(
            f"- [{conv['conversation_id']}] #{conv['channel']} \"{conv['title']}\": "
            f"{conv['summary']}{label}"
        )
    summaries_text = "\n".join(conv_summaries)

    # Format confounder pairs for the prompt
    confounder_context = ""
    if confounder_map:
        pairs = []
        for seed_id, conf_ids in confounder_map.items():
            if conf_ids:
                pairs.append(f"  - {seed_id} → confounders: {', '.join(conf_ids)}")
        if pairs:
            confounder_context = (
                "\n\n## Confounder Pairs (use these as hard negatives)\n"
                "These confounders are topically similar to their seed but discuss different specifics. "
                "Use them as hard_negative_ids — especially for thread_discrimination queries.\n"
                + "\n".join(pairs)
            )

    scenario_guidance = {
        "topic_retrieval": """\
Generate queries that search for a conversation by its general topic.
- Easy: a broad topic area with few overlapping conversations
- Medium: a topic with confounders in the same domain
- Hard: requires understanding intersection of multiple domains (e.g., infra AND game design)
Hard negatives: use the confounder conversations that share vocabulary with the target.""",

        "specific_detail": """\
Generate queries that search for a specific detail, fact, or decision.
- Easy: a unique detail only found in one conversation
- Medium: a detail where confounders discuss similar but different specifics
- Hard: requires finding a very specific data point (a number, a name, a config value)
Hard negatives: conversations mentioning similar but different specific details.""",

        "cross_channel": """\
Generate queries that require finding information spanning multiple channels.
- The query references a topic discussed across engineering + game_design, or art + lore, etc.
- Relevant conversations should come from 2+ different channels
- Hard negatives are single-channel conversations on related topics
Example: a query about how a game system affects infrastructure.""",

        "thread_discrimination": """\
Generate queries designed to test fine-grained discrimination between similar conversations.
- Each query should have exactly ONE correct conversation and multiple plausible-but-wrong ones
- MUST use confounders as hard negatives (see Confounder Pairs below)
- Easy: different topic areas being discussed
- Medium: same topic area, different aspects
- Hard: same aspect, different specific details or approaches
Hard negatives MUST include the confounders for the target seed conversation.""",
    }

    return f"""\
Generate {batch_size} retrieval queries for the "{scenario}" scenario.

## CRITICAL: Paraphrasing Rules
You MUST follow these rules for every query_text you generate:
1. NEVER copy phrases directly from conversation titles or summaries
2. Use synonyms and circumlocution — describe the NEED, not the content
3. Rephrase technical terms: "CockroachDB performance issues" → "database slowdowns in the \
distributed SQL layer"
4. Rephrase game terms: "Channeler ability rework" → "rebalancing the support caster's kit"
5. Use indirect references: "the PR that fixed the scaling bug" → "recent work on handling more \
concurrent players in instanced zones"
6. Self-test: If you removed the conversation summaries above, could someone still understand \
what information the query is looking for? If yes, the query is good.

## Scenario Guidelines
{scenario_guidance[scenario]}

## Query Format
{QUERY_SCHEMA}

## Available Conversations
{summaries_text}
{confounder_context}

## Requirements
- Mix of easy (30%), medium (40%), hard (30%) difficulty
- Each query must have at least 1 relevant conversation and 2-3 hard negatives
- Hard negatives must be real conversation IDs from the list above
- For thread_discrimination: hard negatives MUST include confounders for the target conversation
- Query text should be natural language (how a user would actually search)
- DO NOT quote, paraphrase closely, or echo exact phrases from titles/summaries
- Queries should describe an information need, not summarize a conversation

Return a JSON object: {{"queries": [...]}}"""
