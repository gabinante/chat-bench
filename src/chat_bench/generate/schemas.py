"""Pydantic models for the raw generated corpus."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""

    message_id: str
    author: str
    timestamp: str  # ISO 8601 format
    content: str
    reply_to: str | None = None
    reactions: list[dict] = Field(default_factory=list)


class Conversation(BaseModel):
    """A generated conversation with metadata."""

    conversation_id: str
    channel: str
    title: str
    topic_tags: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    cross_references: list[str] = Field(default_factory=list)
    summary: str = ""
    platform: str = "slack"  # "slack", "discord", "irc"
    phase: str = ""  # "seed", "confounder", "noise"
    confounder_for: str = ""  # seed conversation_id this is a confounder for


class RetrievalQuery(BaseModel):
    """A retrieval query with ground truth."""

    query_id: str
    query_text: str
    scenario: str  # "topic_retrieval", "specific_detail", "cross_channel", "thread_discrimination"
    relevant_conversation_ids: list[str]
    hard_negative_ids: list[str] = Field(default_factory=list)
    difficulty: str = "medium"  # "easy", "medium", "hard"
    bm25_rank_1: bool = False  # whether BM25 places a relevant doc at rank 1
    notes: str = ""


class PhaseState(BaseModel):
    """State for a single generation phase."""

    completed: bool = False
    conversations_generated: int = 0
    queries_generated: int = 0
    batches_completed: int = 0
    total_batches: int = 0


class GenerationState(BaseModel):
    """Tracks generation pipeline progress for resumability."""

    phases: dict[str, PhaseState] = Field(default_factory=lambda: {
        "A": PhaseState(),
        "B": PhaseState(),
        "C": PhaseState(),
        "D": PhaseState(),
        "E": PhaseState(),
        "F": PhaseState(),
    })
    total_conversations: int = 0
    conversations_by_channel: dict[str, int] = Field(default_factory=dict)
    total_queries: int = 0
    validation_passed: bool = False
