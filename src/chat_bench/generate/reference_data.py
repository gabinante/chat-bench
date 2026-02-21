"""Load and format Mesh reference data for prompt context."""

from __future__ import annotations

import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reference"


def get_mesh_context() -> str:
    """Return the curated Mesh world context markdown (~3000 tokens)."""
    path = _DATA_DIR / "mesh_context.md"
    return path.read_text()


def get_channel_config() -> list[dict]:
    """Return channel definitions with participant lists and style guides."""
    path = _DATA_DIR / "channels.json"
    data = json.loads(path.read_text())
    return data["channels"]


def get_participants() -> list[dict]:
    """Return full participant profiles."""
    path = _DATA_DIR / "participants.json"
    data = json.loads(path.read_text())
    return data["participants"]


def get_participant_map() -> dict[str, dict]:
    """Return participants keyed by ID for quick lookup."""
    return {p["id"]: p for p in get_participants()}


def get_channel_map() -> dict[str, dict]:
    """Return channels keyed by ID for quick lookup."""
    return {c["id"]: c for c in get_channel_config()}


def format_channel_context(channel_id: str) -> str:
    """Format a single channel's context for inclusion in a prompt."""
    channels = get_channel_map()
    participants = get_participant_map()

    ch = channels[channel_id]
    platform = ch.get("platform", "slack")
    lines = [
        f"Channel: {ch['name']}",
        f"Platform: {platform}",
        f"Description: {ch['description']}",
        f"Style: {ch['style']}",
        "",
        "Participants:",
    ]
    for pid in ch["participants"]:
        p = participants[pid]
        lines.append(f"  - {p['name']} (@{pid}) — {p['role']}")
        lines.append(f"    Personality: {p['personality']}")

    return "\n".join(lines)


def format_all_channels_summary() -> str:
    """Format a brief summary of all channels for prompt context."""
    channels = get_channel_config()
    lines = ["Slack Channels:"]
    for ch in channels:
        lines.append(f"  - {ch['name']}: {ch['description']} (participants: {', '.join(ch['participants'])})")
    return "\n".join(lines)
