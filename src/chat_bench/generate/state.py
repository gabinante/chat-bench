"""Generation state tracking for resumable pipeline execution."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import GenerationState

_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "metadata" / "generation_state.json"


def load_state(path: Path | None = None) -> GenerationState:
    """Load generation state from disk, or return fresh state."""
    path = path or _DEFAULT_STATE_PATH
    if path.exists():
        data = json.loads(path.read_text())
        return GenerationState.model_validate(data)
    return GenerationState()


def save_state(state: GenerationState, path: Path | None = None) -> None:
    """Persist generation state to disk."""
    path = path or _DEFAULT_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2))


def is_phase_complete(state: GenerationState, phase: str) -> bool:
    """Check if a phase has been completed."""
    return state.phases.get(phase, None) is not None and state.phases[phase].completed


def mark_phase_complete(state: GenerationState, phase: str) -> None:
    """Mark a phase as completed."""
    state.phases[phase].completed = True
