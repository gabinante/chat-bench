"""Download and convert the DISCO dataset (real Discord conversations).

DISCO v2 (MSR 2022): 28,712 disentangled conversations from Python/Go/Clojure/Racket
Discord servers. Each XML file contains messages tagged with conversation_id.

Reference: Subash et al., "DISCO: A Dataset of Discord Chat Conversations for
Software Engineering Research", MSR 2022.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

from .generate.schemas import Conversation, Message

logger = logging.getLogger(__name__)

DISCO_URL = (
    "https://zenodo.org/records/5909202/files/"
    "DISCO-A%20Dataset%20of%20Discord%20Chat%20Conversations%20for%20Software%20Engineering%20Research.zip"
    "?download=1"
)
DISCO_CACHE = Path("~/.cache/chatbench/disco").expanduser()

MIN_MESSAGES = 3


def get_disco_conversations(
    max_per_channel: int | None = None,
) -> list[Conversation]:
    """Download DISCO dataset and convert to Conversation objects.

    Args:
        max_per_channel: Limit conversations per channel (None = no limit).

    Returns:
        List of Conversation objects with disco_ prefixed IDs.
    """
    zip_path = _download_disco()
    xml_files = _extract_xml(zip_path)
    conversations: list[Conversation] = []
    for xml_file in xml_files:
        conversations.extend(_parse_xml(xml_file))
    if max_per_channel:
        conversations = _limit_per_channel(conversations, max_per_channel)
    logger.info("Loaded %d DISCO conversations", len(conversations))
    return conversations


def _download_disco() -> Path:
    """Download the DISCO ZIP from Zenodo (cached)."""
    import requests

    DISCO_CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = DISCO_CACHE / "disco.zip"

    if zip_path.exists():
        logger.info("Using cached DISCO ZIP: %s", zip_path)
        return zip_path

    logger.info("Downloading DISCO dataset from Zenodo...")
    resp = requests.get(DISCO_URL, stream=True, timeout=120)
    resp.raise_for_status()

    tmp_path = zip_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    tmp_path.rename(zip_path)
    logger.info("Downloaded DISCO dataset to %s", zip_path)
    return zip_path


def _extract_xml(zip_path: Path) -> list[Path]:
    """Extract XML files from the DISCO ZIP archive."""
    extract_dir = DISCO_CACHE / "extracted"
    if extract_dir.exists() and list(extract_dir.rglob("*.xml")):
        return sorted(extract_dir.rglob("*.xml"))

    logger.info("Extracting DISCO ZIP...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    xml_files = sorted(extract_dir.rglob("*.xml"))
    logger.info("Extracted %d XML files", len(xml_files))
    return xml_files


_KNOWN_DISCO_PARENTS = frozenset({"python", "go", "golang", "clojure", "racket"})


def _channel_from_path(xml_path: Path) -> str:
    """Derive a channel name from the XML file path.

    Uses the parent directory and filename stem to create a channel identifier
    like 'python_general', 'golang', 'clojure', 'racket'.

    Only includes the parent directory name when it's a known DISCO server name
    (python, go, clojure, racket) to avoid polluting channel names with
    temp directory names.
    """
    # Normalize: lowercase, replace hyphens/spaces with underscores
    name = xml_path.stem.lower().replace("-", "_").replace(" ", "_")
    parent = xml_path.parent.name.lower().replace("-", "_").replace(" ", "_")
    if parent in _KNOWN_DISCO_PARENTS:
        return f"{parent}_{name}" if name not in parent else parent
    return name


def _parse_xml(xml_path: Path) -> list[Conversation]:
    """Parse a single DISCO XML file into Conversation objects.

    XML format:
        <message conversation_id="1">
            <ts>2019-12-02T07:28:19.806000</ts>
            <user>Finley</user>
            <text>What does self and init actually mean?</text>
        </message>

    Returns conversations with >= MIN_MESSAGES messages.
    """
    channel = _channel_from_path(xml_path)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        logger.warning("Failed to parse XML: %s", xml_path)
        return []

    # Group messages by conversation_id
    conv_messages: dict[str, list[dict]] = defaultdict(list)
    for msg_elem in root.iter("message"):
        conv_id_attr = msg_elem.get("conversation_id", "0")

        ts_elem = msg_elem.find("ts")
        user_elem = msg_elem.find("user")
        text_elem = msg_elem.find("text")

        if ts_elem is None or user_elem is None or text_elem is None:
            continue

        ts = ts_elem.text or ""
        user = user_elem.text or "unknown"
        text = text_elem.text or ""

        if not text.strip():
            continue

        conv_messages[conv_id_attr].append({
            "ts": ts,
            "user": user,
            "text": text,
        })

    # Convert to Conversation objects
    conversations: list[Conversation] = []
    for conv_id_attr, msgs in conv_messages.items():
        if len(msgs) < MIN_MESSAGES:
            continue

        conv_id = f"disco_{channel}_{conv_id_attr}"
        messages = []
        for i, m in enumerate(msgs, 1):
            messages.append(Message(
                message_id=f"{conv_id}_msg_{i:03d}",
                author=m["user"],
                timestamp=m["ts"],
                content=m["text"],
            ))

        conversations.append(Conversation(
            conversation_id=conv_id,
            channel=channel,
            title="",
            summary="",
            topic_tags=[],
            participants=list({m["user"] for m in msgs}),
            messages=messages,
            cross_references=[],
            platform="discord",
            phase="real",
            confounder_for="",
        ))

    return conversations


def _limit_per_channel(
    conversations: list[Conversation],
    max_per_channel: int,
) -> list[Conversation]:
    """Limit the number of conversations per channel."""
    by_channel: dict[str, list[Conversation]] = defaultdict(list)
    for conv in conversations:
        by_channel[conv.channel].append(conv)

    result: list[Conversation] = []
    for channel, convs in by_channel.items():
        result.extend(convs[:max_per_channel])

    return result
