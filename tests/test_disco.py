"""Tests for DISCO dataset downloader/converter."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from chat_bench.disco import MIN_MESSAGES, _channel_from_path, _parse_xml


def _write_xml(tmp_path: Path, filename: str, messages: list[dict]) -> Path:
    """Write a minimal DISCO XML file for testing."""
    xml_path = tmp_path / filename
    root = ET.Element("messages")
    for msg in messages:
        m = ET.SubElement(root, "message", conversation_id=str(msg["conv_id"]))
        ET.SubElement(m, "ts").text = msg.get("ts", "2020-01-01T00:00:00")
        ET.SubElement(m, "user").text = msg.get("user", "Alice")
        ET.SubElement(m, "text").text = msg.get("text", "hello")
    tree = ET.ElementTree(root)
    tree.write(xml_path, xml_declaration=True, encoding="unicode")
    return xml_path


class TestParseXml:
    def test_produces_correct_conversations(self, tmp_path):
        """_parse_xml creates Conversation objects from XML messages."""
        messages = [
            {"conv_id": "1", "user": "Alice", "text": "What is Python?", "ts": "2020-01-01T10:00:00"},
            {"conv_id": "1", "user": "Bob", "text": "A programming language", "ts": "2020-01-01T10:01:00"},
            {"conv_id": "1", "user": "Alice", "text": "Thanks!", "ts": "2020-01-01T10:02:00"},
            {"conv_id": "2", "user": "Carol", "text": "How do I use pip?", "ts": "2020-01-01T11:00:00"},
            {"conv_id": "2", "user": "Dave", "text": "pip install package", "ts": "2020-01-01T11:01:00"},
            {"conv_id": "2", "user": "Carol", "text": "Got it", "ts": "2020-01-01T11:02:00"},
        ]
        xml_path = _write_xml(tmp_path, "test_channel.xml", messages)
        convs = _parse_xml(xml_path)

        assert len(convs) == 2
        assert convs[0].conversation_id.startswith("disco_")
        assert convs[0].platform == "discord"
        assert convs[0].phase == "real"
        assert convs[0].title == ""
        assert convs[0].summary == ""
        assert len(convs[0].messages) == 3

    def test_ids_are_namespaced(self, tmp_path):
        """Conversation IDs are prefixed with disco_{channel}_{conv_id}."""
        messages = [
            {"conv_id": "42", "user": "Alice", "text": f"msg {i}"}
            for i in range(3)
        ]
        xml_path = tmp_path / "python_general.xml"
        _write_xml(tmp_path, "python_general.xml", messages)
        convs = _parse_xml(xml_path)

        assert len(convs) == 1
        assert convs[0].conversation_id == "disco_python_general_42"

    def test_short_conversations_filtered(self, tmp_path):
        """Conversations with fewer than MIN_MESSAGES messages are filtered out."""
        messages = [
            {"conv_id": "1", "user": "Alice", "text": "hi"},
            {"conv_id": "1", "user": "Bob", "text": "bye"},
            # conv_id=2 has enough messages
            {"conv_id": "2", "user": "Alice", "text": "msg 1"},
            {"conv_id": "2", "user": "Bob", "text": "msg 2"},
            {"conv_id": "2", "user": "Carol", "text": "msg 3"},
        ]
        xml_path = _write_xml(tmp_path, "test.xml", messages)
        convs = _parse_xml(xml_path)

        assert len(convs) == 1
        assert "2" in convs[0].conversation_id

    def test_empty_text_messages_skipped(self, tmp_path):
        """Messages with empty text are skipped."""
        messages = [
            {"conv_id": "1", "user": "Alice", "text": "msg 1"},
            {"conv_id": "1", "user": "Bob", "text": ""},
            {"conv_id": "1", "user": "Carol", "text": "msg 3"},
            {"conv_id": "1", "user": "Dave", "text": "msg 4"},
        ]
        xml_path = _write_xml(tmp_path, "test.xml", messages)
        convs = _parse_xml(xml_path)

        assert len(convs) == 1
        assert len(convs[0].messages) == 3

    def test_message_ids_sequential(self, tmp_path):
        """Message IDs within a conversation are sequential."""
        messages = [
            {"conv_id": "1", "user": "Alice", "text": f"msg {i}"}
            for i in range(5)
        ]
        xml_path = _write_xml(tmp_path, "test.xml", messages)
        convs = _parse_xml(xml_path)

        msg_ids = [m.message_id for m in convs[0].messages]
        conv_id = convs[0].conversation_id
        assert msg_ids == [
            f"{conv_id}_msg_001",
            f"{conv_id}_msg_002",
            f"{conv_id}_msg_003",
            f"{conv_id}_msg_004",
            f"{conv_id}_msg_005",
        ]

    def test_participants_extracted(self, tmp_path):
        """Participants are the unique set of users."""
        messages = [
            {"conv_id": "1", "user": "Alice", "text": "a"},
            {"conv_id": "1", "user": "Bob", "text": "b"},
            {"conv_id": "1", "user": "Alice", "text": "c"},
        ]
        xml_path = _write_xml(tmp_path, "test.xml", messages)
        convs = _parse_xml(xml_path)

        assert set(convs[0].participants) == {"Alice", "Bob"}


class TestChannelFromPath:
    def test_simple_filename(self, tmp_path):
        path = tmp_path / "python_general.xml"
        assert _channel_from_path(path) == "python_general"

    def test_known_parent_included(self):
        path = Path("/data/python/general.xml")
        assert _channel_from_path(path) == "python_general"

    def test_unknown_parent_excluded(self, tmp_path):
        path = tmp_path / "some_channel.xml"
        assert _channel_from_path(path) == "some_channel"

    def test_hyphenated_name(self, tmp_path):
        path = tmp_path / "some-channel.xml"
        assert _channel_from_path(path) == "some_channel"


@pytest.mark.network
def test_download_works():
    """Integration test: DISCO download produces conversations."""
    from chat_bench.disco import get_disco_conversations
    convs = get_disco_conversations(max_per_channel=5)
    assert len(convs) > 0
    assert all(c.platform == "discord" for c in convs)
    assert all(c.phase == "real" for c in convs)
    assert all(c.conversation_id.startswith("disco_") for c in convs)
