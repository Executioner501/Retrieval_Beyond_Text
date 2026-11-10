"""
Tests for WhatsApp chat parser (src.parser.parse_whatsapp).
Covers:
- Message count after filtering
- Multi-line merge
- Edited-marker removal
- Media-only messages dropped
- System lines dropped
- Sender and text splitting
- 12-hour AM/PM timestamp support
"""

from src.parser.parse_whatsapp import parse_whatsapp_chat


def test_message_count_after_filtering(mini_chat_path):
    # With filter_system=False: 1 system line + 1 media-only + 14 normal messages = 16 total
    all_records = parse_whatsapp_chat(str(mini_chat_path), filter_system=False)
    assert len(all_records) == 16

    # With filter_system=True: 14 clean user messages
    clean_records = parse_whatsapp_chat(str(mini_chat_path), filter_system=True)
    assert len(clean_records) == 14


def test_system_lines_dropped(mini_chat_path):
    records_unfiltered = parse_whatsapp_chat(str(mini_chat_path), filter_system=False)
    system_msgs = [r for r in records_unfiltered if r.is_system]
    assert len(system_msgs) >= 1
    assert any("end-to-end encrypted" in r.text for r in system_msgs)

    # When filtered, no system messages should remain
    records_filtered = parse_whatsapp_chat(str(mini_chat_path), filter_system=True)
    assert not any(r.is_system for r in records_filtered)
    assert not any("end-to-end encrypted" in r.text for r in records_filtered)


def test_multi_line_merge(mini_chat_messages):
    # Message by Alice at 09:34 was multi-line
    multi_line_msgs = [m for m in mini_chat_messages if "\n" in m.text]
    assert len(multi_line_msgs) == 1
    m = multi_line_msgs[0]
    assert m.sender == "Alice"
    assert "preliminary statistics" in m.text
    assert "second line of details" in m.text


def test_edited_marker_removal(mini_chat_messages):
    # Message by Charlie had "<This message was edited>"
    matching = [m for m in mini_chat_messages if "Meeting room is 402" in m.text]
    assert len(matching) == 1
    m = matching[0]
    assert "<This message was edited>" not in m.text
    assert m.text == "Meeting room is 402"


def test_media_only_messages_dropped(mini_chat_path):
    records_unfiltered = parse_whatsapp_chat(str(mini_chat_path), filter_system=False)
    media_records = [r for r in records_unfiltered if r.has_media]
    assert len(media_records) >= 1
    assert any("<Media omitted>" in r.text for r in media_records)

    # When filtered, media-only lines must be discarded
    records_filtered = parse_whatsapp_chat(str(mini_chat_path), filter_system=True)
    assert not any(r.text.strip() == "<Media omitted>" for r in records_filtered)


def test_sender_and_text_split(mini_chat_messages):
    expected_senders = {"Alice", "Bob", "Charlie"}
    senders_found = {m.sender for m in mini_chat_messages}
    assert senders_found == expected_senders

    for m in mini_chat_messages:
        assert m.sender in expected_senders
        assert len(m.text.strip()) > 0
        assert m.word_count > 0
        assert m.char_count == len(m.text)
        assert not m.is_system


def test_12_hour_am_pm_timestamps(mini_chat_12h_path):
    records = parse_whatsapp_chat(str(mini_chat_12h_path), filter_system=True)
    assert len(records) == 4
    senders = [r.sender for r in records]
    assert senders == ["Alice", "Bob", "Charlie", "Alice"]
    # Verify timestamps are parsed into valid strings
    for r in records:
        assert len(r.timestamp) > 0
        assert r.time in ("09:31 AM", "09:32 AM", "02:15 PM", "02:20 PM")
