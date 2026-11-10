"""
Module 1: WhatsApp Chat Parser & Normalizer.
Handles 12-hour and 24-hour timestamps, multi-line continuations,
and filters noise (encryption notices, media omitted).
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import re
import json
import logging

logger = logging.getLogger(__name__)

# Regex patterns covering common WhatsApp export variants (Android & iOS)
TIMESTAMP_PATTERNS = [
    # 10/01/2026, 09:30 - Sender: Message
    re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\s*-\s*(.+)$"),
    # [10/01/2026, 09:30:15] Sender: Message
    re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\]\s*(.+)$"),
]

# Lines WITHOUT a "Sender: " part are always system notices (joins, leaves, encryption
# notices, "X is a contact", ...). The patterns below are only for lines that DO have
# a sender but carry no real content. They must match the WHOLE text, otherwise real
# messages such as "I added the text" or "No time left" get dropped.
SYSTEM_PATTERNS = [
    re.compile(r"^You deleted this message\.?$", re.IGNORECASE),
    re.compile(r"^This message was deleted\.?$", re.IGNORECASE),
    re.compile(r"^null$", re.IGNORECASE),  # view-once media placeholder
    re.compile(r"^Messages and calls are end-to-end encrypted.*$", re.IGNORECASE),
]

EDITED_MARKER = re.compile(r"\s*<This message was edited>\s*$", re.IGNORECASE)

MEDIA_OMITTED_PATTERN = re.compile(r"<(?:Media|document|image|video|audio|attached|Contact card|Location) omitted(?:\s+[^>]+)?>", re.IGNORECASE)


@dataclass
class MessageRecord:
    message_id: str
    timestamp: str
    date: str
    time: str
    sender: str
    text: str
    is_system: bool
    has_media: bool
    char_count: int
    word_count: int

    def to_dict(self):
        return asdict(self)


def _detect_day_first(lines: List[str]) -> bool:
    has_first_gt_12 = False
    has_second_gt_12 = False
    for line in lines:
        for pat in TIMESTAMP_PATTERNS:
            m = pat.match(line.rstrip("\r\n"))
            if m:
                date_str = m.group(1)
                parts = re.split(r"[/.-]", date_str)
                if len(parts) >= 2:
                    try:
                        n1, n2 = int(parts[0]), int(parts[1])
                        if n1 > 12:
                            has_first_gt_12 = True
                        if n2 > 12:
                            has_second_gt_12 = True
                    except ValueError:
                        pass
                break
    if has_first_gt_12:
        return True
    if has_second_gt_12:
        return False
    return True


def parse_whatsapp_chat(filepath: str, filter_system: bool = True) -> List[MessageRecord]:
    """
    Parses a raw WhatsApp .txt export into structured MessageRecords.
    Correctly merges multi-line messages without timestamps into the parent turn.
    """
    records: List[MessageRecord] = []
    current_record: Optional[dict] = None
    msg_counter = 0

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    day_first = _detect_day_first(lines)
    total_messages = 0
    failed_count = 0

    def _flush_current(curr: dict):
        nonlocal total_messages, failed_count
        total_messages += 1
        rec = _finalize_record(curr, day_first=day_first)
        if rec is None:
            failed_count += 1
            logger.warning("Unparseable timestamp in message %s: '%s %s'", curr.get("id"), curr.get("date"), curr.get("time"))
        else:
            records.append(rec)

    for line in lines:
        raw_line = line.rstrip("\r\n")
        if not raw_line.strip():
            continue

        matched = False
        for pat in TIMESTAMP_PATTERNS:
            match = pat.match(raw_line)
            if match:
                date_str, time_str, body = match.groups()
                matched = True

                # Flush previous record
                if current_record is not None:
                    _flush_current(current_record)

                msg_counter += 1
                sender = "System"
                text = body.strip()
                is_system = False

                if ": " in body:
                    sender_part, text_part = body.split(": ", 1)
                    sender = sender_part.strip()
                    text = text_part.strip()
                else:
                    is_system = True

                text = EDITED_MARKER.sub("", text)
                for sys_pat in SYSTEM_PATTERNS:
                    if sys_pat.match(text):
                        is_system = True
                        break

                has_media = bool(MEDIA_OMITTED_PATTERN.search(text))

                current_record = {
                    "id": f"msg_{msg_counter:05d}",
                    "date": date_str,
                    "time": time_str,
                    "sender": sender,
                    "text": text,
                    "is_system": is_system,
                    "has_media": has_media,
                }
                break

        if not matched and current_record is not None:
            # Continuation line of previous message
            current_record["text"] += "\n" + raw_line

    if current_record is not None:
        _flush_current(current_record)

    if total_messages > 0 and (failed_count / total_messages) > 0.01:
        raise ValueError(
            f"Failed to parse timestamps for {failed_count} of {total_messages} messages "
            f"({100 * failed_count / total_messages:.1f}% > 1%)"
        )

    if filter_system:
        records = [r for r in records if not r.is_system]
        # Also drop messages that are PURELY a media placeholder (no other text).
        # A message that mentions media alongside real text (e.g. a caption) is kept.
        records = [
            r for r in records
            if not (r.has_media and MEDIA_OMITTED_PATTERN.sub("", r.text).strip() == "")
        ]

    return records


def _finalize_record(data: dict, day_first: bool = True) -> Optional[MessageRecord]:
    text = EDITED_MARKER.sub("", data["text"])
    dt_str = f"{data['date']} {data['time']}"

    if day_first:
        formats = (
            "%d/%m/%Y, %H:%M", "%d/%m/%Y %H:%M", "%d/%m/%y, %H:%M", "%d/%m/%y %H:%M",
            "%d/%m/%Y, %I:%M %p", "%d/%m/%Y %I:%M %p", "%d/%m/%y, %I:%M %p", "%d/%m/%y %I:%M %p",
            "%d/%m/%Y, %I:%M:%S %p", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%y, %I:%M:%S %p", "%d/%m/%y %I:%M:%S %p",
            "%d/%m/%Y, %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%y, %H:%M:%S", "%d/%m/%y %H:%M:%S",
            "%d/%m/%Y, %I:%M%p", "%d/%m/%Y %I:%M%p", "%d/%m/%y, %I:%M%p", "%d/%m/%y %I:%M%p",
        )
    else:
        formats = (
            "%m/%d/%Y, %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%y, %H:%M", "%m/%d/%y %H:%M",
            "%m/%d/%Y, %I:%M %p", "%m/%d/%Y %I:%M %p", "%m/%d/%y, %I:%M %p", "%m/%d/%y %I:%M %p",
            "%m/%d/%Y, %I:%M:%S %p", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%y, %I:%M:%S %p", "%m/%d/%y %I:%M:%S %p",
            "%m/%d/%Y, %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%y, %H:%M:%S", "%m/%d/%y %H:%M:%S",
            "%m/%d/%Y, %I:%M%p", "%m/%d/%Y %I:%M%p", "%m/%d/%y, %I:%M%p", "%m/%d/%y %I:%M%p",
        )

    iso_ts = None
    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            iso_ts = dt.isoformat()
            break
        except ValueError:
            continue

    if iso_ts is None:
        try:
            from dateutil import parser
            dt = parser.parse(dt_str, dayfirst=day_first)
            iso_ts = dt.isoformat()
        except Exception:
            iso_ts = None

    if iso_ts is None:
        return None

    return MessageRecord(
        message_id=data["id"],
        timestamp=iso_ts,
        date=data["date"],
        time=data["time"],
        sender=data["sender"],
        text=text,
        is_system=data["is_system"],
        has_media=data["has_media"],
        char_count=len(text),
        word_count=len(text.split()),
    )
