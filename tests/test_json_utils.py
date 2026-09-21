"""Regression test for the "found tables: []" bug: with extended thinking,
Anthropic's ChatAnthropic returns AIMessage.content as a LIST of content
blocks (a "thinking" block plus a "text" block) rather than a plain string.
extract_json used to check `isinstance(text, str)` and return non-strings
as-is, so the raw block list got serialized as garbage "STTM rows" instead
of the real JSON array sitting inside the text block -- exactly what a
production trace showed (a `{"thinking": ...}` / `{"text": "[...]"}` pair
turning into two blank STTM rows).
"""
from __future__ import annotations

import json

from core.json_utils import extract_json, message_content_to_text

REAL_STTM_ROWS = [
    {"source_schema": "raw", "source_table": "orders", "source_column": "order_id",
     "target_schema": "bronze", "target_table": "orders", "target_column": "order_id",
     "transformation_type": "direct", "transformation_logic": "straight copy"},
    {"source_schema": "raw", "source_table": "orders", "source_column": "order_date",
     "target_schema": "bronze", "target_table": "orders", "target_column": "order_date",
     "transformation_type": "indirect", "transformation_logic": "cast to datetime"},
]


def _anthropic_thinking_content(rows) -> list[dict]:
    """Shape observed in a real trace: AIMessage.content as a list of blocks."""
    return [
        {"signature": "abc123", "thinking": "", "type": "thinking"},
        {"text": json.dumps(rows), "type": "text"},
    ]


def test_message_content_to_text_extracts_only_text_blocks():
    content = _anthropic_thinking_content(REAL_STTM_ROWS)
    text = message_content_to_text(content)
    assert "thinking" not in text or json.loads(text) == REAL_STTM_ROWS
    assert json.loads(text) == REAL_STTM_ROWS


def test_message_content_to_text_passthrough_for_plain_string():
    assert message_content_to_text("hello") == "hello"


def test_extract_json_handles_anthropic_thinking_block_list():
    content = _anthropic_thinking_content(REAL_STTM_ROWS)
    rows = extract_json(content)
    assert rows == REAL_STTM_ROWS
    assert len(rows) == 2
    assert rows[0]["target_table"] == "orders"


def test_extract_json_still_handles_plain_string():
    assert extract_json(json.dumps(REAL_STTM_ROWS)) == REAL_STTM_ROWS


def test_extract_json_still_handles_fenced_code_block():
    text = "Here is the STTM:\n```json\n" + json.dumps(REAL_STTM_ROWS) + "\n```"
    assert extract_json(text) == REAL_STTM_ROWS
