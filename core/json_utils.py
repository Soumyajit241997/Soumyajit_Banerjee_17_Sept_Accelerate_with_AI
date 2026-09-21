"""Best-effort JSON extraction from LLM free-text responses."""
from __future__ import annotations

import json
import re


def message_content_to_text(content) -> str:
    """Normalise a LangChain message's `.content` into plain text.

    Anthropic models with extended thinking enabled (e.g. claude-sonnet-5) return
    `.content` as a LIST of content blocks — a "thinking" block plus a "text"
    block — rather than a plain string. Treating that list as opaque and handing
    it straight to json.loads (or worse, returning it as-is) silently produces
    garbage: exactly what happened here, where a `{"thinking": ...}` /
    `{"text": "[...]"}` block pair got serialized as two blank STTM rows instead
    of the real JSON array inside the text block. Only "text" blocks are kept;
    thinking/redacted_thinking/tool_use blocks are dropped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def extract_json(content):
    """Pull the first valid JSON object/array out of an LLM response.

    `content` may be a plain string or a LangChain message's `.content` (which
    can be a list of content blocks — see `message_content_to_text`). LLMs also
    frequently wrap JSON in prose or ```json fences even when told not to. Try,
    in order: a direct parse, a fenced code block, then the widest {...} or
    [...] span.
    """
    text = message_content_to_text(content).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]!r}")
