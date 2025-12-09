"""Lightweight helpers for extracting JSON from LLM outputs."""
from __future__ import annotations

import json


def strip_code_fences(text: str) -> str:
    """Remove surrounding ``` fences (with optional language tag)."""
    text = (text or "").strip()
    if "```" not in text:
        return text
    first = text.find("```")
    last = text.rfind("```")
    if first == -1 or last == -1 or last <= first:
        return text
    inner = text[first + 3 : last].strip()
    if "\n" in inner:
        first_line, rest = inner.split("\n", 1)
        if first_line.strip().isalpha():
            return rest.strip()
    return inner.strip()


def extract_first_json(text: str) -> str:
    """
    Best-effort extraction of the first JSON object:
    - Remove code fences and other wrappers
    - Find the first '{', track depth with curly braces
    - If missing the last '}', but has a closing ']', try adding a '}'
    """
    # assume you have strip_code_fences; if not, you can just use text.strip()
    stripped = strip_code_fences(text) if text is not None else ""
    stripped = (stripped or "").strip()

    # sometimes the whole string is wrapped in single quotes: '{"queries": [...]}'
    if stripped.startswith("'") and stripped.endswith("'") and len(stripped) >= 2:
        stripped = stripped[1:-1].strip()

    # find the first '{'
    start = stripped.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escape = False

    # scan until the first '{' depth goes back to 0
    for i, ch in enumerate(stripped[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # found the first complete '{...}'
                    return stripped[start : i + 1]

    # if we get here, the '{' was not closed; handle cases like:
    # '{"queries": ["...", "...", "..."]'
    # which are missing a '}'
    if '"queries"' in stripped[start:]:
        bracket_end = stripped.rfind("]")
        if bracket_end != -1 and bracket_end > start:
            return stripped[start : bracket_end + 1] + "}"

    return ""

__all__ = ["strip_code_fences", "extract_first_json"]
