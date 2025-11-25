import re
from typing import Optional, Tuple


_POSITIVE_TOKENS = {
    "up",
    "upward",
    "increase",
    "increases",
    "increasing",
    "rise",
    "rises",
    "rising",
    "higher",
    "gain",
    "gains",
    "gaining",
    "bullish",
    "positive",
    "improve",
    "improves",
    "improving",
    "surge",
    "surges",
    "surging",
    "appreciate",
    "appreciates",
    "appreciating",
    "advance",
    "advances",
    "advancing",
    "strengthen",
    "strengthens",
    "strengthening",
}

_NEGATIVE_TOKENS = {
    "down",
    "downward",
    "decrease",
    "decreases",
    "decreasing",
    "decline",
    "declines",
    "declining",
    "fall",
    "falls",
    "falling",
    "lower",
    "drop",
    "drops",
    "dropping",
    "bearish",
    "negative",
    "weaken",
    "weakens",
    "weakening",
    "loss",
    "losing",
    "plunge",
    "plunges",
    "plunging",
    "selloff",
    "sell-off",
}

_DIRECTION_PATTERNS = [
    re.compile(
        r"stock\s*return[s]?\s*(?:will\s+be|would\s+be|is|=|:)?\s*(?:of\s*)?(?:[-+]?\d+(?:\.\d+)?\s*%?)?\s*(?:\(|\[)?\s*(?P<direction>up|down)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"stock\s*(?:return|price)[^.\n]{0,80}\b(?P<direction>up|down|increase|decrease|rise|fall|gain|drop|higher|lower|bullish|bearish|positive|negative)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:forecast|predict|expect|anticipate|project|outlook|therefore|overall)[^.]{0,80}\b(?P<direction>up|down|increase|decrease|rise|fall|gain|drop|higher|lower|bullish|bearish|positive|negative)\b",
        re.IGNORECASE,
    ),
]

_CONTEXT_TRIGGERS = (
    "stock return",
    "next day",
    "next-day",
    "tomorrow",
    "forecast",
    "predict",
    "expect",
    "anticipate",
    "project",
    "therefore",
    "overall",
    "in summary",
    "outlook",
    "hence",
)

_PERCENT_PATTERN = re.compile(r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*%")

_PERCENT_WITH_CONTEXT_PATTERN = re.compile(
    r"(?:stock\s*return[s]?|return\s+estimate)\s*(?:will\s+be|would\s+be|is|=|:)?\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def extract_stock_direction(text: Optional[str]) -> str:
    direction, _ = extract_stock_direction_and_value(text)
    return direction


def extract_stock_direction_and_value(text: Optional[str]) -> Tuple[str, Optional[float]]:
    """Return (direction, percent_value) parsed from text."""
    cleaned = _clean_text(text)
    if not cleaned:
        return "Unknown", None

    for pattern in _DIRECTION_PATTERNS:
        match = pattern.search(cleaned)
        if not match:
            continue
        direction_token = match.group("direction")
        direction = _token_to_direction(direction_token)
        if direction:
            value = _percent_near_index(cleaned, match.start())
            return direction, value

    contextual_direction = _direction_from_context(cleaned)
    if contextual_direction:
        value = _percent_near_index(cleaned, len(cleaned) - 200)
        return contextual_direction, value

    value_from_phrase = _percent_from_stock_return_phrase(cleaned)
    if value_from_phrase is not None and value_from_phrase != 0:
        direction = "Positive" if value_from_phrase > 0 else "Negative"
        return direction, value_from_phrase

    paren_direction = _paren_direction(cleaned)
    if paren_direction:
        value = _percent_near_index(cleaned, len(cleaned) - 200)
        return paren_direction, value

    tail_direction = _tail_direction(cleaned)
    if tail_direction:
        value = _percent_near_index(cleaned, len(cleaned) - 200)
        return tail_direction, value

    return "Unknown", None


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = text.replace("**", "").strip()
    return cleaned.lower()


def _token_to_direction(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    normalized = re.sub(r"[^a-z\-]", "", token.lower())
    if normalized in _POSITIVE_TOKENS:
        return "Positive"
    if normalized in _NEGATIVE_TOKENS:
        return "Negative"
    return None


def _direction_from_context(text: str) -> Optional[str]:
    sentences = re.split(r"[.!?\n]+", text)
    for sentence in reversed([s.strip() for s in sentences if s.strip()]):
        if not any(trigger in sentence for trigger in _CONTEXT_TRIGGERS):
            continue
        direction = _token_in_sentence(sentence)
        if direction:
            return direction
    return None


def _token_in_sentence(sentence: str) -> Optional[str]:
    tokens = re.findall(r"[a-z\-]+", sentence)
    for token in tokens:
        direction = _token_to_direction(token)
        if direction:
            return direction
    return None


def _paren_direction(text: str) -> Optional[str]:
    match = re.search(r"\((up|down)\)", text)
    if not match:
        return None
    return "Positive" if match.group(1) == "up" else "Negative"


def _tail_direction(text: str) -> Optional[str]:
    tail = text[-400:]
    return _token_in_sentence(tail)


def _percent_near_index(text: str, index: int, window: int = 160) -> Optional[float]:
    start = max(0, index - 40)
    end = min(len(text), index + window)
    snippet = text[start:end]
    match = _PERCENT_PATTERN.search(snippet)
    if not match:
        return None
    return _percent_str_to_float(match.group("value"))


def _percent_from_stock_return_phrase(text: str) -> Optional[float]:
    match = _PERCENT_WITH_CONTEXT_PATTERN.search(text)
    if not match:
        return None
    return _percent_str_to_float(match.group("value"))


def _percent_str_to_float(value_str: str) -> Optional[float]:
    if not value_str:
        return None
    try:
        return float(value_str)
    except ValueError:
        return None
