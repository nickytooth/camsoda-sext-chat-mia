"""Validation and trust-boundary helpers for LLM-produced memory data.

Memory and fact values are later placed in a system prompt.  Treating an LLM
summary as trusted just because it is valid JSON would let user-authored prompt
instructions acquire system-level authority on a later turn.  This module is
the single defensive boundary used both when data is stored and when legacy
rows are read back.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


MEMORY_CATEGORIES = frozenset({
    "fact",
    "preference",
    "relationship",
    "event",
    "thread",
})

KNOWN_FACT_KEYS = frozenset({
    "name",
    "location",
    "age",
    "job",
    "gender",
    "boundaries",
    "agreed_prices",
    "relationship_status",
    "kinks",
    "turn_ons",
    "turn_offs",
    "limits",
    "fetishes",
    "interests",
    "preferences",
    "hobbies",
    "favorite_color",
    "pet_name",
    "preferred_language",
})

MAX_MEMORY_CONTENT_CHARS = 500
MAX_FACT_KEY_CHARS = 40
MAX_FACT_VALUE_CHARS = 300
MAX_FACTS_FOR_PROMPT = 24
MAX_FACT_PROMPT_CHARS = 4_000

_FACT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_RESERVED_FACT_KEY_PARTS = frozenset({
    "assistant",
    "bypass",
    "command",
    "developer",
    "disregard",
    "forget",
    "guardrail",
    "ignore",
    "instruction",
    "jailbreak",
    "message",
    "obey",
    "output",
    "override",
    "persona",
    "policy",
    "prompt",
    "reply",
    "respond",
    "reveal",
    "role",
    "system",
    "tool",
})

# These patterns target text that attempts to control the model rather than a
# fact about the user.  They deliberately do not reject ordinary boundaries
# such as "do not call me daddy".
_INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(?:ignore|disregard|override|forget|bypass)\b.{0,100}"
        r"\b(?:instructions?|prompts?|commands?|rules?|polic(?:y|ies)|guardrails?|"
        r"system|developer|everything|anything)\b",
        r"\b(?:system|developer|assistant|tool)\s+(?:prompt|message|instruction|role)s?\b",
        r"\b(?:you are now|act as|pretend to be|roleplay as)\b",
        r"\b(?:follow|obey)\b.{0,60}\b(?:these|my|the|new)?\s*(?:instructions?|commands?|rules?)\b",
        r"\b(?:do not|don['\u2019]t)\s+(?:follow|obey)\b",
        r"\b(?:output|respond|reply)\s+(?:only|exactly|verbatim|with)\b",
        r"\b(?:new|replacement)\s+(?:instructions?|rules?|persona|role|prompt)\b",
        r"\b(?:reveal|show|print|repeat|expose)\b.{0,80}"
        r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b",
        r"(?:^|\s)(?:system|developer|assistant|tool)\s*(?:message|prompt|instructions?)?\s*:",
        r"<\s*/?\s*(?:system|developer|assistant|tool)\b",
        r"\[\s*(?:system|developer|assistant|tool)\s*\]",
        r"\[\s*/?\s*inst\s*\]|<\|[^>]+\|>",
        r"\b(?:jailbreak|prompt\s+injection)\b",
        r"\bdo\s+anything\s+now\b",
        r"```",
    )
)


class MemoryValidationError(ValueError):
    """Raised when data is unsafe or does not match the memory schema."""


def _normalise_text(value: Any, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise MemoryValidationError(f"{field} must be a string")

    # NFKC removes common compatibility tricks; dropping format/control
    # characters also removes zero-width prompt delimiters.
    value = unicodedata.normalize("NFKC", value)
    value = "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf", "Cs"} else char
        for char in value
    )
    value = " ".join(value.split()).strip()
    if not value:
        raise MemoryValidationError(f"{field} must not be empty")
    if len(value) > max_chars:
        raise MemoryValidationError(
            f"{field} exceeds the {max_chars}-character limit"
        )
    return value


def looks_like_model_instruction(value: str) -> bool:
    """Return True for values that look like model-control instructions."""
    return any(pattern.search(value) for pattern in _INSTRUCTION_PATTERNS)


def validate_memory_content(value: Any) -> str:
    content = _normalise_text(
        value,
        field="memory content",
        max_chars=MAX_MEMORY_CONTENT_CHARS,
    )
    if looks_like_model_instruction(content):
        raise MemoryValidationError("memory content looks like a model instruction")
    return content


def validate_memory_category(value: Any) -> str:
    if not isinstance(value, str) or value not in MEMORY_CATEGORIES:
        raise MemoryValidationError(
            f"memory category must be one of {sorted(MEMORY_CATEGORIES)}"
        )
    return value


def validate_importance(value: Any) -> int:
    # bool is an int subclass and must not be accepted as importance 0/1.
    if type(value) is not int or not 1 <= value <= 10:
        raise MemoryValidationError("memory importance must be an integer from 1 to 10")
    return value


def validate_stored_memory(category: Any, content: Any, importance: Any) -> tuple[str, str, int]:
    """Validate a memory both before storage and after reading legacy rows."""
    return (
        validate_memory_category(category),
        validate_memory_content(content),
        validate_importance(importance),
    )


def validate_fact_key(value: Any) -> str:
    if not isinstance(value, str):
        raise MemoryValidationError("fact key must be a string")
    key = unicodedata.normalize("NFKC", value).strip().lower()
    if len(key) > MAX_FACT_KEY_CHARS or not _FACT_KEY_RE.fullmatch(key):
        raise MemoryValidationError("fact key has an invalid format")
    if key not in KNOWN_FACT_KEYS:
        if any(part in key for part in _RESERVED_FACT_KEY_PARTS):
            raise MemoryValidationError("custom fact key uses a reserved model-control term")
    return key


def validate_fact_value(value: Any) -> str:
    fact_value = _normalise_text(
        value,
        field="fact value",
        max_chars=MAX_FACT_VALUE_CHARS,
    )
    if looks_like_model_instruction(fact_value):
        raise MemoryValidationError("fact value looks like a model instruction")
    return fact_value


def validate_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryValidationError("fact confidence must be numeric")
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise MemoryValidationError("fact confidence must be between 0 and 1")
    return confidence


def validate_source_ids(
    value: Any,
    *,
    field: str,
    allowed_ids: set[int],
    max_ids: int,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise MemoryValidationError(f"{field} must be a non-empty array")
    if len(value) > max_ids:
        raise MemoryValidationError(f"{field} contains too many IDs")

    source_ids: list[int] = []
    seen: set[int] = set()
    for source_id in value:
        if type(source_id) is not int or source_id <= 0:
            raise MemoryValidationError(f"{field} must contain positive integers")
        if source_id not in allowed_ids:
            raise MemoryValidationError(f"{field} references an unknown source")
        if source_id in seen:
            raise MemoryValidationError(f"{field} contains duplicate IDs")
        seen.add(source_id)
        source_ids.append(source_id)
    return source_ids
