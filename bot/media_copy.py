"""Safe first-person rendering for trusted media catalog metadata."""

from __future__ import annotations

import re


# Controlled catalog descriptions use these nouns after possessive ``her``.
# Everywhere else, ``<preposition> her`` is an object phrase and must become
# ``<preposition> me`` (for example, ``a photo of her in bed``).
_POSSESSED_NOUNS = (
    r"bed|bedroom|bathroom|kitchen|shower|gym|locker(?:\s+room)?|bar|"
    r"stockroom|club|concert|beach|car|hotel|home|house|room|couch|shift|"
    r"outfit|clothes|dress|lingerie|bikini|towel|body|face|boobs?|breasts?|"
    r"pussy|ass|legs?|feet|hair|smile|phone|mirror|photo|picture|pic|selfie|"
    r"video|clip|content|file|friend|boyfriend|customers?|coworkers?"
)

_OBJECT_HER_AFTER_PREPOSITION_RE = re.compile(
    rf"\b(?P<preposition>of|with|for|to|around|behind|beside|near|from|by)"
    rf"\s+her\b(?!\s+(?:{_POSSESSED_NOUNS})\b)",
    re.IGNORECASE,
)

_OBJECT_MIA_AFTER_PREPOSITION_RE = re.compile(
    r"\b(?P<preposition>of|with|for|to|around|behind|beside|near|from|by)"
    r"\s+Mia\b",
    re.IGNORECASE,
)


def _clean(value: object, *, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def spoken_item_description(description: object) -> str:
    """Convert neutral item metadata into safe first-person copy for Mia."""

    value = _clean(description, limit=240)
    value = re.sub(r"\bMia(?:'s|\u2019s)\b", "my", value, flags=re.IGNORECASE)
    value = _OBJECT_MIA_AFTER_PREPOSITION_RE.sub(
        r"\g<preposition> me", value
    )
    subject_replacements = (
        (r"\bMia\s+is\b", "I am"),
        (r"\bMia\s+has\s+not\b", "I have not"),
        (r"\bMia\s+has\b", "I have"),
        (r"\bMia\s+does\s+not\b", "I do not"),
        (r"\bMia\s+does\b", "I do"),
        (r"\bMia\s+was\b", "I was"),
    )
    for pattern, replacement in subject_replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    value = re.sub(
        r"\bMia\s+(?=(?:took|made|shot|filmed|recorded|saved|kept|sent|"
        r"picked|chose|posed)\b)",
        "I ",
        value,
        flags=re.IGNORECASE,
    )
    value = _OBJECT_HER_AFTER_PREPOSITION_RE.sub(
        r"\g<preposition> me", value
    )
    value = re.sub(
        r"\b(shows?|showing|features?|featuring|captures?|capturing)\s+her\b",
        r"\1 me",
        value,
        flags=re.IGNORECASE,
    )
    replacements = (
        (r"\bshe\s+has\s+not\b", "I have not"),
        (r"\bshe\s+does\s+not\b", "I do not"),
        (r"\bshe\s+was\s+not\b", "I was not"),
        (r"\bshe(?:'|\u2019)s\b", "I'm"),
        (r"\bshe\s+hasn(?:'|\u2019)t\b", "I haven't"),
        (r"\bshe\s+doesn(?:'|\u2019)t\b", "I don't"),
        (r"\bshe\s+wasn(?:'|\u2019)t\b", "I wasn't"),
        (r"\bshe\s+is\b", "I am"),
        (r"\bshe\s+has\b", "I have"),
        (r"\bshe\s+does\b", "I do"),
        (r"\bshe\s+was\b", "I was"),
        (r"\bshe\b", "I"),
        (r"\bherself\b", "me"),
        (r"\bhers\b", "mine"),
        (r"\bher\b", "my"),
        (r"\bMia\b", "me"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value.strip(" ,.;")


def spoken_fallback_context(context: object) -> str:
    """Convert a trusted planner reason to Mia's voice without changing it."""

    value = _clean(context, limit=320)
    value = re.sub(r"\bMia(?:'s|\u2019s)\b", "my", value, flags=re.IGNORECASE)
    value = _OBJECT_MIA_AFTER_PREPOSITION_RE.sub(
        r"\g<preposition> me", value
    )
    value = _OBJECT_HER_AFTER_PREPOSITION_RE.sub(
        r"\g<preposition> me", value
    )
    replacements = (
        (r"\b(?:Mia|she)\s+hasn(?:'|\u2019)t\b", "I haven't"),
        (r"\b(?:Mia|she)\s+doesn(?:'|\u2019)t\b", "I don't"),
        (r"\b(?:Mia|she)\s+wasn(?:'|\u2019)t\b", "I wasn't"),
        (r"\b(?:Mia|she)(?:'|\u2019)s\b", "I'm"),
        (r"\b(?:Mia|she)\s+is\b", "I am"),
        (r"\b(?:Mia|she)\s+has\s+not\b", "I have not"),
        (r"\b(?:Mia|she)\s+has\b", "I have"),
        (r"\b(?:Mia|she)\s+does\s+not\b", "I do not"),
        (r"\b(?:Mia|she)\s+does\b", "I do"),
        (r"\b(?:Mia|she)\s+was\b", "I was"),
        (r"\b(?:Mia|she)\b", "I"),
        (r"\bherself\b", "me"),
        (r"\bhers\b", "mine"),
        (r"\bher\b", "my"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    return value.strip(" ,.;")


__all__ = ["spoken_fallback_context", "spoken_item_description"]
