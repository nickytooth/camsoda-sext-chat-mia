"""Deterministic text style fixes applied to every outgoing message.

The prompts already instruct the model that people's names are the ONE
exception to Mia's lowercase texting style, but Grok drops the ball often
enough (especially in openings) that the guarantee has to live in code:
whatever the model emits, the user never sees "tyler".
"""

import re

# Canonical spellings of every named character in the persona + content
# library (personas/mia.yaml, library/*.yaml). Keep in sync when new named
# characters are introduced.
CAST_NAMES = ("Mia", "Tyler", "Jess", "Cara", "Lena", "Ryan", "Marco")

_CANONICAL = {n.lower(): n for n in CAST_NAMES}

# Lookahead instead of a plain \b so the apostrophe-less possessive of
# texting style still matches: "tylers girl" -> "Tylers girl".
_NAME_RE = re.compile(
    r"\b(" + "|".join(CAST_NAMES) + r")(?='?s\b|\b)", re.IGNORECASE
)


def capitalize_names(text: str, extra_names: tuple = ()) -> str:
    """Rewrite any casing of a known character name to its canonical form
    ("tyler" / "TYLER" -> "Tyler"). Word-boundary matching keeps possessives
    working ("tyler's" / "tylers" -> "Tyler's" / "Tylers") without touching
    unrelated words. `extra_names` adds per-conversation names (the user's own
    name) — the model lowercases those just as happily as the fixed cast."""
    if not text:
        return text
    out = _NAME_RE.sub(lambda m: _CANONICAL[m.group(0).lower()], text)
    for name in extra_names:
        if not name or not name.strip():
            continue
        pat = re.compile(r"\b" + re.escape(name.strip()) + r"(?='?s\b|\b)", re.IGNORECASE)
        out = pat.sub(lambda m: capitalize_user_name(m.group(0)), out)
    return out


def capitalize_user_name(name: str) -> str:
    """First-letter capitalization for the user's own name ("nikolay" ->
    "Nikolay", "john smith" -> "John Smith"). Leaves the rest of each word
    untouched so "McKay" doesn't become "Mckay"."""
    if not name:
        return name
    return " ".join(w[:1].upper() + w[1:] if w else w for w in name.split(" "))
