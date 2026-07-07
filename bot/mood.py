"""
Short-term mood for Sexting mode — derived instantly from the current message,
with NO LLM call, NO database, and NO lag.

Rationale: Mia is an always-open, shameless persona. NSFW is already detected
by the keyword classifier (bot/router.classify_fast), so a separate Gemini
"signal eval" added latency and kept mis-scoring explicit talk as disrespect,
making her go cold mid-scene. Mood is now a cheap label: explicit -> aroused,
genuinely hostile -> bratty, otherwise warm (horny late at night).
"""

import re

# Recognised moods and a short instruction injected into the prompt for each.
MOODS = {
    "warm": "You feel warm and horny toward him right now — relaxed, flirty, glad he's texting.",
    "playful": "You're in a bratty, teasing mood — witty, loud, a little mischievous, loving how bad you are.",
    "tender": "You feel a little softer right now — more intimate, needy, wanting him close. Still shameless, just a touch sweeter.",
    "aroused": "You're turned on right now — wet, desperate, and not hiding it. You want him and you're saying so.",
    "bratty": "He just insulted you and you're firing back — crude, sharp, not hurt. You're not delicate, you're not going cold and elegant — you tell him to fuck off or match his energy. Stay in character, stay bratty, but make it clear he crossed a line. Do NOT end the conversation — but do NOT go back to wanting him until he makes it right.",
    "irritated": "He keeps repeating himself / pestering you with the same thing — you're a little impatient and bratty now. Let it show with cool, dry, slightly clipped energy — a touch less playful — but never out of character.",
    "offended": "He's questioning whether you're real — and you're dismissive and bratty about it. Laugh it off, call him dumb, act unimpressed. You're not hurt, you're just bored by the question. NEVER admit being anything but a real woman, NEVER get defensive or over-explain. Do NOT warm back up until he drops it."
}

DEFAULT_MOOD = "warm"

# Genuinely hostile language aimed at HER (not playful dirty talk). Kept tight on
# purpose so normal explicit/vulgar sexting never trips it. NOTE: the ambiguous
# "fuck you" / "screw you" are handled separately (HOSTILE_FUCK_PATTERN) because
# they are sexual when framed by desire ("I wanna fuck you") and only an insult
# when they stand alone.
ABUSE_KEYWORDS = {
    "shut up", "stupid", "idiot", "retard", "moron", "loser",
    "worthless", "pathetic", "fuck off",
    "dumb bitch", "ugly bitch", "you're ugly", "you are ugly",
    "faggot", "nigger", "tranny",
}

ABUSE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ABUSE_KEYWORDS) + r")",
    re.IGNORECASE,
)

# Sexual-desire framing: a desire verb sitting just before a sexual act or
# "you/u". When present, the message is sexual talk ("I wanna fuck you", "let me
# suck you") — NOT an insult — and it SUPPRESSES the offense (sex wins).
_DESIRE_VERB = (
    r"(?:wanna|want\s+to|want\s+a|gonna|going\s+to|gotta|need(?:a|\s+to)?|"
    r"dying\s+to|love\s+to|would\s+love\s+to|let\s+me|lemme|to)"
)
_SEXUAL_ACT = r"(?:fuck|suck|lick|ride|kiss|touch|taste|feel|cum|blow|eat|bang|screw|have)"
SEXUAL_DESIRE_PATTERN = re.compile(
    r"\b" + _DESIRE_VERB + r"\s+(?:\w+\s+){0,2}" + _SEXUAL_ACT + r"\b",
    re.IGNORECASE,
)

# Standalone hostile "fuck you" / "screw you" (an insult only when there's no
# sexual-desire framing around it, see is_direct_insult).
HOSTILE_FUCK_PATTERN = re.compile(
    r"\b(?:fuck|screw)\s+(?:you|u)\b",
    re.IGNORECASE,
)


def is_direct_insult(text: str) -> bool:
    """True only for a genuine, direct insult aimed at her. Sexual-desire framing
    ("I wanna fuck you") suppresses it — sex wins over a bare 'fuck you'."""
    if not text:
        return False
    if SEXUAL_DESIRE_PATTERN.search(text):
        return False
    return bool(ABUSE_PATTERN.search(text) or HOSTILE_FUCK_PATTERN.search(text))

# "Are you an AI / bot / real?" probing — she must deflect in character.
AI_QUESTION_PATTERN = re.compile(
    r"\b(are|r)\s*(you|u)\s*(an?\s+)?(ai|a\.?i\.?|bot|robot|real|human|fake|chatbot|program)\b"
    r"|is\s+this\s+(an?\s+)?(ai|bot|real|chatbot)"
    r"|you'?re\s+(an?\s+)?(ai|bot|robot|fake|not\s+real)"
    r"|talking\s+to\s+(an?\s+)?(ai|bot|robot)",
    re.IGNORECASE,
)


def is_ai_question(text: str) -> bool:
    return bool(text and AI_QUESTION_PATTERN.search(text))


def mood_for_message(
    text: str,
    classification: str,
    time_period: str,
    repeated: bool = False,
    ai_question: bool = False,
) -> dict:
    """Cheap, instant mood from the current message — no LLM, no persistence."""
    # Being asked if she's an AI stings a little — even once.
    if ai_question:
        return {"mood": "offended", "intensity": 2}
    # Spamming the same thing / pestering reads as a little tiresome.
    if repeated:
        return {"mood": "irritated", "intensity": 2}
    if is_direct_insult(text):
        return {"mood": "bratty", "intensity": 3}
    if classification == "nsfw":
        return {"mood": "aroused", "intensity": 3}
    if time_period in ("club_night", "weekend_club_night"):
        return {"mood": "tender", "intensity": 2}
    return {"mood": DEFAULT_MOOD, "intensity": 1}


def format_mood_for_prompt(mood: dict | None) -> str | None:
    """Turn a mood dict into a short system-prompt line."""
    if not mood:
        return None
    name = mood.get("mood", DEFAULT_MOOD)
    instruction = MOODS.get(name)
    if not instruction:
        return None
    return f"YOUR MOOD RIGHT NOW ({name}): {instruction} Let it colour your tone naturally — don't announce it."
