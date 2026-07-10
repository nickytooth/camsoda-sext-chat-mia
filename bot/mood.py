"""
Short-term mood for Sexting mode — derived from the current message with NO LLM
call and NO database, plus a small in-memory state so moods have INERTIA.

Rationale: Mia is an always-open, shameless persona. NSFW is already detected
by the keyword classifier (bot/router.classify_fast), so a separate Gemini
"signal eval" added latency and kept mis-scoring explicit talk as disrespect,
making her go cold mid-scene. Mood is a cheap label: explicit -> aroused,
genuinely hostile -> bratty, mentions of other women -> jealous, otherwise warm.

Inertia: a real person doesn't reset emotionally between texts. Negative moods
(bratty/offended/jealous/irritated) stick for a couple of turns and thaw one
step per calm message; arousal ramps up and cools down gradually. A long gap
(>1h) resets everything — new session, fresh start.
"""

import re
import time

# Recognised moods and a short instruction injected into the prompt for each.
MOODS = {
    "warm": "You feel warm and flirty toward him right now — relaxed, playful, a little forward, glad it's HIM texting and letting it show.",
    "playful": "You're in a bratty, teasing mood — witty, loud, a little mischievous, loving how bad you are.",
    "tender": "You feel a little softer right now — affectionate, wanting him close, still teasing but a touch sweeter.",
    "aroused": "You're turned on right now — wet, desperate, and not hiding it. You want him and you're saying so.",
    "bratty": "He just insulted you and you're firing back — crude, sharp, not hurt. You're not delicate, you're not going cold and elegant — you tell him to fuck off or match his energy. Stay in character, stay bratty, but make it clear he crossed a line. Do NOT end the conversation — but do NOT go back to wanting him until he makes it right.",
    "irritated": "He keeps repeating himself / pestering you with the same thing — you're a little impatient and bratty now. Let it show with cool, dry, slightly clipped energy — a touch less playful — but never out of character.",
    "offended": "He's questioning whether you're real — and you're dismissive and bratty about it. Laugh it off, call him dumb, act unimpressed. You're not hurt, you're just bored by the question. NEVER admit being anything but a real woman, NEVER get defensive or over-explain. Do NOT warm back up until he drops it.",
    "jealous": "He just mentioned another woman — and it got under your skin. You're possessive and competitive about it, hypocritically territorial (yes, YOU have Tyler — irrelevant, he's YOURS). Tease-interrogate him about her, one-up her shamelessly, remind him nobody does it like you. Bratty-jealous heat — never hurt, never insecure."
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


# A joking frame ("fuck off lol", "shut up 😂") reads as banter, not an attack.
LAUGH_PATTERN = re.compile(
    r"\b(lol|lmao|lmfao|haha+|hehe+|jk|j/k|kidding|joking)\b|😂|🤣|😹",
    re.IGNORECASE,
)


def is_direct_insult(text: str) -> bool:
    """True only for a genuine, direct insult aimed at her. Sexual-desire framing
    ("I wanna fuck you") suppresses it — sex wins over a bare 'fuck you'. A
    laughing marker in the same message reads as banter and suppresses it too."""
    if not text:
        return False
    if SEXUAL_DESIRE_PATTERN.search(text):
        return False
    if not (ABUSE_PATTERN.search(text) or HOSTILE_FUCK_PATTERN.search(text)):
        return False
    if LAUGH_PATTERN.search(text):
        return False
    return True


# Mentions of another woman in his life — fires her possessive streak.
JEALOUSY_PATTERN = re.compile(
    r"\b(my ex\b|my girlfriend|my wife|this girl|that girl|some girl|another girl|"
    r"other girls|a girl (i|at|from)|met a girl|on a date|went on a date|"
    r"dating (someone|a girl|this girl))",
    re.IGNORECASE,
)


def is_jealousy_trigger(text: str) -> bool:
    return bool(text and JEALOUSY_PATTERN.search(text))

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


# ---------------------------------------------------------------------------
# Mood inertia — tiny in-memory state per user (no LLM, no DB)
# ---------------------------------------------------------------------------

_NEGATIVE_MOODS = ("bratty", "offended", "irritated", "jealous")
# How many turns a freshly-triggered mood lingers before she's over it.
_STICKY_TURNS = {"bratty": 2, "offended": 2, "jealous": 2, "irritated": 1, "aroused": 3}
# A gap this long resets any lingering mood — new session, fresh start.
_MOOD_RESET_GAP_SECONDS = 3600

_mood_state: dict[int, dict] = {}


def clear_mood_state(user_id: int) -> None:
    """Drop lingering mood for a user (called on reset)."""
    _mood_state.pop(user_id, None)


def _set_mood_state(user_id: int, mood: str, intensity: int, now: float) -> dict:
    _mood_state[user_id] = {
        "mood": mood,
        "intensity": intensity,
        "turns_left": _STICKY_TURNS.get(mood, 2),
        "updated_at": now,
    }
    return {"mood": mood, "intensity": intensity}


def mood_for_message(
    user_id: int,
    text: str,
    classification: str,
    time_period: str,
    repeated: bool = False,
    ai_question: bool = False,
) -> dict:
    """Mood for this turn — instant triggers from the current message, blended
    with a lingering state so her emotions have inertia instead of resetting
    between texts."""
    now = time.time()
    state = _mood_state.get(user_id)
    if state and now - state.get("updated_at", 0) > _MOOD_RESET_GAP_SECONDS:
        _mood_state.pop(user_id, None)
        state = None

    # 1) Fresh triggers in THIS message set (or refresh) a sticky mood.
    if ai_question:
        return _set_mood_state(user_id, "offended", 2, now)
    if repeated:
        return _set_mood_state(user_id, "irritated", 2, now)
    if is_direct_insult(text):
        return _set_mood_state(user_id, "bratty", 3, now)
    if is_jealousy_trigger(text):
        return _set_mood_state(user_id, "jealous", 2, now)

    # 2) A lingering negative mood thaws ONE step per calm message — a nice
    #    text softens her, it doesn't instantly flip her back to warm.
    if state and state["mood"] in _NEGATIVE_MOODS:
        state["turns_left"] -= 1
        if state["turns_left"] <= 0:
            _mood_state.pop(user_id, None)
            state = None  # thawed — continue as a normal message below
        else:
            state["updated_at"] = now
            return {"mood": state["mood"], "intensity": max(1, state["intensity"] - 1)}

    # 3) Arousal ramps up while he keeps it explicit, and cools down gradually.
    if classification == "nsfw":
        prev = state["intensity"] if state and state["mood"] == "aroused" else 1
        return _set_mood_state(user_id, "aroused", min(3, prev + 1), now)
    if state and state["mood"] == "aroused":
        state["turns_left"] -= 1
        state["intensity"] = max(1, state["intensity"] - 1)
        if state["turns_left"] <= 0:
            _mood_state.pop(user_id, None)
        else:
            state["updated_at"] = now
            return {"mood": "aroused", "intensity": state["intensity"]}

    # 4) Baseline colour by time of day.
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
