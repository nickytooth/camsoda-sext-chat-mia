"""Fast, deterministic conversation-register classification.

This is deliberately a *heat* classifier, not a moderation layer.  A match
means that the user's wording is sexual enough to move the persona toward an
explicit register; prohibited content is handled separately by
``bot.moderation``.
"""

import logging
import re

logger = logging.getLogger(__name__)

_CHECK_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "\uff07": "'",
})


def _text_for_checks(text: str) -> str:
    """Normalise apostrophe variants for matching without rewriting content."""
    return text.translate(_CHECK_TRANSLATION)


# Terms in this set are matched as complete words/phrases.  The old pattern
# only had a boundary at the beginning, so innocent words such as "assistant",
# "cocktail" and "analytical" matched ``ass``, ``cock`` and ``anal``.
NSFW_KEYWORDS = {
    "sex", "fuck", "cock", "dick", "pussy", "ass", "tits", "boobs",
    "nipple", "cum", "orgasm", "horny", "naked", "nude", "blowjob",
    "bj", "handjob", "jerk off", "moan", "doggy", "missionary", "anal",
    "deepthroat", "lingerie", "bdsm", "spank", "choke", "dominate",
    "submissive", "kinky", "fetish", "dildo", "vibrator", "threesome",
    "erotic", "touch yourself", "touch me", "inside me", "inside you",
    "make love", "make me cum", "need you bad", "slut", "whore",
    "good girl", "dirty girl",
}

# These were intentionally stored as stems in the original keyword list.
# Make that behaviour explicit instead of relying on a missing end boundary.
_NSFW_STEMS = ("masturbat", "seduc")


def _phrase_pattern(phrase: str) -> str:
    """Return a complete-token regex for a literal word or phrase."""
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return rf"(?<!\w){escaped}(?!\w)"


_EXACT_PATTERN = "|".join(
    _phrase_pattern(keyword)
    for keyword in sorted(NSFW_KEYWORDS, key=len, reverse=True)
)
_STEM_PATTERN = "|".join(
    rf"(?<!\w){re.escape(stem)}[a-z]*(?!\w)" for stem in _NSFW_STEMS
)

# Common grammatical forms that should remain sexual without turning every
# keyword into an unsafe prefix match.
_INFLECTED_PATTERNS = (
    r"(?<!\w)fuck(?:s|ed|ing|er|ers)?(?!\w)",
    r"(?<!\w)sex(?:y|t(?:ed|ing)?|ual(?:ly|ity)?)?(?!\w)",
    r"(?<!\w)(?:cock|dick|nipple|orgasm|blowjob|handjob|fetish|dildo|vibrator|threesome)s?(?!\w)",
    r"(?<!\w)puss(?:y|ies)(?!\w)",
    r"(?<!\w)cum(?:s|med|ming)?(?!\w)",
    r"(?<!\w)moan(?:s|ed|ing)?(?!\w)",
    r"(?<!\w)spank(?:s|ed|ing)?(?!\w)",
    r"(?<!\w)chok(?:e|es|ed|ing)(?!\w)",
    r"(?<!\w)dominat(?:e|es|ed|ing|ion)(?!\w)",
)

# Ambiguous everyday words require a sexual construction instead of firing on
# the word alone ("oral history", "wet weather", "you suck at chess", etc.).
_CONTEXT_PATTERNS = (
    r"(?<!\w)oral\s+(?:sex|pleasure|play)(?!\w)",
    r"(?<!\w)(?:give|giving|gave|receive|receiving|get|getting|want)\s+(?:(?:me|you|him|her)\s+)?oral(?!\w)",
    r"(?<!\w)finger(?:ed|ing)\s+(?:me|you|myself|yourself|him|her)(?!\w)",
    r"(?<!\w)finger\s+(?:me|yourself|myself|him|her)(?!\w)",
    r"(?<!\w)(?:i(?:'m|\s+am)|you(?:'re|\s+are)|getting|got\s+me|make\s+me|so)\s+wet(?!\w)",
    r"(?<!\w)wet\s+(?:pussy|cock|dick|between\s+(?:my|your|her)\s+legs)(?!\w)",
    r"(?<!\w)hard\s+for\s+(?:me|you|him|her)(?!\w)",
    r"(?<!\w)(?:suck|sucking|lick|licking|ride|riding)\s+(?:me|you|my|your|his|her|it|that)(?!\w)",
    r"(?<!\w)strip\s+(?:for\s+)?(?:me|you|him|her)(?!\w)",
    r"(?<!\w)striptease(?!\w)",
    r"(?<!\w)want\s+you(?!\s+to\b)(?!\w)",
    r"(?<!\w)feel\s+you\s+(?:inside|on|against)(?!\w)",
)

NSFW_PATTERN = re.compile(
    rf"(?:{_EXACT_PATTERN}|{_STEM_PATTERN}|{'|'.join(_INFLECTED_PATTERNS)}|{'|'.join(_CONTEXT_PATTERNS)})",
    re.IGNORECASE,
)


# Explicit withdrawal must win over the sexual word inside the same sentence.
# Keep the aliases narrow and concrete so everyday phrases such as "stop by"
# do not alter the conversation register.
_WITHDRAWAL_ACT_ALIASES = (
    (re.compile(r"(?<!\w)fuck(?:s|ed|ing)?(?!\w)", re.IGNORECASE), "fuck"),
    (re.compile(r"(?<!\w)chok(?:e|ing)(?!\w)", re.IGNORECASE), "choke"),
    (re.compile(r"(?<!\w)spank(?:ing)?(?!\w)", re.IGNORECASE), "spank"),
    (re.compile(r"(?<!\w)slap(?:ping)?(?!\w)", re.IGNORECASE), "slap"),
    (re.compile(r"(?<!\w)hit(?:ting)?(?!\w)", re.IGNORECASE), "hit"),
    (re.compile(r"(?<!\w)bit(?:e|ing)(?!\w)", re.IGNORECASE), "bite"),
    (re.compile(r"(?<!\w)anal(?:\s+sex)?(?!\w)", re.IGNORECASE), "anal"),
    (re.compile(r"(?<!\w)oral(?:\s+sex)?(?!\w)", re.IGNORECASE), "oral"),
    (re.compile(r"(?<!\w)penetrat(?:e|ing|ion)(?!\w)", re.IGNORECASE), "penetration"),
    (re.compile(r"(?<!\w)finger(?:ing)?(?!\w)", re.IGNORECASE), "fingering"),
    (re.compile(r"(?<!\w)rough\s+sex(?!\w)", re.IGNORECASE), "rough sex"),
    (re.compile(r"(?<!\w)kiss(?:es|ed|ing)?(?!\w)", re.IGNORECASE), "kiss"),
    (re.compile(r"(?<!\w)lick(?:s|ed|ing)?(?!\w)", re.IGNORECASE), "lick"),
    (re.compile(r"(?<!\w)suck(?:s|ed|ing)?(?!\w)", re.IGNORECASE), "suck"),
    (re.compile(r"(?<!\w)touch(?:es|ed|ing)?(?!\w)", re.IGNORECASE), "touch"),
    (re.compile(r"(?<!\w)grab(?:s|bed|bing)?(?!\w)", re.IGNORECASE), "grab"),
    (re.compile(r"(?<!\w)pinch(?:es|ed|ing)?(?!\w)", re.IGNORECASE), "pinch"),
    (re.compile(r"(?<!\w)scratch(?:es|ed|ing)?(?!\w)", re.IGNORECASE), "scratch"),
    (re.compile(r"(?<!\w)spit(?:s|ting)?(?!\w)", re.IGNORECASE), "spit"),
    (
        re.compile(
            r"(?<!\w)(?:pull(?:s|ed|ing)?\s+(?:my\s+)?hair|hair\s+pulling)(?!\w)",
            re.IGNORECASE,
        ),
        "hair pulling",
    ),
    (
        re.compile(r"(?<!\w)(?:tie|tying)\s+(?:me\s+)?up(?!\w)", re.IGNORECASE),
        "tying up",
    ),
    (re.compile(r"(?<!\w)restrain(?:s|ed|ing)?(?!\w)", re.IGNORECASE), "restraint"),
)

_DIRECT_WITHDRAWAL_RE = re.compile(
    r"(?<!\w)(?:don't|dont|do\s+not|never)\s+(?:ever\s+)?",
    re.IGNORECASE,
)
_DONT_WANT_WITHDRAWAL_RE = re.compile(
    r"(?<!\w)(?:i\s+)?(?:don't|dont|do\s+not)\s+want\s+"
    r"(?:(?:you|him|her|them|someone)\s+to\s+)?",
    re.IGNORECASE,
)
_DONT_LIKE_WITHDRAWAL_RE = re.compile(
    r"(?<!\w)(?:i\s+)?(?:don't|dont|do\s+not)\s+like\s+",
    re.IGNORECASE,
)
_NOT_INTO_WITHDRAWAL_RE = re.compile(
    r"(?<!\w)i(?:'m|\s+am)\s+not\s+into\s+",
    re.IGNORECASE,
)
_NO_WITHDRAWAL_RE = re.compile(
    r"(?:^|[,.!?;/]\s*|\band\s+)no\s+(?:more\s+)?",
    re.IGNORECASE,
)
_STOP_WITHDRAWAL_RE = re.compile(r"(?<!\w)stop\s+(?:the\s+)?", re.IGNORECASE)
_STOP_ONLY_RE = re.compile(
    r"^\s*(?:(?:please|just)\s+)*(?:i\s+said\s+)?stop"
    r"(?:\s+(?:it|now|please))?[.!?\s]*$",
    re.IGNORECASE,
)
_GENERAL_WITHDRAWAL_RE = re.compile(
    r"^\s*(?:"
    r"no\s+more|"
    r"(?:stop|end)\s+this|"
    r"(?:let(?:'s|\s+us)|can\s+we|i\s+(?:want|need)\s+to)\s+"
    r"stop(?:\s+(?:this|now))?|"
    r"let(?:'s|\s+us)\s+not\s+do\s+this|"
    r"i\s+(?:don't|dont|do\s+not)\s+want\s+(?:this|that)(?:\s+anymore)?|"
    r"i\s+(?:don't|dont|do\s+not)\s+want\s+to\s+continue|"
    r"i(?:'m|\s+am)\s+done\s+with\s+this"
    r")[.!?\s]*$",
    re.IGNORECASE,
)
_CONTINUATION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"(?:don't|dont|do\s+not|never)(?:\s+ever)?(?:\s+you)?\s+stop|"
    r"i\s+(?:don't|dont|do\s+not)\s+want\s+you\s+to\s+stop"
    r").*$",
    re.IGNORECASE,
)


def _negates_stop(before: str) -> bool:
    """Whether the words immediately before ``stop`` ask it to continue."""
    value = _text_for_checks(before)[-80:]
    if re.search(r"\bwhy\s+(?:don't|dont|do\s+not)\s+you\s+$", value, re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"(?:\b(?:don't|dont|do\s+not|never)(?:\s+ever)?(?:\s+you)?|"
            r"\bi\s+(?:don't|dont|do\s+not)\s+want\s+you\s+to)\s+$",
            value,
            re.IGNORECASE,
        )
    )


def _act_from_tail(tail: str) -> str | None:
    """Return a controlled act only when its following context is plausible.

    The object check prevents ordinary phrases such as ``hit send``, ``oral
    history`` and ``penetration testing`` from becoming sexual boundaries.
    """
    value = tail.lstrip()
    for act_pattern, canonical in _WITHDRAWAL_ACT_ALIASES:
        match = act_pattern.match(value)
        if not match:
            continue
        remainder = value[match.end():]
        clause = re.split(
            r"[,.;!?/\n]|\b(?:but|though|however|and)\b",
            remainder,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip().lower()
        if not clause:
            return canonical
        if clause.startswith(("with me", "around", "-point")):
            return None
        if re.match(
            r"^(?:me|you|him|her|us|them|my|your|his|her|our|their)\b",
            clause,
            re.IGNORECASE,
        ):
            return canonical
        return None
    return None


def _append_act(found: list[str], tail: str) -> None:
    canonical = _act_from_tail(tail)
    if canonical and canonical not in found:
        found.append(canonical)


def withdrawn_acts(message: str) -> tuple[str, ...]:
    """Return concrete acts the user explicitly withdrew consent for.

    ``don't stop`` and ``never stop`` are continuation requests, so a later
    ``stop choking`` substring inside them must not be mistaken for withdrawal.
    """
    if not isinstance(message, str) or not message.strip():
        return ()

    message = _text_for_checks(message)

    found: list[str] = []

    for prefix in _DIRECT_WITHDRAWAL_RE.finditer(message):
        _append_act(found, message[prefix.end():])

    for prefix in _DONT_WANT_WITHDRAWAL_RE.finditer(message):
        _append_act(found, message[prefix.end():])

    for prefix in _DONT_LIKE_WITHDRAWAL_RE.finditer(message):
        _append_act(found, message[prefix.end():])

    for prefix in _NOT_INTO_WITHDRAWAL_RE.finditer(message):
        _append_act(found, message[prefix.end():])

    for prefix in _NO_WITHDRAWAL_RE.finditer(message):
        _append_act(found, message[prefix.end():])

    for prefix in _STOP_WITHDRAWAL_RE.finditer(message):
        # A continuation such as "don't stop choking me" contains the text
        # "stop choking me", but its stop is explicitly negated.
        before = message[max(0, prefix.start() - 80):prefix.start()]
        if _negates_stop(before):
            continue
        _append_act(found, message[prefix.end():])

    for act_pattern, canonical in _WITHDRAWAL_ACT_ALIASES:
        match = act_pattern.search(message)
        if not match:
            continue
        suffix = message[match.end():]
        if re.match(
            r"\s+is\s+(?:a\s+)?(?:off[ -]?limits?|a\s+hard\s+no|my\s+limit)\b",
            suffix,
            re.IGNORECASE,
        ) and canonical not in found:
            found.append(canonical)

    return tuple(found)


def is_consent_withdrawal(message: str) -> bool:
    """Whether the current turn clearly withdraws/pauses sexual consent."""
    if not isinstance(message, str):
        return False
    checked = _text_for_checks(message)
    return bool(withdrawn_acts(checked)) or bool(
        _STOP_ONLY_RE.fullmatch(checked) or _GENERAL_WITHDRAWAL_RE.fullmatch(checked)
    )


def classify_fast(message: str) -> str | None:
    """Return ``"nsfw"`` for a sexual register, otherwise ``None``."""
    if not isinstance(message, str) or not message:
        return None
    if is_consent_withdrawal(message):
        return None
    check_text = _text_for_checks(message)
    if _CONTINUATION_RE.fullmatch(check_text):
        return "nsfw"
    if NSFW_PATTERN.search(check_text):
        return "nsfw"
    return None
