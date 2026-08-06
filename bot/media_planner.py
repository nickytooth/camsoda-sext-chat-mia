"""Deterministic media-intent classification and catalog rotation.

The LLM never sees or chooses catalog entries.  This planner normalizes the
user's words into the catalog's controlled vocabulary, excludes entitlements,
and ranks exactly one real item using delivered-offer history.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from bot import config
from bot.media_catalog import (
    EXPLICITNESS_ALIASES,
    EXPLICITNESS_LEVELS,
    MEDIA_TYPE_ALIASES,
    TAG_ALIASES,
    MediaCatalog,
    MediaItem,
    heat_allows,
)
from bot.media_repository import MediaRepository, OfferRecord
from bot.time_context import get_media_live_capture_blocker, get_media_locations


_GENERIC_REQUESTS = (
    "show me",
    "let me see you",
    "want to see you",
    "i want to see you",
    "can i see you",
    "could i see you",
    "may i see you",
    "see you now",
    "show me what you're wearing",
    "show me what you are wearing",
    "can i see what you're wearing",
    "can i see what you are wearing",
    "i want to see what you're wearing",
    "i want to see what you are wearing",
    "i wish i could see you right now",
    "wish i could see you right now",
)

_REQUEST_CUES = (
    "send",
    "show",
    "let me see",
    "can i see",
    "could i see",
    "can i have",
    "could i have",
    "may i have",
    "can i get",
    "could i get",
    "may i get",
    "i want",
    "i wanna",
    "i need",
    "give me",
    "drop me",
    "share me",
    "share with me",
    "share a",
    "share some",
)

_CONTEXTUAL_REQUESTS = frozenset(
    {
        "another one",
        "one more",
        "more",
        "show me another",
        "show me more",
        "send another",
        "send me another",
        "send more",
        "send me more",
        "something else",
    }
)

_CONTEXTUAL_MARKERS = frozenset(
    {
        "another",
        "more",
        "instead",
        "else",
    }
)

_POLITE_REQUEST_WORDS = frozenset({"please", "pls"})

_MEDIA_TYPE_PLURAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "photo": ("photos", "pictures", "pics", "selfies"),
    "video": ("videos", "clips", "vids"),
}

_VISUAL_CONTENT_TERMS = ("content",)
_UNSUPPORTED_SCRIPT_RE = re.compile(
    r"[\u0400-\u052f\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff]"
)

_INVENTORY_REQUEST_PREFIX_RE = re.compile(
    r"^(?:do\s+you\s+(?:have|sell|offer)|have\s+you\s+got|"
    r"you\s+got|got)\s+",
    re.IGNORECASE,
)

_PRICE_REQUEST_RE = re.compile(
    r"^(?:how\s+much(?:\s+(?:is|are|for|do\s+you\s+charge\s+for))?|"
    r"what(?:'s|\s+is)\s+the\s+price(?:\s+of|\s+for)?|"
    r"what\s+do\s+you\s+charge\s+for)\b",
    re.IGNORECASE,
)

_CONTEXTUAL_REFINEMENT_RE = re.compile(
    r"^(?:(?:what|how)\s+about|and|maybe)\s+(?:a\s+|an\s+|some\s+)?"
    r"(?:photo|photos|picture|pictures|pic|pics|selfie|selfies|"
    r"video|videos|clip|clips|vid|vids|nude|nudes)(?:\s+instead)?$|"
    r"^(?:a\s+|an\s+)?(?:photo|picture|pic|selfie|video|clip|vid)"
    r"\s+instead$",
    re.IGNORECASE,
)

_CURRENT_REFINEMENT_RE = re.compile(
    r"^(?:(?:make|keep)\s+it\s+)?(?:live|fresh)(?:\s+(?:one|instead))?$|"
    r"^right\s+now$",
    re.IGNORECASE,
)

_SUBJECTLESS_VISUAL_DESIRE_RE = re.compile(
    r"^(?:please\s+)?(?:wanna|want\s+to)\s+see\s+"
    r"(?P<target>.+)$",
    re.IGNORECASE,
)

_SUBJECTLESS_VISUAL_DESIRE_SUFFIX_RE = re.compile(
    r"\s+(?P<modifier>right\s+now|so\s+bad|rn|again|babe|baby|mia)$",
    re.IGNORECASE,
)

_CURRENT_MEDIA_NOUN = r"(?:photo|picture|pic|selfie|video|clip|vid)"
_CURRENT_MEDIA_QUALIFIER = (
    r"(?:nude|naked|explicit|sexy|private|teasing|hot|topless|pussy|ass|"
    r"butt|booty|boobs?|tits?|breasts?|full[- ]body|lingerie|bikini|"
    r"quick|little|new|fresh|live)"
)

_CAPTURE_CURRENT_RE = re.compile(
    rf"^(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+)?"
    rf"(?P<verb>take|snap|film)\s+(?:me\s+)?"
    rf"(?:(?:a|an)\s+(?:{_CURRENT_MEDIA_QUALIFIER}\s+){{0,3}}"
    rf"{_CURRENT_MEDIA_NOUN}|one)"
    # Accept a scoped target here, then let the external-target trust boundary
    # decide whether it is Mia (``your pussy``) or an unrelated object/person
    # (``the menu``, ``the band``, ``Tyler``). This makes a later disallowed
    # capture command decisive inside a multi-message batch.
    r"(?:\s+(?:of|from)\s+.+?)?"
    r"(?:\s+for\s+me)?\s+(?:right\s+)?now$",
    re.IGNORECASE,
)

_SHOW_CURRENT_RE = re.compile(
    r"^(?:(?:(?:can|could|would|will)\s+you\s+)?show\s+me|"
    r"(?:can|could|may)\s+i\s+see)\s+what\s+you"
    r"(?:\s+look\s+like|(?:'re|\s+are)\s+wearing)"
    r"\s+right\s+now$",
    re.IGNORECASE,
)

_SHOW_BODY_CURRENT_RE = re.compile(
    rf"^(?:(?:(?:can|could|would|will)\s+you\s+)?show\s+me|"
    rf"(?:can|could|may)\s+i\s+see)\s+(?:(?:a|an)\s+"
    rf"(?:{_CURRENT_MEDIA_QUALIFIER}\s+){{0,3}}{_CURRENT_MEDIA_NOUN}"
    rf"\s+(?:of|from)\s+)?"
    rf"(?:your\s+.+|you(?:\s+(?:naked|nude))?)\s+right\s+now$",
    re.IGNORECASE,
)

_SOFT_DECLINES = (
    "not now",
    "no thanks",
    "don't send",
    "do not send",
    "stop sending",
    "don't sell me content",
    "dont sell me content",
    "do not sell me content",
    "don't want",
    "dont want",
    "do not want",
    "don't need",
    "dont need",
    "do not need",
)

_GLOBAL_SOFT_DECLINES = (
    "don't send",
    "do not send",
    "stop sending",
    "don't sell me content",
    "dont sell me content",
    "do not sell me content",
)

_HARD_DECLINES = (
    "never ask me again",
    "don't ever ask",
    "do not ever ask",
    "never send me",
    "stop asking forever",
)

_NEGATIVE_REQUEST_RE = re.compile(
    r"(?:\b(?:don't|dont|do\s+not|can't|cannot|won't|wouldn't)\b.*"
    r"\b(?:send|show|give|share|want|need|photo|picture|pic|video|clip|nude)s?\b|"
    r"\b(?:no|without)\s+(?:more\s+)?"
    r"(?:photos?|pictures?|pics?|videos?|clips?|nudes?)\b)",
    re.IGNORECASE,
)

_TRAILING_NEGATIVE_RE = re.compile(
    r"(?:^|[,;.!?]\s*|\b(?:but|actually|wait|sorry)\s+)"
    r"(?:no|nope|nah|never\s+mind|forget\s+it|"
    r"cancel(?:\s+(?:it|that))?|i\s+changed\s+my\s+mind)\s*[.!?]*$",
    re.IGNORECASE,
)

_REPORTED_OR_PAST_RE = re.compile(
    r"^(?:(?:yesterday|earlier|previously|before|last\s+"
    r"(?:night|week|time))\s+)?"
    r"(?:i|we|you|he|she|they|mia|tyler|my\s+friend|someone)\s+"
    r"(?:said|says|told|asked|wanted|needed|wrote|texted|mentioned|"
    r"watched|saw|sent|showed|shared|had|bought|opened|unlocked|"
    r"received|liked|were\s+going|was\s+going)\b|"
    r"^(?:remember\s+when|the\s+(?:message|prompt|story|novel|character)\s+"
    r"(?:said|says))\b",
    re.IGNORECASE,
)

_REPORTING_BEFORE_COMMAND_RE = re.compile(
    r"\b(?:said|says|told|asked|wrote|texted|quoted)\b[^.!?\n]{0,80}\b"
    r"(?:send|show|give|share)\b",
    re.IGNORECASE,
)

_QUOTED_SEGMENT_RE = re.compile(r'["“”„«»`]([^"“”„«»`]+)["“”„«»`]')
_SINGLE_QUOTED_SEGMENT_RE = re.compile(r"(?<!\w)'([^']+)'(?!\w)")

_EXTERNAL_MEDIA_TARGET_RE = re.compile(
    r"\b(?:photo|picture|pic|selfie|video|clip|vid)s?\s+of\s+(.+)$",
    re.IGNORECASE,
)

_AFFIRMATIVES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "sure",
        "okay",
        "ok",
        "please",
        "yes please",
        "i do",
        "i want to",
        "now",
        "right now",
        "do it",
        "send it",
        "send it now",
        "show me",
        "let me see",
        "give it to me",
    }
)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().replace("’", "'").split())


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)"
    return re.search(pattern, text, flags=re.UNICODE) is not None


def _english_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    """Keep the catalog vocabulary while excluding unsupported scripts."""
    return tuple(alias for alias in aliases if alias.isascii())


def _media_type_aliases(media_type: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *_english_aliases(MEDIA_TYPE_ALIASES[media_type]),
                *_MEDIA_TYPE_PLURAL_ALIASES.get(media_type, ()),
            )
        )
    )


def _explicitness_aliases(level: str) -> tuple[str, ...]:
    return _english_aliases(EXPLICITNESS_ALIASES[level])


def _generic_request_phrase(text: str) -> str:
    """Ignore surrounding punctuation, emoji and polite filler.

    Generic requests deliberately remain an exact phrase match so an open-ended
    sentence such as ``show me what you want`` is not mistaken for a media
    request.  Tokenising first merely makes the exact match resilient to chat
    punctuation and emoji, which are not part of the user's intent.
    """

    words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", text, flags=re.UNICODE)
    while words and words[0] in _POLITE_REQUEST_WORDS:
        words.pop(0)
    while words and words[-1] in _POLITE_REQUEST_WORDS:
        words.pop()
    return " ".join(words)


def _contains_visual_term(text: str) -> bool:
    if any(
        _contains_phrase(text, alias)
        for media_type in MEDIA_TYPE_ALIASES
        for alias in _media_type_aliases(media_type)
    ):
        return True
    if any(
        _contains_phrase(text, alias)
        for level in EXPLICITNESS_LEVELS
        for alias in _explicitness_aliases(level)
    ):
        return True
    return any(
        _contains_phrase(text, alias)
        for group in ("body_focus", "outfit", "activity")
        for aliases in TAG_ALIASES[group].values()
        for alias in _english_aliases(aliases)
    )


def _looks_reported_quoted_or_past(text: str) -> bool:
    """Fail closed when request-shaped words are quoted or merely narrated."""
    if _REPORTED_OR_PAST_RE.search(text) or _REPORTING_BEFORE_COMMAND_RE.search(text):
        return True
    stripped = text.strip()
    if re.match(r"^>+\s*", stripped):
        return True
    if (
        len(stripped) >= 2
        and stripped[0] in "[("
        and stripped[-1] in ")]"
    ):
        return True
    quoted_segments = (
        _QUOTED_SEGMENT_RE.findall(text) + _SINGLE_QUOTED_SEGMENT_RE.findall(text)
    )
    for quoted in quoted_segments:
        quoted_text = _normalize(quoted)
        has_request_words = any(
            _contains_phrase(quoted_text, cue) for cue in _REQUEST_CUES
        )
        is_generic = _generic_request_phrase(quoted_text) in _GENERIC_REQUESTS
        is_subjectless_visual_desire = bool(
            _SUBJECTLESS_VISUAL_DESIRE_RE.fullmatch(
                _generic_request_phrase(quoted_text)
            )
            and _contains_visual_term(quoted_text)
        )
        if (
            is_generic
            or is_subjectless_visual_desire
            or (has_request_words and _contains_visual_term(quoted_text))
        ):
            return True
    return False


def _has_external_media_target(
    text: str,
    *,
    requested_type: str | None,
) -> bool:
    """Reject external-object photos/videos while allowing Mia-focused targets."""
    if requested_type not in {"photo", "video"}:
        return False
    match = _EXTERNAL_MEDIA_TARGET_RE.search(text)
    if match is None:
        return False
    target = _normalize(next(value for value in match.groups() if value is not None))
    target = target.strip(" .,!?\"“”„«»")
    if re.match(r"^(?:you|yourself)\b", target, re.IGNORECASE):
        return False
    if re.match(
        r"^(?:what\s+you(?:'re|\s+are)\s+wearing|your\s+"
        r"(?:outfit|clothes|look))\b",
        target,
        re.IGNORECASE,
    ):
        return False
    return not _contains_visual_term(target)


def _is_contextual_request_form(text: str, *, direct_candidate: bool) -> bool:
    phrase = _generic_request_phrase(text)
    if phrase in _CONTEXTUAL_REQUESTS:
        return True
    if _CONTEXTUAL_REFINEMENT_RE.fullmatch(phrase):
        return True
    words = set(re.findall(r"[^\W_]+", phrase, flags=re.UNICODE))
    return bool(direct_candidate and words.intersection(_CONTEXTUAL_MARKERS))


def _structured_short_request(
    text: str,
    *,
    requested_type: str | None,
    parsed_tags: Mapping[str, tuple[str, ...]],
    explicitness: str | None,
) -> bool:
    """Recognize terse noun requests without turning ordinary mentions into sales."""
    if requested_type is None and explicitness not in {"nude", "explicit"}:
        return False
    residual = text
    phrases: list[str] = (
        list(_media_type_aliases(requested_type)) if requested_type else []
    )
    for group, values in parsed_tags.items():
        for value in values:
            phrases.extend(_english_aliases(TAG_ALIASES[group][value]))
    if explicitness:
        phrases.extend(_explicitness_aliases(explicitness))
    for phrase in sorted(set(phrases), key=len, reverse=True):
        residual = re.sub(
            r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)",
            " ",
            residual,
            flags=re.UNICODE,
        )
    remaining = set(re.findall(r"\w+", residual, flags=re.UNICODE))
    allowed = {
        "a",
        "an",
        "another",
        "any",
        "one",
        "more",
        "some",
        "please",
        "your",
        "of",
        "from",
        "at",
        "behind",
        "in",
        "on",
        "with",
        "and",
        "the",
        "me",
        "her",
        "fresh",
        "live",
    }
    return remaining.issubset(allowed)


def _structured_facet_refinement(
    text: str,
    *,
    parsed_tags: Mapping[str, tuple[str, ...]],
    explicitness: str | None,
) -> bool:
    """Recognize a terse facet that can refine an authorized media request.

    A facet is intentionally not a request by itself.  This grammar merely
    identifies safe ellipsis such as ``from your pussy`` or ``at home`` so the
    ordered-batch classifier can attach it to an active request (or bounded
    recent media context) without promoting ordinary body/location mentions.
    """

    if not parsed_tags and explicitness is None:
        return False
    residual = text
    phrases: list[str] = []
    for group, values in parsed_tags.items():
        for value in values:
            phrases.extend(_english_aliases(TAG_ALIASES[group][value]))
    if explicitness:
        phrases.extend(_explicitness_aliases(explicitness))
    for phrase in sorted(set(phrases), key=len, reverse=True):
        residual = re.sub(
            r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)",
            " ",
            residual,
            flags=re.UNICODE,
        )
    remaining = set(re.findall(r"[^\W_]+", residual, flags=re.UNICODE))
    allowed = {
        "a",
        "an",
        "and",
        "at",
        "behind",
        "from",
        "in",
        "instead",
        "maybe",
        "of",
        "on",
        "please",
        "right",
        "some",
        "the",
        "with",
        "your",
        "now",
    }
    return remaining.issubset(allowed)


def _subjectless_visual_desire(
    text: str,
    *,
    requested_type: str | None,
    parsed_tags: Mapping[str, tuple[str, ...]],
    explicitness: str | None,
) -> tuple[bool, bool]:
    """Recognize conversational ``wanna see your ...`` requests safely.

    The omitted subject is normal chat English, but a bare ``wanna`` cue is
    too broad: it would also promote ``do you wanna see my ass?`` and reported
    speech.  This anchored grammar accepts only a Mia-owned visual target, a
    controlled media noun, or ``you`` by itself. A small allowlist of trailing
    chat modifiers is peeled independently of order; only ``right now``/``rn``
    changes the request into a current-capture request.
    """

    match = _SUBJECTLESS_VISUAL_DESIRE_RE.fullmatch(text)
    if match is None:
        return False, False

    target = _normalize(match.group("target")).strip(" .,!?")
    modifiers: set[str] = set()
    while suffix_match := _SUBJECTLESS_VISUAL_DESIRE_SUFFIX_RE.search(target):
        modifiers.add(_normalize(suffix_match.group("modifier")))
        target = target[: suffix_match.start()].rstrip()
    requires_current = bool(modifiers.intersection({"right now", "rn"}))
    if target in {"you", "you naked", "you nude"}:
        return True, requires_current

    visual_tags = {
        group: values
        for group, values in parsed_tags.items()
        if group in {"body_focus", "outfit", "activity"}
    }
    if target.startswith("your "):
        if not visual_tags and explicitness is None and requested_type is None:
            return False, False
        if requested_type is not None or explicitness in {"nude", "explicit"}:
            requested = _structured_short_request(
                target,
                requested_type=requested_type,
                parsed_tags=visual_tags,
                explicitness=explicitness,
            )
        else:
            requested = _structured_facet_refinement(
                target,
                parsed_tags=visual_tags,
                explicitness=explicitness,
            )
        return requested, requires_current if requested else False

    if requested_type is None and explicitness not in {"nude", "explicit"}:
        return False, False
    requested = _structured_short_request(
        target,
        requested_type=requested_type,
        parsed_tags=visual_tags,
        explicitness=explicitness,
    )
    return requested, requires_current if requested else False


def _structured_inventory_request(
    text: str,
    *,
    requested_type: str | None,
    parsed_tags: Mapping[str, tuple[str, ...]],
    explicitness: str | None,
) -> bool:
    """Recognize direct inventory questions without matching media stories.

    The inventory grammar is deliberately anchored at the beginning. Ordinary
    statements such as ``I watched a video`` may normalize the media type, but
    they never become a commerce request.
    """

    remainder, replacements = _INVENTORY_REQUEST_PREFIX_RE.subn("", text, count=1)
    if replacements != 1:
        return False
    return _structured_short_request(
        remainder,
        requested_type=requested_type,
        parsed_tags=parsed_tags,
        explicitness=explicitness,
    )


@dataclass(frozen=True, slots=True)
class MediaIntent:
    requested: bool = False
    requested_type: str | None = None
    tags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    explicitness: str | None = None
    requires_current: bool = False
    decline_kind: str | None = None
    decline_global: bool = False
    affirmative: bool = False
    # A request-shaped command that deliberately failed the trust boundary
    # (for example a quoted command or a photo of an external object). It is
    # not a Mia-media request, but it remains decisive inside an ordered batch
    # so an earlier command cannot leak through and attach a card.
    blocked_request: bool = False

    @property
    def request_type(self) -> str | None:
        if not self.requested:
            return None
        return self.requested_type or "generic"


def _classify_media_intent_text(
    text: str,
    *,
    recent_media_context: bool,
) -> MediaIntent:
    """Classify one raw user message without consulting the LLM."""
    normalized = _normalize(text)
    if _UNSUPPORTED_SCRIPT_RE.search(normalized):
        # English-only runtime: unsupported script is a decisive trust-boundary
        # block so an earlier request in the same raw batch cannot leak through.
        return MediaIntent(blocked_request=True)

    decline_kind = None
    if any(_contains_phrase(normalized, phrase) for phrase in _HARD_DECLINES):
        decline_kind = "hard"
    elif any(_contains_phrase(normalized, phrase) for phrase in _SOFT_DECLINES):
        decline_kind = "soft"
    elif _NEGATIVE_REQUEST_RE.search(normalized):
        decline_kind = "soft"

    bare_negative = normalized.strip(" .,!?") in {
        "no",
        "nope",
        "nah",
    }
    if bare_negative and decline_kind is None:
        decline_kind = "soft"

    phrase = _generic_request_phrase(normalized)
    capture_current_match = _CAPTURE_CURRENT_RE.fullmatch(phrase)

    requested_types: list[str] = []
    for media_type in MEDIA_TYPE_ALIASES:
        aliases = _media_type_aliases(media_type)
        if any(_contains_phrase(normalized, alias) for alias in aliases):
            requested_types.append(media_type)
    # If both are named, the last explicitly occurring type wins instead of
    # silently biasing to the dict's first entry.
    requested_type = None
    if requested_types:
        positions = {}
        for media_type in requested_types:
            positions[media_type] = max(
                match.start()
                for alias in _media_type_aliases(media_type)
                for match in re.finditer(
                    r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)",
                    normalized,
                    flags=re.UNICODE,
                )
            )
        requested_type = max(requested_types, key=lambda value: positions[value])

    # Capture verbs carry an unambiguous type even when the user says "one".
    # Explicit photo/video nouns still win if both signals are present.
    if requested_type is None and capture_current_match is not None:
        requested_type = (
            "video" if capture_current_match.group("verb").casefold() == "film" else "photo"
        )

    generic_phrase = phrase
    generic = any(generic_phrase == phrase for phrase in _GENERIC_REQUESTS)

    parsed_tags: dict[str, tuple[str, ...]] = {}
    for group, values in TAG_ALIASES.items():
        matches: list[str] = []
        for canonical, aliases in values.items():
            if any(
                _contains_phrase(normalized, alias)
                for alias in _english_aliases(aliases)
            ):
                matches.append(canonical)
        if matches:
            parsed_tags[group] = tuple(matches)

    explicitness = None
    for level in reversed(EXPLICITNESS_LEVELS):
        aliases = _explicitness_aliases(level)
        if any(_contains_phrase(normalized, alias) for alias in aliases):
            explicitness = level
            break

    has_cue = any(_contains_phrase(normalized, cue) for cue in _REQUEST_CUES)
    visual_content = any(
        _contains_phrase(normalized, term) for term in _VISUAL_CONTENT_TERMS
    ) and not _contains_phrase(normalized, "social media")
    short_media_phrase = _structured_short_request(
        normalized,
        requested_type=requested_type,
        parsed_tags=parsed_tags,
        explicitness=explicitness,
    )
    inventory_request = _structured_inventory_request(
        normalized,
        requested_type=requested_type,
        parsed_tags=parsed_tags,
        explicitness=explicitness,
    )
    price_request = bool(
        _PRICE_REQUEST_RE.search(normalized)
        and (requested_type or parsed_tags or explicitness or visual_content)
    )
    current_refinement = bool(_CURRENT_REFINEMENT_RE.fullmatch(generic_phrase))
    has_fresh_or_live = bool(re.search(r"(?<!\w)(?:fresh|live)(?!\w)", normalized))
    show_current = bool(_SHOW_CURRENT_RE.fullmatch(generic_phrase)) or bool(
        (
            parsed_tags.get("body_focus")
            or parsed_tags.get("outfit")
            or explicitness
        )
        and _SHOW_BODY_CURRENT_RE.fullmatch(generic_phrase)
    )
    facet_refinement = _structured_facet_refinement(
        normalized,
        parsed_tags=parsed_tags,
        explicitness=explicitness,
    )
    subjectless_desire, subjectless_desire_current = _subjectless_visual_desire(
        generic_phrase,
        requested_type=requested_type,
        parsed_tags=parsed_tags,
        explicitness=explicitness,
    )
    facet_right_now = bool(
        facet_refinement and re.search(r"\bright\s+now\s*[.!?]*$", normalized)
    )
    # ``live`` and ``fresh`` modify an already-established media request; they
    # are not media nouns by themselves. Keeping them out of the request gate
    # prevents ordinary phrases such as ``I want to live with you``, ``show me
    # where you live``, and ``I want a fresh start`` from attaching a card.
    current_direct_candidate = bool(capture_current_match or show_current)
    direct_candidate = generic or (
        has_cue and bool(requested_type or parsed_tags or explicitness)
    ) or (has_cue and visual_content) or short_media_phrase or inventory_request
    direct_candidate = direct_candidate or current_direct_candidate
    direct_candidate = direct_candidate or price_request
    direct_candidate = direct_candidate or subjectless_desire

    contextual_form = _is_contextual_request_form(
        normalized,
        direct_candidate=direct_candidate,
    )
    contextual_form = bool(
        contextual_form
        or (not direct_candidate and (facet_refinement or current_refinement))
    )
    requested = recent_media_context if contextual_form else direct_candidate
    requires_current = bool(
        capture_current_match
        or show_current
        or facet_right_now
        or (has_fresh_or_live and (direct_candidate or current_refinement))
        or current_refinement and generic_phrase == "right now"
        or subjectless_desire_current
    )

    # A final refusal wins even when the same message began with an
    # affirmative (for example ``yes, but no``). This must be evaluated
    # independently of ``requested`` so a permission re-ask cannot promote
    # the earlier "yes" into a media offer.
    if _TRAILING_NEGATIVE_RE.search(normalized):
        decline_kind = "soft"
    blocked_request = bool(
        requested
        and (
            _looks_reported_quoted_or_past(normalized)
            or _has_external_media_target(
                normalized,
                requested_type=requested_type,
            )
        )
    )
    if blocked_request:
        requested = False
    if decline_kind is not None:
        requested = False

    affirmative_text = normalized.strip(" .,!?…")
    affirmative = affirmative_text in _AFFIRMATIVES
    if not affirmative:
        affirmative = bool(
            re.match(
                r"^(?:yes|yeah|yep|sure|okay|ok)\b(?:\s|[,;.!?])",
                affirmative_text,
                flags=re.IGNORECASE,
            )
        )
    if decline_kind is not None:
        affirmative = False

    decline_global = (
        decline_kind == "hard"
        or any(_contains_phrase(normalized, phrase) for phrase in _GLOBAL_SOFT_DECLINES)
        or bool(decline_kind and (requested_type or parsed_tags or explicitness))
    )

    return MediaIntent(
        requested=requested,
        requested_type=requested_type,
        tags=parsed_tags,
        explicitness=explicitness,
        requires_current=requires_current,
        decline_kind=decline_kind,
        decline_global=decline_global,
        affirmative=affirmative,
        blocked_request=blocked_request,
    )


def classify_media_intent(text: str) -> MediaIntent:
    """Normalize one user message into controlled media facets.

    This compatibility API deliberately has no implicit conversation context.
    Use :func:`classify_media_intent_batch` when ordered raw messages or recent
    media context are available.
    """

    return _classify_media_intent_text(text, recent_media_context=False)


def _is_batch_refinement(text: str, intent: MediaIntent) -> bool:
    """Whether ``text`` may safely refine an already-authorized request."""

    normalized = _normalize(text)
    phrase = _generic_request_phrase(normalized)
    return bool(
        _is_contextual_request_form(normalized, direct_candidate=False)
        or _CURRENT_REFINEMENT_RE.fullmatch(phrase)
        or _structured_facet_refinement(
            normalized,
            parsed_tags=intent.tags,
            explicitness=intent.explicitness,
        )
    )


def _merge_media_request(base: MediaIntent, refinement: MediaIntent) -> MediaIntent:
    """Merge a terse later facet into an active same-batch request."""

    tags = dict(base.tags)
    for group, values in refinement.tags.items():
        tags[group] = tuple(values)
    return MediaIntent(
        requested=True,
        requested_type=(
            refinement.requested_type
            if refinement.requested_type is not None
            else base.requested_type
        ),
        tags=tags,
        explicitness=refinement.explicitness or base.explicitness,
        requires_current=base.requires_current or refinement.requires_current,
        affirmative=base.affirmative or refinement.affirmative,
    )


def classify_media_intent_batch(
    messages: Sequence[str],
    *,
    recent_media_context: bool = False,
) -> MediaIntent:
    """Classify an ordered raw-message batch deterministically.

    Each raw message is classified independently. A narrow later type, facet,
    or currentness fragment refines an active request, while a later complete
    request replaces it. Refusals and blocked commands remain decisive in
    order, so they cannot leak an earlier card through the batch. Bounded
    recent context may independently authorize the same narrow ellipsis.
    """

    if isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of raw message strings")

    active_request: MediaIntent | None = None
    last_intent = MediaIntent()
    last_decisive: MediaIntent | None = None
    for message in messages:
        intent = _classify_media_intent_text(
            message,
            recent_media_context=recent_media_context,
        )
        last_intent = intent
        if intent.decline_kind is not None or intent.blocked_request:
            active_request = None
            last_decisive = intent
            continue

        refinement = _is_batch_refinement(message, intent)
        if intent.requested:
            if refinement:
                if active_request is not None:
                    active_request = _merge_media_request(active_request, intent)
                elif last_decisive is not None and (
                    last_decisive.decline_kind is not None
                    or last_decisive.blocked_request
                ):
                    # Recent conversation context may authorize ellipsis, but
                    # it cannot use a fragment to undo a refusal/trust-boundary
                    # block earlier in this same ordered batch.
                    continue
                else:
                    active_request = intent
            else:
                active_request = intent
            last_decisive = active_request
        elif active_request is not None and refinement:
            active_request = _merge_media_request(active_request, intent)
            last_decisive = active_request
    return last_decisive or last_intent


@dataclass(frozen=True, slots=True)
class PlannedItem:
    item: MediaItem
    action: str
    description: str
    fallback_reason: str | None
    request_type: str
    trigger: str
    # Presentation state stays structured until the final copy boundary.  The
    # planner decides facts; it must not write a sentence that is shown to the
    # user as if Mia were describing an inventory search.
    fallback_kind: str | None = None
    requested_detail: str | None = None
    requested_media_type: str | None = None
    live_capture_blocker: str | None = None
    live_capture_blocker_kind: str | None = None


def _live_capture_blocker_kind(blocker: str | None) -> str | None:
    """Canonicalise the controlled schedule blocker for copy validation."""

    value = (blocker or "").lower()
    if not value:
        return None
    if "tyler" in value:
        return "tyler"
    if "customer" in value or "coworker" in value:
        return "work_crowd"
    if "shopper" in value or "store staff" in value:
        return "shopping_crowd"
    if "gym" in value:
        return "gym_crowd"
    return "social_crowd"


class CatalogPlanner:
    """Rank one not-unlocked catalog item using durable delivered history."""

    def __init__(self, catalog: MediaCatalog, repository: MediaRepository):
        self.catalog = catalog
        self.repository = repository

    @staticmethod
    def _tag_match(item: MediaItem, intent: MediaIntent) -> bool:
        for group, wanted in intent.tags.items():
            available = set(item.tags.get(group, ()))
            if wanted and not available.intersection(wanted):
                return False
        if intent.explicitness:
            if EXPLICITNESS_LEVELS.index(item.explicitness) < EXPLICITNESS_LEVELS.index(
                intent.explicitness
            ):
                return False
        return True

    @staticmethod
    def _similarity(
        item: MediaItem,
        intent: MediaIntent,
        affinity: Mapping[tuple[str, str], float],
    ) -> float:
        score = 0.0
        weights = {
            "body_focus": 4.0,
            "activity": 3.0,
            "outfit": 2.0,
            "vibe": 1.5,
            "capture": 1.0,
            "location": 1.0,
        }
        for group, wanted in intent.tags.items():
            overlap = set(item.tags.get(group, ())).intersection(wanted)
            score += weights.get(group, 1.0) * len(overlap)
        if intent.explicitness == item.explicitness:
            score += 1.5
        for group, values in item.tags.items():
            score += sum(affinity.get((group, value), 0.0) for value in values)
        return score

    async def choose(
        self,
        user_id: int,
        intent: MediaIntent,
        *,
        batch_number: int,
        heat: str,
        period: str,
        trigger: str,
        state: Mapping[str, object],
    ) -> PlannedItem | None:
        unlocked = await self.repository.unlocked_content_ids(user_id)
        reserved = await self.repository.active_reserved_content_ids(user_id)
        history = await self.repository.delivered_offer_history(user_id)
        affinity = await self.repository.get_tag_affinity(user_id)
        last_by_content: dict[str, int] = {}
        offered_ids: set[str] = set()
        for offer in history:
            offered_ids.add(offer.content_id)
            last_by_content.setdefault(offer.content_id, offer.batch_number)

        request_type = "proactive" if trigger == "proactive" else (intent.request_type or "generic")
        target_type = intent.requested_type
        if target_type is None:
            if trigger == "proactive":
                target_type = "photo"
            else:
                last_generic = state.get("last_generic_media_type")
                target_type = "video" if last_generic == "photo" else "photo"

        active = [
            item
            for item in self.catalog.active_items()
            if item.id not in unlocked and item.id not in reserved and heat_allows(item, heat)
        ]
        if not active:
            return None

        exact_by_id = {
            item.id: self._tag_match(item, intent)
            for item in active
        }

        def is_current_compatible(item: MediaItem) -> bool:
            return item.description_for_period(period)[1]

        current_locations = set(get_media_locations(period))

        def matches_current_location(item: MediaItem) -> bool:
            return bool(
                current_locations.intersection(item.tags.get("location", ()))
            )

        def tier(item: MediaItem) -> int:
            requested_type_match = item.media_type == target_type
            exact_match = exact_by_id[item.id]
            if exact_match and requested_type_match:
                return 0
            if exact_match:
                return 1
            if requested_type_match:
                return 2
            return 3

        tiered: tuple[list[MediaItem], ...] = tuple(
            [item for item in active if tier(item) == index]
            for index in range(4)
        )

        def new_rank_key(item: MediaItem) -> tuple[object, ...]:
            current_key = 0 if is_current_compatible(item) else 1
            location_key = 0 if matches_current_location(item) else 1
            similarity_key = -self._similarity(item, intent, affinity)
            if intent.requires_current:
                return (current_key, similarity_key, location_key, item.id)
            # For ordinary saved-content requests, substantive similarity and
            # affinity lead; present location then breaks otherwise equal
            # saved-item matches, even when both entries are ``past_only``.
            return (similarity_key, location_key, current_key, item.id)

        selected = None
        for candidates in tiered:
            new_candidates = [item for item in candidates if item.id not in offered_ids]
            if new_candidates:
                selected = sorted(new_candidates, key=new_rank_key)[0]
                break

        if selected is None:
            def repeat_allowed(item: MediaItem) -> bool:
                if trigger in {"direct", "permission_reask"}:
                    return True
                return (
                    batch_number - last_by_content.get(item.id, -10**9)
                    >= config.MEDIA_REPEAT_COOLDOWN_BATCHES
                )

            for candidates in tiered:
                repeats = [
                    item
                    for item in candidates
                    if item.id in offered_ids and repeat_allowed(item)
                ]
                if not repeats:
                    continue
                # Repeat rotation is tier-first, then oldest delivered item;
                # currentness/similarity only break equal-age ties.
                selected = sorted(
                    repeats,
                    key=lambda item: (
                        last_by_content.get(item.id, -10**9),
                        *new_rank_key(item),
                    ),
                )[0]
                break
        if selected is None:
            return None

        description, presentation_is_current = selected.description_for_period(period)
        current_match = presentation_is_current
        has_requested_facets = bool(intent.tags or intent.explicitness)
        requested_mismatch = has_requested_facets and not exact_by_id[selected.id]
        used_alternative_type = bool(
            intent.requested_type is not None
            and selected.media_type != intent.requested_type
        )
        fallback = bool(
            used_alternative_type
            or requested_mismatch
            or (intent.requires_current and not current_match)
        )
        if fallback:
            action = "offer_fallback"
        elif current_match:
            action = "offer_current"
        else:
            action = "offer_saved"
        missing: list[str] = []
        if requested_mismatch:
            for group, wanted in intent.tags.items():
                available = set(selected.tags.get(group, ()))
                if not available.intersection(wanted):
                    missing.extend(value.replace("_", " ") for value in wanted)
            if intent.explicitness and EXPLICITNESS_LEVELS.index(
                selected.explicitness
            ) < EXPLICITNESS_LEVELS.index(intent.explicitness):
                missing.append(intent.explicitness)
        requested_detail = ", ".join(dict.fromkeys(missing)) or None

        # Keep one primary, human-relevant reason.  Multiple planner axes may
        # miss at once, but making Mia recite all of them sounds like a search
        # report.  The selected card already communicates its actual type.
        fallback_kind = None
        live_capture_blocker = None
        reason = None
        if intent.requires_current and not current_match:
            live_capture_blocker = get_media_live_capture_blocker(period)
            if live_capture_blocker:
                fallback_kind = "live_blocked"
                reason = (
                    f"{live_capture_blocker}, so she cannot take a fresh one right now"
                )
            else:
                fallback_kind = "live_unavailable"
                reason = "she does not have a fresh one in that exact angle right now"
        elif used_alternative_type:
            fallback_kind = "type_swap"
            reason = (
                f"she does not have that as a {intent.requested_type}, but she does "
                f"have it as a {selected.media_type}"
            )
        elif requested_mismatch:
            fallback_kind = "semantic_near_match"
            detail = requested_detail or "exact"
            reason = f"this is not quite the {detail} shot he asked for"
        return PlannedItem(
            item=selected,
            action=action,
            description=description,
            fallback_reason=reason,
            request_type=request_type,
            trigger=trigger,
            fallback_kind=fallback_kind,
            requested_detail=requested_detail,
            requested_media_type=intent.requested_type,
            live_capture_blocker=live_capture_blocker,
            live_capture_blocker_kind=_live_capture_blocker_kind(
                live_capture_blocker
            ),
        )


def last_relevant_locked_offer(
    history: Sequence[OfferRecord], unlocked: set[str], *, batch_number: int
) -> OfferRecord | None:
    """Return a recent visible locked offer that a refusal can refer to."""
    for offer in history:
        if offer.content_id not in unlocked and batch_number - offer.batch_number <= 8:
            return offer
    return None
