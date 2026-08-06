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
from bot.time_context import get_media_fallback_reason, get_media_locations


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
        if is_generic or (has_request_words and _contains_visual_term(quoted_text)):
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
    }
    return remaining.issubset(allowed)


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

    generic_phrase = _generic_request_phrase(normalized)
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
    direct_candidate = generic or (
        has_cue and bool(requested_type or parsed_tags or explicitness)
    ) or (has_cue and visual_content) or short_media_phrase or inventory_request
    direct_candidate = direct_candidate or price_request

    contextual_form = _is_contextual_request_form(
        normalized,
        direct_candidate=direct_candidate,
    )
    requested = recent_media_context if contextual_form else direct_candidate

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


def classify_media_intent_batch(
    messages: Sequence[str],
    *,
    recent_media_context: bool = False,
) -> MediaIntent:
    """Classify an ordered raw-message batch deterministically.

    Each raw message is classified independently. The last decisive message
    (a request or decline) wins, so a later correction supersedes an earlier
    command without conflating hundreds of debounced messages into one string.
    Contextual ellipsis is enabled only by the explicit boolean supplied by the
    caller; activity earlier in this same batch does not manufacture context.
    """

    if isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of raw message strings")

    last_intent = MediaIntent()
    last_decisive: MediaIntent | None = None
    for message in messages:
        intent = _classify_media_intent_text(
            message,
            recent_media_context=recent_media_context,
        )
        last_intent = intent
        if (
            intent.requested
            or intent.decline_kind is not None
            or intent.blocked_request
        ):
            last_decisive = intent
    return last_decisive or last_intent


@dataclass(frozen=True, slots=True)
class PlannedItem:
    item: MediaItem
    action: str
    description: str
    fallback_reason: str | None
    request_type: str
    trigger: str


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
        current_locations: Sequence[str],
        affinity: Mapping[tuple[str, str], float],
    ) -> float:
        score = 0.0
        if set(item.tags.get("location", ())).intersection(current_locations):
            score += 5.0
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

        current_locations = tuple(get_media_locations(period))
        active = [
            item
            for item in self.catalog.active_items()
            if item.id not in unlocked and item.id not in reserved and heat_allows(item, heat)
        ]
        typed = [item for item in active if item.media_type == target_type]
        used_alternative_type = False
        if not typed:
            typed = [item for item in active if item.media_type != target_type]
            used_alternative_type = bool(typed)
        if not typed:
            return None

        exact = [item for item in typed if self._tag_match(item, intent)]

        def is_current_location(item: MediaItem) -> bool:
            return bool(set(item.tags.get("location", ())).intersection(current_locations))

        exact_new = [item for item in exact if item.id not in offered_ids]
        current_new = [item for item in exact_new if is_current_location(item)]
        other_exact_new = [item for item in exact_new if not is_current_location(item)]
        # If the requested location has no exact item, preserve the substantive
        # visual request (body/activity/outfit/etc.) before falling back to a
        # current-location item with the wrong content. This is the catalog
        # priority "same type/body/activity, different location".
        semantic_tags = {
            group: values
            for group, values in intent.tags.items()
            if group != "location"
        }
        has_semantic_request = bool(semantic_tags or intent.explicitness)
        semantic_intent = MediaIntent(
            requested=True,
            requested_type=intent.requested_type,
            tags=semantic_tags,
            explicitness=intent.explicitness,
        )
        semantic_new = [
            item
            for item in typed
            if has_semantic_request
            and item.id not in offered_ids
            and item not in exact_new
            and self._tag_match(item, semantic_intent)
        ]
        other_similar_new = [
            item
            for item in typed
            if item.id not in offered_ids
            and item not in exact_new
            and item not in semantic_new
        ]

        def ranked(items: Sequence[MediaItem]) -> list[MediaItem]:
            return sorted(
                items,
                key=lambda item: (
                    -self._similarity(item, intent, current_locations, affinity),
                    item.id,
                ),
            )

        selected = None
        if current_new:
            selected = ranked(current_new)[0]
        elif other_exact_new:
            selected = ranked(other_exact_new)[0]
        elif semantic_new:
            selected = ranked(semantic_new)[0]
        elif other_similar_new:
            selected = ranked(other_similar_new)[0]
        else:
            # All suitable new candidates are exhausted. Prefer an old exact
            # match, otherwise the closest old item of the requested type.
            def repeat_allowed(item: MediaItem) -> bool:
                if trigger in {"direct", "permission_reask"}:
                    return True
                return (
                    batch_number - last_by_content.get(item.id, -10**9)
                    >= config.MEDIA_REPEAT_COOLDOWN_BATCHES
                )

            repeats = [item for item in exact if repeat_allowed(item)]
            if not repeats:
                repeats = [item for item in typed if repeat_allowed(item)]
            if repeats:
                # Oldest delivered candidate first; similarity breaks ties.
                selected = sorted(
                    repeats,
                    key=lambda item: (
                        last_by_content.get(item.id, -10**9),
                        -self._similarity(item, intent, current_locations, affinity),
                        item.id,
                    ),
                )[0]
        if selected is None:
            return None

        description, presentation_is_current = selected.description_for_period(period)
        current_match = is_current_location(selected) and presentation_is_current
        has_requested_facets = bool(intent.tags or intent.explicitness)
        requested_mismatch = has_requested_facets and selected not in exact
        fallback = used_alternative_type or requested_mismatch or not current_match
        action = "offer_fallback" if fallback else "offer_current"
        if not fallback:
            reason = None
        else:
            reasons: list[str] = []
            if not current_match:
                reasons.append(get_media_fallback_reason(period))
            if used_alternative_type:
                reasons.append(
                    f"she does not have a {target_type} that fits, so this "
                    f"{selected.media_type} is the closest alternative"
                )
            elif requested_mismatch:
                reasons.append(
                    "she does not have the exact requested variation right now"
                )
            reason = "; ".join(reasons)
        return PlannedItem(
            item=selected,
            action=action,
            description=description,
            fallback_reason=reason,
            request_type=request_type,
            trigger=trigger,
        )


def last_relevant_locked_offer(
    history: Sequence[OfferRecord], unlocked: set[str], *, batch_number: int
) -> OfferRecord | None:
    """Return a recent visible locked offer that a refusal can refer to."""
    for offer in history:
        if offer.content_id not in unlocked and batch_number - offer.batch_number <= 8:
            return offer
    return None
