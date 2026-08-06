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
    "see you now",
    "искам да те видя",
    "може ли да те видя",
    "покажи ми се",
    "искам да те гледам",
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
    "want",
    "wanna",
    "give me",
    "прати",
    "пратиш",
    "изпрати",
    "изпратиш",
    "покажи",
    "искам",
    "може ли",
    "може ли да видя",
    "дай ми",
)

_POLITE_REQUEST_WORDS = frozenset({"please", "pls", "моля"})

_MEDIA_TYPE_PLURAL_ALIASES: Mapping[str, tuple[str, ...]] = {
    "photo": ("photos", "pictures", "pics", "selfies"),
    "video": ("videos", "clips", "vids"),
}

_INVENTORY_REQUEST_PREFIX_RE = re.compile(
    r"^(?:do\s+you\s+(?:have|sell|offer)|have\s+you\s+got|"
    r"you\s+got|got)\s+",
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
    "не сега",
    "не ми пращай",
    "не ми продавай контент",
    "не ми продавай съдържание",
    "не искам",
    "няма нужда",
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
    "не ми пращай",
    "не ми продавай контент",
    "не ми продавай съдържание",
)

_HARD_DECLINES = (
    "never ask me again",
    "don't ever ask",
    "do not ever ask",
    "never send me",
    "stop asking forever",
    "никога повече не ме питай",
    "не ме питай повече никога",
    "никога не ми пращай",
    "спри завинаги",
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
        "да",
        "да моля",
        "искам",
        "може",
        "добре",
        "ок",
    }
)


def _normalize(text: str) -> str:
    return " ".join(str(text or "").casefold().replace("’", "'").split())


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)"
    return re.search(pattern, text, flags=re.UNICODE) is not None


def _media_type_aliases(media_type: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *MEDIA_TYPE_ALIASES[media_type],
                *_MEDIA_TYPE_PLURAL_ALIASES.get(media_type, ()),
            )
        )
    )


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


def _structured_short_request(
    text: str,
    *,
    requested_type: str | None,
    parsed_tags: Mapping[str, tuple[str, ...]],
    explicitness: str | None,
) -> bool:
    """Recognize terse noun requests without turning ordinary mentions into sales."""
    if requested_type is None:
        return False
    residual = text
    phrases: list[str] = list(_media_type_aliases(requested_type))
    for group, values in parsed_tags.items():
        for value in values:
            phrases.extend(TAG_ALIASES[group][value])
    if explicitness:
        phrases.extend(EXPLICITNESS_ALIASES[explicitness])
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
        "една",
        "едно",
        "още",
        "твоя",
        "твое",
        "от",
        "в",
        "на",
        "моля",
        "зад",
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

    @property
    def request_type(self) -> str | None:
        if not self.requested:
            return None
        return self.requested_type or "generic"


def classify_media_intent(text: str) -> MediaIntent:
    """Normalize one processed user batch into controlled media facets."""
    normalized = _normalize(text)

    decline_kind = None
    if any(_contains_phrase(normalized, phrase) for phrase in _HARD_DECLINES):
        decline_kind = "hard"
    elif any(_contains_phrase(normalized, phrase) for phrase in _SOFT_DECLINES):
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
            if any(_contains_phrase(normalized, alias) for alias in aliases):
                matches.append(canonical)
        if matches:
            parsed_tags[group] = tuple(matches)

    explicitness = None
    for level in reversed(EXPLICITNESS_LEVELS):
        aliases = EXPLICITNESS_ALIASES[level]
        if any(_contains_phrase(normalized, alias) for alias in aliases):
            explicitness = level
            break

    has_cue = any(_contains_phrase(normalized, cue) for cue in _REQUEST_CUES)
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
    requested = generic or (
        has_cue and bool(requested_type or parsed_tags or explicitness)
    ) or short_media_phrase or inventory_request

    affirmative_text = normalized.strip(" .,!?")
    affirmative = affirmative_text in _AFFIRMATIVES
    if not affirmative:
        affirmative = any(
            affirmative_text.startswith(prefix)
            for prefix in ("yes ", "yeah ", "sure ", "да ", "искам ")
        )

    decline_global = (
        decline_kind == "hard"
        or any(_contains_phrase(normalized, phrase) for phrase in _GLOBAL_SOFT_DECLINES)
        or bool(decline_kind and (requested_type or parsed_tags or explicitness))
    )
    bare_negative = normalized.strip(" .,!?") in {
        "no",
        "nope",
        "nah",
        "не",
        "не благодаря",
    }
    if bare_negative and decline_kind is None:
        decline_kind = "soft"

    return MediaIntent(
        requested=requested,
        requested_type=requested_type,
        tags=parsed_tags,
        explicitness=explicitness,
        decline_kind=decline_kind,
        decline_global=decline_global,
        affirmative=affirmative,
    )


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
                if trigger in {"direct", "confirmed_direct", "permission_reask"}:
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
