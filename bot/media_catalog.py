"""Validated catalog metadata for paywalled photos and videos.

The catalog is authored configuration, not user input.  It is nevertheless
validated strictly because its descriptions are later included in a trusted
commerce brief and its object keys are handed to the delivery layer.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Awaitable, Callable, Iterable, Mapping

import yaml

from bot.config import MEDIA_CATALOG_FILE


MEDIA_TYPES = frozenset({"photo", "video"})
MEDIA_STATUSES = frozenset({"active", "inactive"})
EXPLICITNESS_LEVELS = ("tease", "suggestive", "nude", "explicit")
HEAT_LEVELS = ("low", "rising", "high")
PRESENTATION_MODES = frozenset({"current_compatible", "past_only"})

TAG_VALUES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "location": frozenset(
            {
                "home",
                "bedroom",
                "bathroom",
                "kitchen",
                "shower",
                "gym",
                "locker_room",
                "bar",
                "stockroom",
                "club",
                "concert",
                "beach",
                "car",
                "hotel",
                "outdoors",
            }
        ),
        "body_focus": frozenset(
            {"face", "boobs", "pussy", "ass", "legs", "feet", "full_body"}
        ),
        "activity": frozenset(
            {
                "selfie",
                "posing",
                "undressing",
                "showering",
                "working_out",
                "bartending",
                "dancing",
                # Explicit activities stay a closed, curated allowlist.
                "masturbating",
                "oral",
                "vaginal",
                "anal",
                "toy_play",
            }
        ),
        "outfit": frozenset(
            {
                "casual",
                "bartender_outfit",
                "gymwear",
                "dress",
                "lingerie",
                "bikini",
                "towel",
                "topless",
                "nude",
            }
        ),
        "vibe": frozenset(
            {"teasing", "playful", "intimate", "dominant", "submissive", "risky"}
        ),
        "capture": frozenset(
            {"selfie", "mirror", "pov", "closeup", "full_body", "tripod"}
        ),
    }
)

# Conversation aliases are deliberately centralized so classification and
# catalog validation cannot drift into two competing vocabularies.
MEDIA_TYPE_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "photo": (
            "photo",
            "picture",
            "pic",
            "selfie",
            "снимка",
            "снимки",
        ),
        "video": ("video", "clip", "vid", "видео", "клип"),
    }
)

TAG_ALIASES: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "location": MappingProxyType(
            {
                "bar": ("bar", "at work", "behind the bar", "на работа", "в бара"),
                "stockroom": ("stockroom", "back room", "склада"),
                "home": ("home", "at your place", "у вас", "вкъщи"),
                "bedroom": ("bedroom", "in bed", "спалня", "леглото"),
                "bathroom": ("bathroom", "restroom", "тоалетната", "банята"),
                "shower": ("shower", "under the water", "под душа"),
                "gym": ("gym", "workout", "фитнес"),
                "locker_room": ("locker room", "съблекалня"),
                "club": ("club", "nightclub", "дискотека", "клуба"),
                "concert": ("concert", "festival", "концерт"),
                "beach": ("beach", "on the sand", "плаж"),
                "car": ("car", "uber", "кола", "такси"),
                "hotel": ("hotel", "hotel room", "хотел"),
                "kitchen": ("kitchen", "кухня"),
                "outdoors": ("outside", "outdoors", "навън"),
            }
        ),
        "body_focus": MappingProxyType(
            {
                "face": ("face", "smile", "лице"),
                "boobs": ("boobs", "tits", "breasts", "chest", "гърди", "цици"),
                "pussy": ("pussy", "between your legs", "путка"),
                "ass": ("ass", "butt", "booty", "дупе", "задник"),
                "legs": ("legs", "thighs", "крака", "бедра"),
                "feet": ("feet", "toes", "стъпала", "крака долу"),
                "full_body": ("full body", "all of you", "цял ръст", "цялата"),
            }
        ),
        "activity": MappingProxyType(
            {
                "selfie": ("selfie", "селфи"),
                "posing": ("posing", "pose", "поза"),
                "undressing": ("undressing", "taking it off", "съблич"),
                "showering": ("showering", "taking a shower", "къпеш", "под душа"),
                "working_out": ("working out", "workout", "тренираш"),
                "bartending": ("bartending", "making drinks", "правиш коктейл"),
                "dancing": ("dancing", "dance", "танцуваш"),
                "masturbating": ("masturbating", "touching yourself", "пипаш се"),
                "oral": ("oral", "blowjob", "свирка"),
                "vaginal": ("vaginal", "having sex", "секс"),
                "anal": ("anal", "анал"),
                "toy_play": ("toy", "vibrator", "играчка", "вибратор"),
            }
        ),
        "outfit": MappingProxyType(
            {
                "casual": ("casual", "everyday clothes", "ежедневни дрехи"),
                "bartender_outfit": ("bartender outfit", "work outfit", "работните дрехи"),
                "gymwear": ("gymwear", "gym outfit", "клин", "спортни дрехи"),
                "dress": ("dress", "рокля"),
                "lingerie": ("lingerie", "underwear", "бельо"),
                "bikini": ("bikini", "бански"),
                "towel": ("towel", "кърпа"),
                "topless": ("topless", "без горнище"),
                "nude": ("nude", "naked", "голa", "гола"),
            }
        ),
        "vibe": MappingProxyType(
            {
                "teasing": ("teasing", "tease", "дразниш"),
                "playful": ("playful", "fun", "игрива"),
                "intimate": ("intimate", "private", "интимна"),
                "dominant": ("dominant", "in control", "доминантна"),
                "submissive": ("submissive", "obedient", "послушна"),
                "risky": ("risky", "public", "рискова", "публично"),
            }
        ),
        "capture": MappingProxyType(
            {
                "selfie": ("selfie", "селфи"),
                "mirror": ("mirror", "огледало"),
                "pov": ("pov", "point of view", "от твоя гледна точка"),
                "closeup": ("closeup", "close up", "отблизо"),
                "full_body": ("full body", "цял ръст"),
                "tripod": ("tripod", "camera", "статив"),
            }
        ),
    }
)

EXPLICITNESS_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "tease": ("tease", "safe", "дразнеща"),
        "suggestive": ("sexy", "hot", "suggestive", "секси"),
        "nude": ("nude", "nudes", "naked", "гола", "голa"),
        "explicit": ("explicit", "hardcore", "graphic", "експлицитно"),
    }
)

KNOWN_PERIODS = frozenset(
    {
        "night_bed",
        "morning_home",
        "midday_gym",
        "prework_home",
        "bar_shift",
        "evening_pregame",
        "club_night",
        "weekend_night_bed",
        "weekend_hungover",
        "weekend_brunch",
        "weekend_shopping",
        "weekend_home_tyler",
        "weekend_getting_ready",
        "weekend_club_night",
    }
)

PERIOD_LOCATIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "night_bed": ("bedroom", "home"),
        "morning_home": ("bedroom", "home"),
        "midday_gym": ("gym", "locker_room"),
        "prework_home": ("home", "bathroom", "kitchen"),
        "bar_shift": ("bar", "stockroom"),
        "evening_pregame": ("home", "bathroom", "bedroom"),
        "club_night": ("club", "bathroom", "car"),
        "weekend_night_bed": ("bedroom", "home"),
        "weekend_hungover": ("home", "bedroom"),
        "weekend_brunch": ("outdoors",),
        "weekend_shopping": ("outdoors", "car"),
        "weekend_home_tyler": ("home",),
        "weekend_getting_ready": ("home", "bathroom", "bedroom"),
        "weekend_club_night": ("club", "bathroom", "car"),
    }
)

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MIME_RE = re.compile(r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "id",
        "type",
        "status",
        "full_key",
        "preview_key",
        "poster_key",
        "mime_type",
        "aspect_ratio",
        "duration_seconds",
        "sha256",
        "explicitness",
        "min_heat",
        "presentation",
        "tags",
    }
)


class CatalogValidationError(ValueError):
    """Raised when one or more catalog invariants are invalid."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("Invalid media catalog:\n- " + "\n- ".join(self.errors))


@dataclass(frozen=True, slots=True)
class MediaPresentation:
    mode: str
    periods: tuple[str, ...]
    current_description: str | None
    past_description: str


@dataclass(frozen=True, slots=True)
class MediaItem:
    id: str
    media_type: str
    status: str
    full_key: str
    preview_key: str
    poster_key: str | None
    mime_type: str
    aspect_ratio: float
    duration_seconds: float | None
    sha256: str
    explicitness: str
    min_heat: str
    presentation: MediaPresentation
    tags: Mapping[str, tuple[str, ...]]

    @property
    def active(self) -> bool:
        return self.status == "active"

    def description_for_period(self, period: str) -> tuple[str, bool]:
        is_current = (
            self.presentation.mode == "current_compatible"
            and period in self.presentation.periods
        )
        if is_current and self.presentation.current_description:
            return self.presentation.current_description, True
        return self.presentation.past_description, False


class MediaCatalog:
    """Immutable, indexed collection of validated media items."""

    def __init__(self, items: Iterable[MediaItem], *, version: int = 1):
        materialized = tuple(items)
        self.version = version
        self.items = materialized
        self._by_id = {item.id: item for item in materialized}

    def get(self, content_id: str) -> MediaItem | None:
        return self._by_id.get(content_id)

    def require(self, content_id: str) -> MediaItem:
        item = self.get(content_id)
        if item is None:
            raise KeyError(f"Unknown media content_id: {content_id}")
        return item

    def active_items(self) -> tuple[MediaItem, ...]:
        return tuple(item for item in self.items if item.active)


def _clean_text(value: object, field: str, errors: list[str], *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be a non-empty string")
        return None
    cleaned = value.strip()
    if len(cleaned) > 240 or any(ord(char) < 32 for char in cleaned):
        errors.append(f"{field} must be a single safe line of at most 240 characters")
        return None
    return cleaned


def _object_key(value: object, field: str, errors: list[str], *, optional: bool = False) -> str | None:
    key = _clean_text(value, field, errors, optional=optional)
    if key is None:
        return None
    if (
        "://" in key
        or key.startswith(("/", "\\"))
        or "\\" in key
        or any(part in ("", ".", "..") for part in key.split("/"))
    ):
        errors.append(f"{field} must be a relative R2 object key, not a URL or path traversal")
        return None
    return key


def _parse_item(raw: object, index: int, errors: list[str]) -> MediaItem | None:
    prefix = f"items[{index}]"
    if not isinstance(raw, dict):
        errors.append(f"{prefix} must be a mapping")
        return None

    unknown = sorted(set(raw) - _TOP_LEVEL_FIELDS)
    if unknown:
        errors.append(f"{prefix} has unknown fields: {', '.join(unknown)}")

    content_id = _clean_text(raw.get("id"), f"{prefix}.id", errors)
    if content_id and not _ID_RE.fullmatch(content_id):
        errors.append(f"{prefix}.id must match {_ID_RE.pattern}")

    media_type = _clean_text(raw.get("type"), f"{prefix}.type", errors)
    if media_type not in MEDIA_TYPES:
        errors.append(f"{prefix}.type must be photo or video")

    status = _clean_text(raw.get("status"), f"{prefix}.status", errors)
    if status not in MEDIA_STATUSES:
        errors.append(f"{prefix}.status must be active or inactive")

    full_key = _object_key(raw.get("full_key"), f"{prefix}.full_key", errors)
    preview_key = _object_key(raw.get("preview_key"), f"{prefix}.preview_key", errors)
    poster_key = _object_key(
        raw.get("poster_key"), f"{prefix}.poster_key", errors, optional=True
    )
    if full_key and preview_key and full_key == preview_key:
        errors.append(f"{prefix} full_key and preview_key must be different objects")
    if media_type == "video" and not poster_key:
        errors.append(f"{prefix}.poster_key is required for video")
    if poster_key and poster_key in {full_key, preview_key}:
        errors.append(f"{prefix}.poster_key must be a separate preview object")
    image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".avif")
    if preview_key and not preview_key.lower().endswith(image_extensions):
        errors.append(f"{prefix}.preview_key must reference a supported image preview")
    if poster_key and not poster_key.lower().endswith(image_extensions):
        errors.append(f"{prefix}.poster_key must reference a supported image poster")

    mime_type = _clean_text(raw.get("mime_type"), f"{prefix}.mime_type", errors)
    if mime_type and not _MIME_RE.fullmatch(mime_type.lower()):
        errors.append(f"{prefix}.mime_type is not valid")
    if media_type == "photo" and mime_type and not mime_type.startswith("image/"):
        errors.append(f"{prefix}.mime_type must be image/* for photo")
    if media_type == "video" and mime_type and not mime_type.startswith("video/"):
        errors.append(f"{prefix}.mime_type must be video/* for video")

    aspect = raw.get("aspect_ratio")
    if isinstance(aspect, bool) or not isinstance(aspect, (int, float)) or aspect <= 0:
        errors.append(f"{prefix}.aspect_ratio must be a positive number")
        aspect = 1.0
    duration = raw.get("duration_seconds")
    if media_type == "video":
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            errors.append(f"{prefix}.duration_seconds must be positive for video")
            duration = None
    elif duration is not None:
        errors.append(f"{prefix}.duration_seconds must be null for photo")
        duration = None

    checksum = _clean_text(raw.get("sha256"), f"{prefix}.sha256", errors)
    if checksum:
        checksum = checksum.lower()
        if not _SHA256_RE.fullmatch(checksum):
            errors.append(f"{prefix}.sha256 must be 64 hexadecimal characters")

    explicitness = _clean_text(raw.get("explicitness"), f"{prefix}.explicitness", errors)
    if explicitness not in EXPLICITNESS_LEVELS:
        errors.append(f"{prefix}.explicitness must be one of {', '.join(EXPLICITNESS_LEVELS)}")
    min_heat = _clean_text(raw.get("min_heat"), f"{prefix}.min_heat", errors)
    if min_heat not in HEAT_LEVELS:
        errors.append(f"{prefix}.min_heat must be one of {', '.join(HEAT_LEVELS)}")

    presentation_raw = raw.get("presentation")
    presentation: MediaPresentation | None = None
    if not isinstance(presentation_raw, dict):
        errors.append(f"{prefix}.presentation must be a mapping")
    else:
        unknown_presentation = sorted(
            set(presentation_raw)
            - {"mode", "periods", "current_description", "past_description"}
        )
        if unknown_presentation:
            errors.append(
                f"{prefix}.presentation has unknown fields: {', '.join(unknown_presentation)}"
            )
        mode = _clean_text(
            presentation_raw.get("mode"), f"{prefix}.presentation.mode", errors
        )
        if mode not in PRESENTATION_MODES:
            errors.append(
                f"{prefix}.presentation.mode must be current_compatible or past_only"
            )
        periods_raw = presentation_raw.get("periods", [])
        periods: list[str] = []
        if not isinstance(periods_raw, list) or not all(
            isinstance(value, str) for value in periods_raw
        ):
            errors.append(f"{prefix}.presentation.periods must be a list of period names")
        else:
            periods = [value.strip() for value in periods_raw]
            unknown_periods = sorted(set(periods) - KNOWN_PERIODS)
            if unknown_periods:
                errors.append(
                    f"{prefix}.presentation.periods contains unknown values: "
                    + ", ".join(unknown_periods)
                )
            if len(periods) != len(set(periods)):
                errors.append(f"{prefix}.presentation.periods contains duplicates")
        current_description = _clean_text(
            presentation_raw.get("current_description"),
            f"{prefix}.presentation.current_description",
            errors,
            optional=True,
        )
        past_description = _clean_text(
            presentation_raw.get("past_description"),
            f"{prefix}.presentation.past_description",
            errors,
        )
        if mode == "current_compatible" and (not periods or not current_description):
            errors.append(
                f"{prefix}.presentation current_compatible requires periods and current_description"
            )
        if mode == "past_only" and periods:
            errors.append(f"{prefix}.presentation past_only must have an empty periods list")
        if mode and past_description:
            presentation = MediaPresentation(
                mode=mode,
                periods=tuple(periods),
                current_description=current_description,
                past_description=past_description,
            )

    tags_raw = raw.get("tags")
    parsed_tags: dict[str, tuple[str, ...]] = {}
    if not isinstance(tags_raw, dict):
        errors.append(f"{prefix}.tags must be a mapping")
    else:
        unknown_groups = sorted(set(tags_raw) - set(TAG_VALUES))
        if unknown_groups:
            errors.append(f"{prefix}.tags has unknown groups: {', '.join(unknown_groups)}")
        for group, values in tags_raw.items():
            if group not in TAG_VALUES:
                continue
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                errors.append(f"{prefix}.tags.{group} must be a non-empty string list")
                continue
            normalized = tuple(value.strip().lower() for value in values)
            unknown_values = sorted(set(normalized) - TAG_VALUES[group])
            if unknown_values:
                errors.append(
                    f"{prefix}.tags.{group} has unknown values: {', '.join(unknown_values)}"
                )
            if len(normalized) != len(set(normalized)):
                errors.append(f"{prefix}.tags.{group} contains duplicates")
            parsed_tags[group] = normalized
        if not parsed_tags.get("location"):
            errors.append(f"{prefix}.tags.location must contain at least one location")

    required_values = (
        content_id,
        media_type if media_type in MEDIA_TYPES else None,
        status if status in MEDIA_STATUSES else None,
        full_key,
        preview_key,
        mime_type,
        checksum if checksum and _SHA256_RE.fullmatch(checksum) else None,
        explicitness if explicitness in EXPLICITNESS_LEVELS else None,
        min_heat if min_heat in HEAT_LEVELS else None,
        presentation,
    )
    if any(value is None for value in required_values) or not parsed_tags.get("location"):
        return None

    return MediaItem(
        id=content_id,
        media_type=media_type,
        status=status,
        full_key=full_key,
        preview_key=preview_key,
        poster_key=poster_key,
        mime_type=mime_type.lower(),
        aspect_ratio=float(aspect),
        duration_seconds=float(duration) if duration is not None else None,
        sha256=checksum,
        explicitness=explicitness,
        min_heat=min_heat,
        presentation=presentation,
        tags=MappingProxyType(parsed_tags),
    )


def load_media_catalog(path: str | Path = MEDIA_CATALOG_FILE) -> MediaCatalog:
    """Load and fully validate a YAML catalog.

    This validates shape, controlled values and object-key safety. Use
    :func:`validate_catalog_objects` at deployment/startup when an R2 HEAD
    adapter is available to additionally verify remote object existence.
    """

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogValidationError([f"cannot load {source}: {exc}"]) from exc

    errors: list[str] = []
    if not isinstance(payload, dict):
        raise CatalogValidationError(["catalog root must be a mapping"])
    unknown_root = sorted(set(payload) - {"version", "items"})
    if unknown_root:
        errors.append(f"catalog has unknown root fields: {', '.join(unknown_root)}")
    version = payload.get("version")
    if version != 1:
        errors.append("catalog.version must be 1")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        errors.append("catalog.items must be a list")
        raw_items = []
    items: list[MediaItem] = []
    for index, raw in enumerate(raw_items):
        item = _parse_item(raw, index, errors)
        if item:
            items.append(item)

    def duplicates(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        duplicate: set[str] = set()
        for value in values:
            if value in seen:
                duplicate.add(value)
            seen.add(value)
        return sorted(duplicate)

    all_object_keys = (
        key
        for item in items
        for key in (item.full_key, item.preview_key, item.poster_key)
        if key
    )
    for label, values in (
        ("content IDs", (item.id for item in items)),
        ("R2 object keys", all_object_keys),
        ("full object checksums", (item.sha256 for item in items)),
    ):
        found = duplicates(values)
        if found:
            errors.append(f"duplicate {label}: {', '.join(found)}")

    if errors:
        raise CatalogValidationError(errors)
    return MediaCatalog(items, version=version)


async def validate_catalog_objects(
    catalog: MediaCatalog,
    object_exists: Callable[
        [str], bool | Mapping[str, object] | Awaitable[bool | Mapping[str, object]]
    ],
) -> None:
    """Verify referenced R2 objects with a caller-provided HEAD probe.

    A boolean probe provides backwards-compatible existence checks. A mapping
    probe may additionally provide ``exists``, ``content_type``/``mime_type``,
    ``sha256``/``checksum`` and ``content_length``; when supplied, full object
    MIME/checksum and derivative image MIME are validated as deployment gates.
    """

    validation_errors: list[str] = []
    checked: set[str] = set()
    for item in catalog.items:
        for role, key in (
            ("full", item.full_key),
            ("preview", item.preview_key),
            ("poster", item.poster_key),
        ):
            if not key or key in checked:
                continue
            checked.add(key)
            result = object_exists(key)
            result = await result if inspect.isawaitable(result) else result
            metadata = result if isinstance(result, Mapping) else None
            exists = bool(metadata.get("exists", True)) if metadata is not None else bool(result)
            if not exists:
                validation_errors.append(f"missing R2 object: {key}")
                continue
            if metadata is None:
                continue
            content_type = str(
                metadata.get("content_type") or metadata.get("mime_type") or ""
            ).lower()
            if role == "full":
                if not content_type:
                    validation_errors.append(f"R2 object {key} has no content type")
                elif content_type != item.mime_type:
                    validation_errors.append(
                        f"R2 object {key} MIME {content_type} does not match {item.mime_type}"
                    )
                checksum = str(
                    metadata.get("sha256") or metadata.get("checksum") or ""
                ).lower()
                if not checksum:
                    validation_errors.append(f"R2 object {key} has no SHA-256 metadata")
                elif checksum != item.sha256:
                    validation_errors.append(f"R2 object {key} SHA-256 does not match catalog")
            elif not content_type.startswith("image/"):
                validation_errors.append(
                    f"R2 {role} object {key} must have an image MIME type"
                )
            length = metadata.get("content_length")
            try:
                parsed_length = int(length) if not isinstance(length, bool) else 0
            except (TypeError, ValueError):
                parsed_length = 0
            if parsed_length <= 0:
                validation_errors.append(
                    f"R2 object {key} must have a positive content length"
                )
    if validation_errors:
        raise CatalogValidationError(validation_errors)


def heat_allows(item: MediaItem, heat: str) -> bool:
    """Return whether the conversation heat reaches the item's minimum."""

    normalized_heat = "rising" if heat == "medium" else heat
    if normalized_heat not in HEAT_LEVELS:
        normalized_heat = "low"
    return HEAT_LEVELS.index(normalized_heat) >= HEAT_LEVELS.index(item.min_heat)
