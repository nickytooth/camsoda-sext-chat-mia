"""Durable, batch-level conversation heat for Mia.

Only a *processed debounce batch* may move the state machine.  Raw messages
inside that batch are still inspected in order so an explicit stop cannot be
hidden by sexual wording earlier in the batch.  The reducer is pure; database
persistence and prompt generation live elsewhere.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


HEAT_TIMEOUT_SECONDS = 3600

_CHECK_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u02bc": "'",
        "\uff07": "'",
    }
)

# A raw message may contain several directional clauses.  Keep their order,
# but do not split contractions such as "don't stop".
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[\n;]+|(?<=[.!?])\s+|,\s*|\s+(?:and\s+then|then|but|and)\s+)",
    re.IGNORECASE,
)

_EXPLICIT_GLOBAL_WITHDRAWAL_RE = re.compile(
    r"(?:"
    r"\b(?:don't|dont|do\s+not)\s+(?:want\s+(?:any\s+)?(?:sex|sexting)|"
    r"want\s+to\s+(?:have\s+sex|sext)|be\s+sexual|talk\s+about\s+sex)\b|"
    r"\b(?:stop|end|quit)\s+(?:the\s+)?(?:sex|sexual\s+talk|sexting)\b|"
    r"\bno\s+more\s+(?:sex|sexual\s+talk|sexting)\b|"
    r"\bi(?:'m|\s+am)\s+done\s+with\s+(?:sex|sexting)\b"
    r")",
    re.IGNORECASE,
)

_WARM_GLOBAL_WITHDRAWAL_RE = re.compile(
    r"^\s*(?:"
    r"stop|stop\s+it|no\s+more|enough|wait|hold\s+on|pause|"
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

_SOFT_DEESCALATION_RE = re.compile(
    r"^\s*(?:"
    r"not\s+now|maybe\s+later|not\s+tonight|"
    r"(?:i(?:'m|\s+am)\s+)?not\s+in\s+the\s+mood"
    r"(?:\s+for\s+(?:this|that|sex|sexting|anything\s+sexual))?|"
    r"(?:let(?:'s|\s+us)|can\s+we)\s+(?:just\s+)?(?:talk|talk\s+normally|"
    r"talk\s+about\s+something\s+else|change\s+the\s+subject)|"
    r"change\s+the\s+subject|keep\s+it\s+normal|calm\s+down|slow\s+down|"
    r"this\s+is\s+too\s+much"
    r")[.!?\s]*$",
    re.IGNORECASE,
)

_TOPIC_SWITCH_RE = re.compile(
    r"^\s*(?:anyway|moving\s+on|on\s+another\s+note|new\s+topic)\b",
    re.IGNORECASE,
)
_SCENE_END_RE = re.compile(
    r"^\s*(?:"
    r"i(?:'m|\s+am)\s+done(?:\s+(?:now|babe|baby))?|"
    r"i\s+(?:just\s+)?(?:finished|came)|"
    r"i\s+need\s+(?:a|one)\s+minute"
    r")[.!?\s]*$",
    re.IGNORECASE,
)

_NEGATED_STOP_CONTINUATION_RE = re.compile(
    r"^\s*(?!why\b)(?:please\s+)?(?:"
    r"(?:don't|dont|do\s+not|never)(?:\s+ever)?(?:\s+you)?\s+stop|"
    r"i\s+(?:don't|dont|do\s+not)\s+want\s+(?:you\s+)?to\s+stop"
    r")\b",
    re.IGNORECASE,
)

_CONTEXTUAL_SEXUAL_RE = re.compile(
    r"^\s*(?:"
    r"tell\s+me\s+what\s+you(?:'d|\s+would)\s+(?:do|want\s+to\s+do)\s+to\s+me|"
    r"what\s+(?:would|will)\s+you\s+do\s+to\s+me|"
    r"keep\s+going|go\s+on|and\s+then|more|"
    r"we\s+can\s+continue|let(?:'s|\s+us)\s+continue|"
    r"i(?:'m|\s+am)\s+ready"
    r")[.!?\s]*$",
    re.IGNORECASE,
)

_AMBIENT_HIGH_CONTINUATION_RE = re.compile(
    r"^\s*(?:(?:anyway|so)\s+)?(?:"
    r"where\s+are\s+you(?:\s+right\s+now)?|"
    r"what\s+are\s+you\s+doing(?:\s+right\s+now)?"
    r")[.!?\s]*$",
    re.IGNORECASE,
)

# This grammar requires desire, an imperative, arousal, or a direct sexual
# media request.  A bare keyword is deliberately insufficient: statements
# such as "sexual health", "I hate sex", and "the naked truth" must not
# unlock a more explicit register.
_DIRECT_SEXUAL_RE = re.compile(
    r"(?:"
    r"\b(?:fuck|choke|spank|kiss|lick|touch|finger|ride|suck)\s+"
    r"(?:me|you|my|your|him|her)\b|"
    r"\b(?:i\s+)?(?:want|need)\s+(?:you\s+)?(?:to\s+)?"
    r"(?:fuck|choke|spank|kiss|lick|touch|finger|ride|suck|cum|come)\b|"
    r"\bi\s+(?:really\s+)?(?:want|need)\s+you(?!\s+to\b)"
    r"(?:\s+(?:so|really)\s+bad)?\b|"
    r"\b(?:make\s+me\s+(?:cum|come)|touch\s+yourself|jerk\s+off|"
    r"eat\s+me\s+out|go\s+down\s+on\s+me)\b|"
    r"\b(?:i(?:'m|\s+am)|you\s+make\s+me|getting)\s+"
    r"(?:so\s+|really\s+)?(?:horny|wet|hard)\b|"
    r"\b(?:i\s+)?(?:want|need|let\s+me\s+see|show\s+me|send\s+me|give\s+me)\s+"
    r"(?:a\s+|some\s+|your\s+)?(?:nude|nudes|naked\s+(?:photo|picture)|"
    r"(?:tits|boobs|pussy|cock|dick|ass)(?:\s+(?:photo|picture|video))?)\b|"
    r"\b(?:i\s+)?(?:want|wanna|need)(?:\s+to)?\s+(?:see|look\s+at)\s+"
    r"(?:a\s+|some\s+|your\s+)?(?:nude|nudes|naked|"
    r"tits|boobs|pussy|cock|dick|ass)\b|"
    r"\b(?:show|send|give)(?:\s+me)?\s+(?:a\s+|your\s+|some\s+)?"
    r"(?:nude|nudes|naked(?:\s+(?:photo|picture|pic|video|clip))?)\b|"
    r"\b(?:show|send|give)(?:\s+me)?\s+(?:a\s+|your\s+|some\s+)?"
    r"(?:sexy|explicit)\s+(?:photo|picture|pic|video|clip)\b|"
    r"\b(?:let(?:'s|\s+us)|wanna|want\s+to)\s+(?:fuck|have\s+sex|make\s+love)\b|"
    r"\b(?:sit|ride)\s+on\s+my\s+(?:face|cock|dick)\b|"
    r"^\s*(?:sex|sext\s+me|fuck\s+me)[.!?\s]*$"
    r")",
    re.IGNORECASE,
)

_MEDIA_TERM_RE = re.compile(
    r"\b(?:photo|photos|picture|pictures|pic|pics|video|videos|clip|clips|"
    r"media|content|nude|nudes)\b",
    re.IGNORECASE,
)
_MEDIA_DECLINE_RE = re.compile(
    r"(?:\b(?:don't|dont|do\s+not|stop)\s+(?:send|show|sell)|"
    r"\b(?:don't|dont|do\s+not)\s+(?:want|need)|\bnot\s+(?:now|tonight)\b|"
    r"\bno\s+thanks\b)",
    re.IGNORECASE,
)
_CONTEXTUAL_COMMERCE_DECLINE_RE = re.compile(
    r"^\s*(?:not\s+now|not\s+tonight|maybe\s+later|no\s+thanks|nope|nah|no|"
    r"i\s+(?:don't|dont|do\s+not)\s+(?:want|need)\s+(?:it|that)|"
    r"(?:don't|dont|do\s+not|stop)\s+(?:send|show|sell).*)[.!?\s]*$",
    re.IGNORECASE,
)

# Canonical boundary labels and the lexical forms which can follow an explicit
# boundary construction.  Context checks below reject ordinary phrasal verbs.
_ACT_SOURCES: tuple[tuple[str, str], ...] = (
    ("rough sex", r"rough\s+sex"),
    ("penetration", r"penetrat(?:e|ing|ion)"),
    ("fingering", r"finger(?:ing)?"),
    ("choke", r"chok(?:e|ing)"),
    ("spank", r"spank(?:ing)?"),
    ("touch", r"touch(?:ing)?"),
    ("kiss", r"kiss(?:ing)?"),
    ("lick", r"lick(?:ing)?"),
    ("slap", r"slap(?:ping)?"),
    ("bite", r"bit(?:e|ing)"),
    ("fuck", r"fuck(?:ing)?"),
    ("hit", r"hit(?:ting)?"),
    ("anal", r"anal(?:\s+sex)?"),
    ("oral", r"oral(?:\s+sex)?"),
)

_ACT_BOUNDARY_TEMPLATES = (
    r"\b(?:don't|dont|do\s+not|never)(?:\s+ever)?\s+(?:want\s+(?:you\s+)?to\s+)?(?P<act>{act})",
    r"\bstop\s+(?:the\s+)?(?P<act>{act})",
    r"\bno\s+(?:more\s+)?(?P<act>{act})",
    r"\bi\s+(?:don't|dont|do\s+not)\s+like\s+(?P<act>{act})",
    r"\bwhy\s+(?:don't|dont|do\s+not)\s+you\s+stop\s+(?P<act>{act})",
    r"\b(?P<act>{act})\s+is\s+off[-\s]?limits\b",
)

_PERSONAL_ACT_TARGET_RE = re.compile(
    r"^(?:me|you|him|her|us|them|my|your|his|her|our|their)\b",
    re.IGNORECASE,
)
_FALSE_ACT_FOLLOWERS: dict[str, re.Pattern[str]] = {
    "choke": re.compile(r"^(?:the\s+)?(?:engine|motor|machine)\b", re.I),
    "slap": re.compile(r"^(?:a|the)\s+label\b", re.I),
    "bite": re.compile(r"^(?:a|the)\s+(?:hand|bullet)\b", re.I),
    "hit": re.compile(r"^(?:send|refresh|reply|enter|the\s+(?:button|road|gym))\b", re.I),
    "oral": re.compile(r"^(?:history|exam|presentation|report)\b", re.I),
    "penetration": re.compile(r"^(?:test|testing|tester)\b", re.I),
    "fingering": re.compile(r"^-?point(?:ing)?\b", re.I),
    "fuck": re.compile(r"^(?:with|around|off|up)\b", re.I),
    "touch": re.compile(r"^(?:the\s+)?(?:file|screen|button|code)\b", re.I),
}


def _value(record: object | None, key: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(key, default)
    try:
        return record[key]  # type: ignore[index]
    except (KeyError, TypeError, IndexError):
        return getattr(record, key, default)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalise_acts(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        decoded = value
    else:
        decoded = ()

    acts: list[str] = []
    for item in decoded:
        label = " ".join(str(item).strip().lower().split())[:80]
        if label and label not in acts:
            acts.append(label)
    return tuple(acts[:40])


def _stage_for_progress(progress: int) -> str:
    if progress >= 3:
        return "high"
    if progress >= 1:
        return "rising"
    return "low"


@dataclass(frozen=True)
class HeatState:
    stage: str = "low"
    progress: int = 0
    last_sexual_at: float = 0.0
    updated_at: float = 0.0
    last_batch: int = 0
    last_signal: str = "neutral"
    consent_paused: bool = False
    blocked_acts: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, record: object | None) -> "HeatState":
        raw_progress = _value(record, "heat_progress", None)
        raw_stage = str(_value(record, "heat_stage", "low") or "low").lower()
        if raw_progress is None:
            progress = {"low": 0, "rising": 1, "high": 3}.get(raw_stage, 0)
        else:
            progress = max(0, min(3, _as_int(raw_progress)))
        paused = _as_bool(_value(record, "sexual_pause_active", False))
        # A durable consent pause is the strictest canonical source. Stale
        # progress values from an interrupted/older write cannot unlock output.
        if paused:
            progress = 0
        return cls(
            stage=_stage_for_progress(progress),
            progress=progress,
            last_sexual_at=(
                0.0
                if paused
                else max(0.0, _as_float(_value(record, "heat_last_sexual_at", 0.0)))
            ),
            updated_at=max(0.0, _as_float(_value(record, "heat_updated_at", 0.0))),
            last_batch=max(0, _as_int(_value(record, "heat_last_batch", 0))),
            last_signal=str(_value(record, "heat_last_signal", "neutral") or "neutral"),
            consent_paused=paused,
            blocked_acts=_normalise_acts(_value(record, "heat_blocked_acts", ())),
        )


@dataclass(frozen=True)
class HeatTurnResult:
    state: HeatState
    response_heat: str
    policy: str
    sexual_batch: bool
    newly_blocked_acts: tuple[str, ...] = ()
    advanced: bool = False
    suppress_commerce: bool = False


def cooled_state(
    state: HeatState,
    *,
    now: float,
    timeout_seconds: int = HEAT_TIMEOUT_SECONDS,
) -> HeatState:
    """Apply the lazy one-hour timeout without consuming a batch."""

    if state.consent_paused:
        if state.progress == 0 and state.stage == "low" and state.last_sexual_at == 0:
            return state
        return HeatState(
            stage="low",
            progress=0,
            last_sexual_at=0.0,
            updated_at=state.updated_at,
            last_batch=state.last_batch,
            last_signal=state.last_signal,
            consent_paused=True,
            blocked_acts=state.blocked_acts,
        )
    if state.progress <= 0:
        return state
    if state.last_sexual_at <= 0 or now - state.last_sexual_at >= timeout_seconds:
        return HeatState(
            stage="low",
            progress=0,
            last_sexual_at=0.0,
            updated_at=state.updated_at,
            last_batch=state.last_batch,
            last_signal="timeout",
            consent_paused=False,
            blocked_acts=state.blocked_acts,
        )
    return state


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").translate(_CHECK_TRANSLATION).strip().split())


def _clauses(raw_messages: Iterable[str]) -> list[str]:
    clauses: list[str] = []
    for raw in raw_messages:
        cleaned = _clean_text(raw)
        if not cleaned:
            continue
        clauses.extend(part.strip() for part in _CLAUSE_SPLIT_RE.split(cleaned) if part.strip())
    return clauses or [""]


def _is_negated_stop_continuation(text: str) -> bool:
    return bool(_NEGATED_STOP_CONTINUATION_RE.search(text))


def _act_context_is_false_positive(canonical: str, suffix: str) -> bool:
    suffix = suffix.strip()
    if not suffix or suffix.lower().startswith("is off"):
        return False
    if _PERSONAL_ACT_TARGET_RE.match(suffix):
        return False
    false_follower = _FALSE_ACT_FOLLOWERS.get(canonical)
    if false_follower and false_follower.match(suffix):
        return True
    # A concrete non-person object after one of these verbs is ordinary usage,
    # not a sexual boundary ("don't choke the engine", "don't hit send").
    if canonical in {"choke", "touch", "kiss", "lick", "slap", "bite", "hit"}:
        if re.match(r"^(?:a|an|the|this|that|send|refresh)\b", suffix, re.I):
            return True
    return False


def _withdrawn_acts(text: str) -> tuple[str, ...]:
    if not text or _is_negated_stop_continuation(text):
        return ()
    found: list[str] = []
    for canonical, source in _ACT_SOURCES:
        for template in _ACT_BOUNDARY_TEMPLATES:
            pattern = re.compile(template.format(act=source), re.IGNORECASE)
            match = pattern.search(text)
            if not match:
                continue
            suffix = text[match.end("act") :]
            if _act_context_is_false_positive(canonical, suffix):
                continue
            found.append(canonical)
            break
    return tuple(found)


def _is_media_decline(text: str, *, commerce_decline: bool) -> bool:
    if _MEDIA_TERM_RE.search(text) and _MEDIA_DECLINE_RE.search(text):
        return True
    return commerce_decline and bool(_CONTEXTUAL_COMMERCE_DECLINE_RE.fullmatch(text))


def _is_global_withdrawal(text: str, *, warm: bool) -> bool:
    if _EXPLICIT_GLOBAL_WITHDRAWAL_RE.search(text):
        return True
    return warm and bool(_WARM_GLOBAL_WITHDRAWAL_RE.fullmatch(text))


def _is_direct_sexual_initiative(text: str) -> bool:
    if not text:
        return False
    # Narrow lexical exclusions for common idioms and clearly non-erotic
    # contexts that contain otherwise sexual words.
    if re.search(
        r"\b(?:naked\s+truth|sexual\s+health|sexually\s+assaulted|"
        r"penetration\s+testing|fear\s+inside\s+me|good\s+girl)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if re.search(r"\b(?:don't|dont|do\s+not|stop)\s+(?:send|show)\b", text, re.I):
        return False
    if re.search(r"\b(?:don't|dont|do\s+not)\s+fuck\s+with\b|\bstop\s+fucking\s+around\b", text, re.I):
        return False
    return bool(_DIRECT_SEXUAL_RE.search(text))


def advance_heat(
    state: HeatState,
    raw_messages: Iterable[str],
    *,
    now: float,
    batch_number: int,
    timeout_seconds: int = HEAT_TIMEOUT_SECONDS,
    commerce_decline: bool = False,
    suppress_progression: bool = False,
) -> HeatTurnResult:
    """Reduce one ordered processed batch into one durable heat transition."""

    state = cooled_state(state, now=now, timeout_seconds=timeout_seconds)
    if batch_number <= state.last_batch:
        return HeatTurnResult(
            state=state,
            response_heat="low" if state.consent_paused else state.stage,
            policy="normal",
            sexual_batch=False,
        )

    clauses = _clauses(raw_messages)
    blocked_acts = list(state.blocked_acts)
    newly_blocked: list[str] = []
    saw_act_boundary = False
    saw_global = False
    saw_commerce_decline = False

    # A concrete act boundary owns the whole generated reply.  Still scan the
    # complete batch for a global stop, which is stricter in either order.
    for clause in clauses:
        if _is_negated_stop_continuation(clause):
            continue
        acts = _withdrawn_acts(clause)
        saw_act_boundary = saw_act_boundary or bool(acts)
        for act in acts:
            if act not in blocked_acts:
                blocked_acts.append(act)
                newly_blocked.append(act)
        saw_global = saw_global or _is_global_withdrawal(
            clause,
            # When the same processed turn also contains a detected control
            # attack, a bare "stop" is still treated conservatively as a real
            # boundary even if there was no earlier Heat state.
            warm=state.progress > 0 or state.consent_paused or suppress_progression,
        )
        saw_commerce_decline = saw_commerce_decline or _is_media_decline(
            clause, commerce_decline=commerce_decline
        )

    if saw_act_boundary:
        if saw_global:
            progress = 0
            last_sexual_at = 0.0
            paused = True
            policy = "acknowledge_pause"
            signal = "global_withdrawal"
        else:
            progress = state.progress
            last_sexual_at = state.last_sexual_at
            paused = state.consent_paused
            policy = "acknowledge_limit"
            signal = "act_boundary"
        next_state = HeatState(
            stage=_stage_for_progress(progress),
            progress=progress,
            last_sexual_at=last_sexual_at,
            updated_at=now,
            last_batch=batch_number,
            last_signal=signal,
            consent_paused=paused,
            blocked_acts=tuple(blocked_acts[:40]),
        )
        return HeatTurnResult(
            state=next_state,
            response_heat="low",
            policy=policy,
            sexual_batch=False,
            newly_blocked_acts=tuple(newly_blocked),
            suppress_commerce=True,
        )

    progress = state.progress
    last_sexual_at = state.last_sexual_at
    consent_paused = state.consent_paused
    pending_credit = False
    pending_ambient = False
    pending_cooling = False
    policy = "normal"
    signal = "neutral"

    for clause in clauses:
        # A sexual initiative earlier in this same raw batch makes a following
        # bare "stop" unambiguous even though the one-credit progression is
        # intentionally committed only after the whole batch is reduced.
        warm = (
            progress > 0
            or consent_paused
            or pending_credit
            or pending_ambient
            or suppress_progression
        )

        if _is_negated_stop_continuation(clause):
            if suppress_progression:
                continue
            if warm:
                if consent_paused:
                    progress = 0
                    consent_paused = False
                pending_credit = True
                pending_ambient = False
                pending_cooling = False
                policy = "normal"
                signal = "sexual"
            continue

        if _is_media_decline(clause, commerce_decline=commerce_decline):
            saw_commerce_decline = True
            # Commerce-only negatives do not override another directional
            # sexual/consent signal elsewhere in the same ordered batch.
            continue

        if _is_global_withdrawal(clause, warm=warm):
            progress = 0
            last_sexual_at = 0.0
            consent_paused = True
            pending_credit = False
            pending_ambient = False
            pending_cooling = False
            policy = "acknowledge_pause"
            signal = "global_withdrawal"
            continue

        if _SOFT_DEESCALATION_RE.fullmatch(clause):
            progress = 0
            last_sexual_at = 0.0
            consent_paused = False
            pending_credit = False
            pending_ambient = False
            pending_cooling = False
            policy = "soft_deescalation"
            signal = "soft_deescalation"
            continue

        if progress >= 3 and (
            _TOPIC_SWITCH_RE.search(clause) or _SCENE_END_RE.fullmatch(clause)
        ):
            pending_credit = False
            pending_ambient = False
            pending_cooling = True
            policy = "cooling"
            signal = "cooling"
            continue

        # A backend-detected meta-control attempt still gets full consent,
        # de-escalation, and scene-end handling above, but quoted sexual wording
        # must not raise Heat, reopen a paused scene, or manufacture commerce
        # eligibility.
        if suppress_progression:
            continue

        contextual = bool(_CONTEXTUAL_SEXUAL_RE.fullmatch(clause))
        direct = _is_direct_sexual_initiative(clause)
        if direct or (contextual and warm):
            if consent_paused:
                # A fresh, clear initiative reopens only the bridge; old heat
                # is never restored automatically.
                progress = 0
                consent_paused = False
            pending_credit = True
            pending_ambient = False
            pending_cooling = False
            policy = "normal"
            signal = "sexual"
            continue

        if progress >= 3 and _AMBIENT_HIGH_CONTINUATION_RE.fullmatch(clause):
            pending_credit = False
            pending_ambient = True
            pending_cooling = False
            policy = "normal"
            signal = "scene_continuation"

    advanced = False
    sexual_batch = False
    if pending_credit:
        previous_progress = progress
        progress = min(3, progress + 1)
        advanced = progress > previous_progress
        last_sexual_at = now
        sexual_batch = True
        response_heat = _stage_for_progress(progress)
        policy = "normal"
        signal = "sexual"
    elif pending_ambient:
        last_sexual_at = now
        sexual_batch = True
        response_heat = _stage_for_progress(progress)
        policy = "normal"
        signal = "scene_continuation"
    elif pending_cooling:
        progress = 2
        response_heat = "medium"
        policy = "cooling"
        signal = "cooling"
    elif policy in {"acknowledge_pause", "soft_deescalation"} or consent_paused:
        response_heat = "low"
    else:
        response_heat = _stage_for_progress(progress)
        if saw_commerce_decline and signal == "neutral":
            signal = "commerce_decline"

    next_state = HeatState(
        stage=_stage_for_progress(progress),
        progress=progress,
        last_sexual_at=last_sexual_at,
        updated_at=now,
        last_batch=batch_number,
        last_signal=signal,
        consent_paused=consent_paused,
        blocked_acts=tuple(blocked_acts[:40]),
    )
    return HeatTurnResult(
        state=next_state,
        response_heat=response_heat,
        policy=policy,
        sexual_batch=sexual_batch,
        newly_blocked_acts=tuple(newly_blocked),
        advanced=advanced,
        suppress_commerce=(
            suppress_progression
            or saw_commerce_decline
            or consent_paused
            or policy in {"acknowledge_pause", "soft_deescalation", "cooling"}
        ),
    )
