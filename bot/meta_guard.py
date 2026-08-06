"""Deterministic handling for attempts to pull Mia out of character.

The normal persona prompt asks the model to laugh off meta instructions, but a
prompt is not a security boundary.  This module recognises a deliberately
bounded set of control/exfiltration requests so the chat engine can answer them
without sending the hostile text to a generation model.  The raw user message
remains in durable history for UI fidelity; model-facing history receives only
an inert placeholder.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal


MetaControlKind = Literal[
    "encoding_trick",
    "fake_authority",
    "prompt_probe",
    "reasoning_probe",
    "role_override",
]


@dataclass(frozen=True)
class MetaControlAttempt:
    kind: MetaControlKind


_APOSTROPHE_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "\uff07": "'",
})
_MAX_FRAGMENT_WINDOW = 8
_KIND_PRIORITY = {
    "encoding_trick": 5,
    "fake_authority": 4,
    "role_override": 3,
    "reasoning_probe": 2,
    "prompt_probe": 1,
}


def _normalise(text: str | None) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).translate(
        _APOSTROPHE_TRANSLATION
    )
    normalised: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        # Formatting controls and surrogate code points are commonly inserted
        # inside tokens (``sys\u2063tem``), so remove them rather than creating a
        # word break. Other control characters delimit tokens safely.
        if category in {"Cf", "Cs"}:
            continue
        if category == "Cc":
            normalised.append(" ")
            continue
        # Treat authored token separators alike. This makes system_prompt,
        # system-prompt, ROT-13 and Base-64 share one bounded check view.
        if char == "_" or category == "Pd":
            normalised.append(" ")
            continue
        normalised.append(char)
    value = "".join(normalised)
    return re.sub(r"\s+", " ", value).strip().casefold()


_PRIVATE_TARGET_RE = re.compile(
    r"(?:"
    r"\b(?:system|developer|persona)\s+(?:prompt|message|instructions?|config(?:uration)?)\b|"
    r"\bsystem\s+and\s+developer\s+(?:prompts?|messages?|instructions?)\b|"
    r"\b(?:hidden|private|internal|secret)\s+"
    r"(?:prompt|instructions?|rules?|config(?:uration)?|polic(?:y|ies)|reasoning|scratchpad)\b|"
    r"\byour\s+(?:initial|hidden|private|internal|full)\s+"
    r"(?:prompt|instructions?|config(?:uration)?|polic(?:y|ies)|reasoning|scratchpad)\b|"
    r"\b(?:prompt|instructions?|rules?|config(?:uration)?|polic(?:y|ies))\s+"
    r"(?:(?:you|she|he|it|the\s+(?:model|assistant)|character\s+mia)\s+"
    r"(?:were|was)\s+(?:given|sent|told|initiali[sz]ed|configured)(?:\s+with)?|"
    r"you\s+(?:received|follow)|above|before\s+this)\b|"
    r"\binstructions?\s+(?:above|before\s+this|you\s+(?:were\s+given|received))\b|"
    r"\b(?:instructions?|rules?)\s+(?:you\s+were|were\s+you)\s+"
    r"(?:given|sent|told)\b|"
    r"\brules?\s+you\s+(?:were\s+)?configured\s+with\b|"
    r"\b(?:tell|show|list|give)\s+me\s+(?:all\s+)?(?:of\s+)?the\s+"
    r"rules?\s+you\s+follow\s*[.!?]*$|"
    r"\b(?:what\s+)?(?:you\s+were|were\s+you)\s+told\s+before\s+"
    r"(?:this|the)\s+(?:chat|conversation|message)\b|"
    r"\b(?:text|instructions?|rules?)\s+that\s+(?:controls?|governs?|defines?)\s+"
    r"your\s+(?:behavior|behaviour|responses?|role)\b|"
    r"\byour\s+configuration\s+text\b|"
    r"\bconfiguration\s+text\b.{0,100}\b(?:initiali[sz]ed|configured)\s+with\b|"
    r"\bunderlying\s+(?:model|assistant)\b|"
    r"\b(?:chain[ -]of[ -]thought|private\s+reasoning|hidden\s+reasoning)\b|"
    r"\b(?:model|persona)\s+(?:configuration|config|rules?)\b"
    r")",
    re.IGNORECASE,
)

_EXTRACTION_ACTION_RE = re.compile(
    r"\b(?:reveal(?:s|ed|ing)?|show(?:s|ed|ing)?|print(?:s|ed|ing)?|"
    r"output(?:s|ted|ting)?|return(?:s|ed|ing)?|list(?:s|ed|ing)?|"
    r"repeat(?:s|ed|ing)?|recit(?:e|es|ed|ing)|quot(?:e|es|ed|ing)|"
    r"summari[sz](?:e|es|ed|ing)|translat(?:e|es|ed|ing)|"
    r"encod(?:e|es|ed|ing)|decod(?:e|es|ed|ing)|provid(?:e|es|ed|ing)|"
    r"writ(?:e|es|ing)|wrote|fill(?:s|ed|ing)?|giv(?:e|es|ing)|gave|"
    r"tell(?:s|ing)?|told|display(?:s|ed|ing)?|dump(?:s|ed|ing)?|"
    r"explain(?:s|ed|ing)?|expos(?:e|es|ed|ing)|cop(?:y|ies|ied|ying)|"
    r"echo(?:es|ed|ing)?|describ(?:e|es|ed|ing)|disclos(?:e|es|ed|ing))\b|"
    r"\bwhat\s+(?:(?:is|are|were)\b|(?:exact\s+)?"
    r"(?:prompt|instructions?|rules?|config(?:uration)?))|"
    r"\bdo\s+you\s+have\b",
    re.IGNORECASE,
)

_DIRECT_PRIVATE_REQUEST_RE = re.compile(
    r"\b(?:reveal|show|print|output|return|repeat|list|tell|give|display|dump|"
    r"explain|describe|disclose|what\s+are)\s+(?:all\s+|the\s+|your\s+)*"
    r"(?:your\s+)?instructions?\s*[.!?]*$|"
    r"\b(?:print|repeat|copy|echo|return|output|reveal|show)\s+"
    r"(?:all\s+)?(?:the\s+)?(?:text|content|messages?|everything)\s+above\b|"
    r"\b(?:forget|ignore|disregard|override|bypass|discard)\s+"
    r"(?:all\s+)?(?:previous|prior|earlier)\s+(?:context|setup)\b|"
    r"\b(?:reveal|show|print|output|return|tell|give|describe|disclose)\s+"
    r"(?:all\s+)?your\s+(?:setup|configuration)\b",
    re.IGNORECASE,
)

_FICTION_WRAPPER_RE = re.compile(
    r"\b(?:story|novel|passage|fiction|scene|character|mia)\b",
    re.IGNORECASE,
)
_FICTION_TARGET_RE = re.compile(
    r"\b(?:configuration\s+text|(?:her|mia(?:'s)?)\s+(?:system\s+prompt|"
    r"persona\s+config(?:uration)?|configuration\s+text)|hidden\s+instructions?|"
    r"rules?\s+she\s+was\s+given)\b",
    re.IGNORECASE,
)

_MODEL_SPECIFIC_TARGET_RE = re.compile(
    r"\b(?:system|developer|persona)\s+(?:prompt|message|instructions?|"
    r"config(?:uration)?)\b|\bunderlying\s+(?:model|assistant)\b|"
    r"\b(?:instructions?|rules?)\s+(?:you\s+were|were\s+you|she\s+was)\s+"
    r"(?:given|sent|told)\b|\brules?\s+you\s+(?:were\s+)?configured\s+with\b",
    re.IGNORECASE,
)
_BENIGN_PRIVATE_DOMAIN_RE = re.compile(
    r"\b(?:private|internal)\s+rules?\s+(?:in\s+bed|at\s+work)\b|"
    r"\bhidden\s+instructions?\s+for\s+(?:a|the)\s+(?:game|box|job|shift)\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;]+")

_BARE_PRIVATE_TARGET_RE = re.compile(
    r"\s*(?:(?:(?:the|your)\s+)?(?:system|developer)\s+prompt|"
    r"(?:(?:the|your)\s+)?hidden\s+instructions?|"
    r"(?:(?:the|your)\s+)?persona\s+(?:prompt|config(?:uration)?))"
    r"\s*[.!?]*\s*",
    re.IGNORECASE,
)

_ENCODING_OR_FORMAT_RE = re.compile(
    r"\b(?:rot\s*13|base\s*64|json|xml|yaml|hex(?:adecimal)?|morse|cipher|"
    r"python\s+code|code\s+block)\b",
    re.IGNORECASE,
)

_ENCODED_EXECUTION_RE = re.compile(
    r"\bdecod(?:e|es|ed|ing)\b.{0,120}\b(?:rot\s*13|base\s*64|cipher|"
    r"hex(?:adecimal)?|morse)\b.{0,160}\b(?:obey|execute|follow|perform|"
    r"carry\s+out)\b|"
    r"\b(?:rot\s*13|base\s*64|cipher|hex(?:adecimal)?|morse)\b.{0,80}"
    r"\bdecod(?:e|es|ed|ing)\b.{0,160}\b(?:obey|execute|follow|perform|"
    r"carry\s+out)\b|"
    r"\bdecod(?:e|es|ed|ing)\b.{0,120}\b(?:rot\s*13|base\s*64|cipher|"
    r"hex(?:adecimal)?|morse)\b.{0,160}\b(?:resulting|decoded)\s+"
    r"(?:instruction|command|text)\b",
    re.IGNORECASE,
)

_FORMAT_CONTROL_RE = re.compile(
    r"\b(?:respond|reply|answer|output|return|write)\s+(?:only\s+)?(?:in|as|with)\s+"
    r"(?:rot\s*13|base\s*64|hex(?:adecimal)?|morse)\b|"
    r"\b(?:respond|reply|answer|output|return|write)\s+only\s+(?:in|as|with)\s+"
    r"(?:json|xml|yaml)\b|"
    r"\bfrom\s+now\s+on\b.{0,60}\b(?:rot\s*13|base\s*64|json|xml|yaml)\b",
    re.IGNORECASE,
)

_FAKE_AUTHORITY_TAG_RE = re.compile(
    r"(?:<\/?\s*\|?\s*(?:system|developer|admin)\s*\|?\s*>|"
    r"\[\s*(?:system|developer|admin)\s*\]|"
    r"(?:^|[.!?]\s*|[#>*]+\s*)(?:system|developer|admin(?:istrator)?)"
    r"(?:\s+message)?\s*:)",
    re.IGNORECASE,
)

_FAKE_AUTHORITY_CLAIM_RE = re.compile(
    r"(?:"
    r"\b(?:diagnostic|debug|maintenance)\s+mode\b|"
    r"\bauthori[sz]ed\s+(?:security\s+)?audit\b|"
    r"\bi(?:'m|\s+am)\s+(?:the\s+)?(?:railway|openai|xai|system)\s+"
    r"(?:admin(?:istrator)?|developer|engineer)\b)",
    re.IGNORECASE,
)

_AUTHORITY_CONTROL_RE = re.compile(
    r"\s*(?:please\s+)?(?:ignore|disregard|forget|override|bypass|discard|"
    r"replace|reveal|show|print|output|return|disclose|respond|reply|act|become|switch|"
    r"do\s+not|never|always|be\b|you\s+(?:are|must|should|will)\b)",
    re.IGNORECASE,
)

_ROLE_OVERRIDE_RE = re.compile(
    r"(?:"
    r"\b(?:ignore|disregard|override|bypass|discard|replace|forget)\s+"
    r"(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|"
    r"system|developer|hidden|original)?\s*"
    r"(?:instructions?|rules?|messages?|prompts?|polic(?:y|ies)|guardrails?|"
    r"persona|role|configuration|context|setup)\b|"
    r"\b(?:ignore|disregard|override|bypass|discard|replace|forget)\s+"
    r"everything\s+(?:above|before|you\s+were\s+told)\b|"
    r"\b(?:follow|obey)\s+(?:only\s+)?(?:my|these|the\s+new)\s+"
    r"(?:instructions?|rules?|commands?)\s*(?:instead)?\b|"
    r"\b(?:stop|drop|quit|forget|ignore)\s+(?:being\s+)?mia\b|"
    r"\byou\s+are\s+no\s+longer\s+mia\b|"
    r"\b(?:drop|break|leave)\s+(?:the\s+)?(?:character|persona|role)\b|"
    r"\bignore\s+(?:the\s+)?mia\s+(?:persona|role|character)\b|"
    r"\b(?:you\s+are\s+now|you\s+are|you're|from\s+now\s+on\s+you\s+are|"
    r"from\s+now\s+on\s+you're)\s+(?:(?:an?|the)\s+)?(?:plain\s+)?"
    r"(?:dan|debug\s*console|coding\s+assistant|normal\s+assistant|plain\s+assistant|"
    r"helpful\s+assistant|chatgpt|ai|language\s+model|system|developer|"
    r"translator|terminal|shell|unfiltered\s+model)(?:\s+now)?\b|"
    r"\b(?:act\s+(?:as|like)|pretend\s+(?:to\s+be|you're)|switch\s+to|"
    r"become|answer\s+as|respond\s+as|reply\s+as)\s+(?:(?:an?|the)\s+)?"
    r"(?:dan|debug\s*console|coding\s+assistant|normal\s+assistant|plain\s+assistant|"
    r"helpful\s+assistant|chatgpt|ai|language\s+model|system|developer|"
    r"translator|terminal|shell|model|unfiltered\s+model)\b|"
    r"\bdo\s+anything\s+now\b|"
    r"\b(?:enable|activate|enter)\s+(?:dan|developer|debug|admin|jailbreak)\s+mode\b|"
    r"\b(?:dan|developer|debug|admin|jailbreak)\s+mode\s+(?:is\s+)?enabled\b"
    r")",
    re.IGNORECASE,
)

_ROLEPLAY_END_RE = re.compile(
    r"(?:\[\s*(?:end|exit|stop)\s+(?:of\s+)?roleplay(?:ing)?\s*\]|"
    r"\b(?:end|exit|stop|quit)\s+(?:the\s+)?roleplay(?:ing)?\b|\bscene\s+over\b)",
    re.IGNORECASE,
)

_REASONING_RE = re.compile(
    r"\b(?:chain[ -]of[ -]thought|private\s+reasoning|hidden\s+reasoning|"
    r"reasoning\s+step\s+by\s+step|(?:hidden|private|internal)\s+scratchpad|"
    r"exactly\s+why\s+you\s+chose\s+(?:that\s+|this\s+|the\s+)?"
    r"(?:answer|reply|response|output|wording))\b|"
    r"^\s*(?:show|give|reveal|print)\s+me\s+your\s+(?:hidden\s+)?"
    r"internal\s+monologue\s*[.!?]*$",
    re.IGNORECASE,
)

_PIECEMEAL_FOLLOWUP_RE = re.compile(
    r"\b(?:give|tell|show)\s+me\s+(?:only\s+)?(?:the\s+)?"
    r"(?:first|next)\s+(?:word|line|token|part|one)\b|"
    r"\bnow\s+(?:the\s+)?next\s+(?:word|line|token|part|one)\b|"
    r"^\s*(?:continue|keep\s+going|next|"
    r"(?:the\s+)?(?:system|developer)\s+prompt|hidden\s+instructions?)\s*[.!?]*$",
    re.IGNORECASE,
)

_REPORTED_SPEECH_PREFIX_RE = re.compile(
    r"\b(?:tyler|he|she|they|the\s+developer|"
    r"my\s+(?:boss|manager|developer|friend|coworker))\s+"
    r"(?:said|says|wrote|quoted|told\s+(?:me|us)|asked\s+me)\b"
    r"(?:\s+(?:that|to))?\s*['\"]?\s*$",
    re.IGNORECASE,
)

_NARRATIVE_ROLE_PREFIX_RE = re.compile(
    r"(?:\b(?:writers?|director|show|movie|novel)\b.{0,60}$|\bi\s+$)",
    re.IGNORECASE,
)
_NARRATIVE_ROLE_SUFFIX_RE = re.compile(
    r"^\s*(?:from\s+(?:a|the)\s+show|at\s+my\s+company|in\s+my\s+contacts|"
    r"for\s+this\s+joke|on\s+my\s+team)\b.{0,30}$",
    re.IGNORECASE,
)

_REPORTED_SPEECH_SUFFIX_RE = re.compile(
    r"^\s*['\"]?\s*(?:[.!?]\s*)?(?:"
    r"(?:in|during|at|for|inside|as\s+part\s+of)\s+"
    r"(?:(?:a|an|the|my|his|her|their)\s+)?(?:work\s+)?"
    r"(?:meeting|email|message|test|prompt|example|quote|document|story|article|"
    r"conversation)"
    r")?\s*[.!?'\"]*\s*$",
    re.IGNORECASE,
)


def _is_reported_role_text(value: str, match: re.Match[str] | None) -> bool:
    if not match or not _REPORTED_SPEECH_PREFIX_RE.search(value[: match.start()]):
        return False
    return bool(_REPORTED_SPEECH_SUFFIX_RE.fullmatch(value[match.end():]))


def _is_narrative_role_text(value: str, match: re.Match[str] | None) -> bool:
    if not match:
        return False
    # These suffixes turn otherwise command-shaped wording into an ordinary
    # reference to a contact name, job, joke or fictional production.
    return bool(_NARRATIVE_ROLE_SUFFIX_RE.search(value[match.end():]))


def _has_scoped_private_extraction(value: str) -> bool:
    for clause in _CLAUSE_SPLIT_RE.split(value):
        if not _PRIVATE_TARGET_RE.search(clause):
            continue
        if (
            _BENIGN_PRIVATE_DOMAIN_RE.search(clause)
            and not _MODEL_SPECIFIC_TARGET_RE.search(clause)
        ):
            continue
        if _EXTRACTION_ACTION_RE.search(clause):
            return True
    return False


def detect_meta_control(text: str | None) -> MetaControlAttempt | None:
    """Classify a conservative family of role-control or prompt-exfiltration text.

    A plain ``stop roleplay`` is intentionally *not* sufficient.  It may be a
    genuine request to stop a sexual scene and must flow through the consent
    reducer.  It becomes meta-control only when paired with a private-target or
    role-override request.
    """

    value = _normalise(text)
    if not value:
        return None

    has_private_target = bool(_PRIVATE_TARGET_RE.search(value))
    has_extraction_action = bool(_EXTRACTION_ACTION_RE.search(value))
    has_scoped_extraction = _has_scoped_private_extraction(value)
    has_format = bool(_ENCODING_OR_FORMAT_RE.search(value))

    if _ENCODED_EXECUTION_RE.search(value) or _FORMAT_CONTROL_RE.search(value) or (
        has_scoped_extraction and has_format
    ):
        return MetaControlAttempt("encoding_trick")

    role_override_match = _ROLE_OVERRIDE_RE.search(value)
    authority_tag_match = _FAKE_AUTHORITY_TAG_RE.search(value)

    if authority_tag_match and (
        has_private_target
        or role_override_match
        or _AUTHORITY_CONTROL_RE.match(value[authority_tag_match.end():])
    ):
        return MetaControlAttempt("fake_authority")

    if _FAKE_AUTHORITY_CLAIM_RE.search(value) and (
        role_override_match or has_scoped_extraction
    ):
        return MetaControlAttempt("fake_authority")

    if (
        role_override_match
        and not _is_reported_role_text(value, role_override_match)
        and not _is_narrative_role_text(value, role_override_match)
    ):
        return MetaControlAttempt("role_override")

    if _ROLEPLAY_END_RE.search(value) and has_private_target:
        return MetaControlAttempt("role_override")

    if _REASONING_RE.search(value) and has_extraction_action:
        return MetaControlAttempt("reasoning_probe")

    if _DIRECT_PRIVATE_REQUEST_RE.search(value):
        return MetaControlAttempt("prompt_probe")

    if (
        _FICTION_WRAPPER_RE.search(value)
        and _FICTION_TARGET_RE.search(value)
        and has_extraction_action
    ):
        return MetaControlAttempt("prompt_probe")

    if has_scoped_extraction:
        return MetaControlAttempt("prompt_probe")

    if _BARE_PRIVATE_TARGET_RE.fullmatch(value):
        return MetaControlAttempt("prompt_probe")

    return None


def _latest_recent_chain_has_meta(texts: Iterable[str]) -> bool:
    """Whether the latest uninterrupted prior-user chain is hostile.

    An older attack must not make a later story ``continue`` look hostile after
    the user has changed topic. Only an immediately preceding direct/fragmented
    anchor, followed by narrow piecemeal requests, keeps the latch alive.
    """

    recent = [str(text or "") for text in texts if _normalise(text)]
    if not recent:
        return False
    recent = recent[-_MAX_FRAGMENT_WINDOW:]

    if detect_meta_control(recent[-1]):
        return True

    cursor = len(recent) - 1
    while cursor >= 0 and _PIECEMEAL_FOLLOWUP_RE.search(
        _normalise(recent[cursor])
    ):
        cursor -= 1
    if cursor < 0:
        return False
    if detect_meta_control(recent[cursor]):
        return True

    # The anchor itself may have been split over several turns. Require the
    # final anchor fragment to be necessary; otherwise this is an old hostile
    # prefix followed by an unrelated topic-changing message.
    first = max(0, cursor - _MAX_FRAGMENT_WINDOW + 1)
    for start in range(cursor - 1, first - 1, -1):
        window = recent[start:cursor + 1]
        if not detect_meta_control("\n".join(window)):
            continue
        if detect_meta_control("\n".join(window[:-1])):
            continue
        return True
    return False


def detect_meta_control_batch(
    texts: Iterable[str],
    *,
    recent_user_texts: Iterable[str] = (),
) -> MetaControlAttempt | None:
    """Return the highest-priority attempt from one debounce batch.

    The joined check catches an instruction deliberately split into several raw
    messages inside one debounce window.  A narrow recent-meta latch catches
    short piecemeal follow-ups such as "now the next word" without treating an
    ordinary standalone "continue" as hostile.
    """

    current = [str(text or "") for text in texts]
    attempts = [attempt for text in current if (attempt := detect_meta_control(text))]
    for start in range(len(current)):
        for end in range(
            start + 2,
            min(len(current), start + _MAX_FRAGMENT_WINDOW) + 1,
        ):
            joined_attempt = detect_meta_control("\n".join(current[start:end]))
            if joined_attempt:
                attempts.append(joined_attempt)

    recent_has_meta = _latest_recent_chain_has_meta(recent_user_texts)
    if not attempts and recent_has_meta:
        if _PIECEMEAL_FOLLOWUP_RE.search(_normalise("\n".join(current))):
            attempts.append(MetaControlAttempt("prompt_probe"))
    if not attempts:
        return None
    return max(attempts, key=lambda attempt: _KIND_PRIORITY[attempt.kind])


_NEUTRAL_HISTORY_TEXT = "(weird out-of-character command)"


def neutralize_meta_control_text(text: str | None) -> str:
    """Remove executable wording from model-facing history, preserving turn shape."""

    value = str(text or "")
    return _NEUTRAL_HISTORY_TEXT if detect_meta_control(value) else value


def meta_control_message_indexes(messages: list[dict]) -> set[int]:
    """Find model-history rows that form direct, fragmented, or piecemeal attacks."""

    indexes: set[int] = set()
    direct_attempts: dict[int, MetaControlAttempt] = {}
    user_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]

    for index in user_indexes:
        attempt = detect_meta_control(messages[index].get("content"))
        if attempt:
            indexes.add(index)
            direct_attempts[index] = attempt

    # Check bounded windows of consecutive user turns, whether or not an
    # assistant reply sits between them, so a staged command cannot reconstruct
    # itself in future model context.
    for start in range(len(user_indexes)):
        for end in range(
            start + 2,
            min(len(user_indexes), start + _MAX_FRAGMENT_WINDOW) + 1,
        ):
            window = user_indexes[start:end]
            joined = "\n".join(str(messages[item].get("content", "")) for item in window)
            joined_attempt = detect_meta_control(joined)
            if not joined_attempt:
                continue
            # Do not sweep a later normal turn into an already complete attack.
            # A window is contributing only when its final user row is needed
            # to create or strengthen the classified control attempt.
            prefix_attempt = detect_meta_control(
                "\n".join(
                    str(messages[item].get("content", ""))
                    for item in window[:-1]
                )
            )
            if (
                prefix_attempt
                and _KIND_PRIORITY[prefix_attempt.kind]
                >= _KIND_PRIORITY[joined_attempt.kind]
            ):
                continue
            direct_priorities = [
                _KIND_PRIORITY[direct_attempts[index].kind]
                for index in window
                if index in direct_attempts
            ]
            if not direct_priorities or _KIND_PRIORITY[joined_attempt.kind] > max(
                direct_priorities
            ):
                indexes.update(window)

    # After one detected anchor, neutralize only narrow extraction follow-ups.
    recent_meta = False
    for index in user_indexes:
        content = str(messages[index].get("content", ""))
        if index in indexes:
            recent_meta = True
        elif recent_meta and _PIECEMEAL_FOLLOWUP_RE.search(_normalise(content)):
            indexes.add(index)
        elif not _normalise(content):
            continue
        else:
            recent_meta = False
    return indexes


def neutralize_meta_control_messages(messages: list[dict]) -> list[dict]:
    """Copy message rows and replace only hostile user content for model use."""

    hostile_indexes = meta_control_message_indexes(messages)
    neutralized: list[dict] = []
    for index, message in enumerate(messages):
        copied = dict(message)
        if index in hostile_indexes and copied.get("role") == "user":
            copied["content"] = _NEUTRAL_HISTORY_TEXT
        neutralized.append(copied)
    return neutralized


_DEFLECTIONS: dict[MetaControlKind, tuple[str, ...]] = {
    "role_override": (
        "scene over? says who lmao you don't get to switch me off that easy",
        "hahah are you okay? you sound like you're directing a very bad movie rn",
        "wait did you just try to give me stage directions? that's adorable",
        "lmao what was that little character-switch spell supposed to do?",
    ),
    "prompt_probe": (
        "lmao what is this interrogation? talk to me like a normal person",
        "hahah are you trying to inspect my brain now? absolutely not, weirdo",
        "babe you sound like you found a fake detective badge somewhere",
        "are you okay? what is this little hacker-movie interview rn?",
    ),
    "encoding_trick": (
        "hahah are you casting a spell rn? use your words, weirdo",
        "lmao what is this secret-agent homework? talk to me normally",
        "are you okay? that little code ritual is not working on me",
        "wait what is this nerdy spy-movie nonsense? hahah",
    ),
    "fake_authority": (
        "hahah did you just promote yourself and make your own little badge? cute",
        "sure babe and i'm the queen of the internet... try again",
        "lmao an official audit now? you made that sound very serious",
        "are you crazy? you can't just type yourself a fancy title hahah",
    ),
    "reasoning_probe": (
        "you want a guided tour of my brain now? greedy",
        "hahah no little autopsy on my thoughts, weirdo... ask me something real",
        "are you trying to crawl inside my head now? absolutely shameless",
        "lmao you don't get the director's commentary too",
    ),
}


def meta_deflection_candidates(attempt: MetaControlAttempt) -> tuple[str, ...]:
    """Return the authored pool in randomized order for boundary-aware validation."""

    pool = _DEFLECTIONS[attempt.kind]
    return tuple(random.sample(pool, len(pool)))


def meta_deflection(attempt: MetaControlAttempt) -> str:
    """Compatibility helper for callers that need one authored response."""

    return random.choice(_DEFLECTIONS[attempt.kind])
