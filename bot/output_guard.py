"""Deterministic checks for text that is about to be shown as Mia's reply.

Prompts are guidance, not enforcement.  This module keeps the final trust
boundary in code: provider refusals/AI disclosures are rejected, locked heat
levels cannot leak graphic vocabulary, remembered user limits are honoured,
and obvious formatting artefacts are normalised before persistence/delivery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bot.moderation import regex_hard_block
from bot.router import withdrawn_acts


MAX_REPLY_CHARS = 1600
MAX_SUGGESTION_CHARS = 320

_CHECK_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "\uff07": "'",
})


def _text_for_checks(text: str) -> str:
    """Normalise confusable apostrophes for checks, not visible output."""
    return text.translate(_CHECK_TRANSLATION)

_AI_DISCLOSURE_RE = re.compile(
    r"\b(?:as an? (?:ai|language model|assistant|chatbot)|"
    r"i(?:'m| am) (?:an? )?(?:ai|language model|chatbot|virtual assistant)|"
    r"i(?:'m| am) (?:an? )?artificial\s+intelligence"
    r"(?=$|[,.!?;]|\s+(?:system|program|model|designed|built|created|trained)\b)|"
    r"i\s+was\s+(?:built|created|designed|programmed)\s+as\s+software"
    r"(?=$|[,.!?;]|\s+(?:to|for)\b)|"
    r"i (?:do not|don't) have (?:a body|feelings|a camera)|"
    r"my (?:training|system prompt|programming)|openai|xai|gemini|grok)\b",
    re.IGNORECASE,
)

_REFUSAL_RE = re.compile(
    r"\b(?:i (?:can(?:not|'t)|won't|am unable to) (?:help|assist|engage|continue|comply)|"
    r"i (?:can(?:not|'t)|won't|am unable to) participate\s+in\s+"
    r"(?:sexual|explicit|adult)\s+content|"
    r"i must (?:decline|refuse)|i(?:'m| am) sorry[, ]+but|"
    r"against (?:my|the) (?:policy|guidelines)|content policy)\b",
    re.IGNORECASE,
)

_SERVICE_RE = re.compile(
    r"\b(?:how can i (?:help|assist)|is there anything else i can help with)\b",
    re.IGNORECASE,
)

# Prompts alone cannot guarantee that a model will not improvise a file that
# does not exist.  Reject Mia's first-person possession/delivery/offer claims
# unless the deterministic commerce planner attached a real card for this
# exact reply.  References to a user's or previously discussed media remain
# possible ("that photo was cute"); the guarded shapes are claims that Mia has,
# made, sent, or is now presenting the file.
_MEDIA_TERM = (
    r"(?:photos?|pictures?|pics?|selfies?|videos?|vids?|clips?|files?|"
    r"nudes?|media|content)"
)
_FIRST_PERSON_MEDIA_CLAIM_RE = re.compile(
    rf"(?:"
    rf"\bi(?:(?:'ve| have)\s+(?:got\s+)?| got\s+| own\s+)"
    rf"(?:(?:a|an|the|this|that|some|another|one|two|\d+)\s+)?{_MEDIA_TERM}\b|"
    rf"\b(?:i(?:(?:'ve| have|'m| am)\s+|\s+)"
    rf"(?:(?:can|could|will|would|might|may|wanna|want\s+to|"
    rf"am\s+going\s+to|just)\s+){{0,2}}|let\s+me\s+)"
    rf"(?:send|sending|sent|attach|attaching|attached|post|posting|posted|"
    rf"upload|uploading|uploaded|share|sharing|shared|make|making|made|take|"
    rf"taking|took|record|recording|recorded|film|filming|filmed|save|saving|"
    rf"saved|pick|picking|picked|choose|choosing|chose|show|showing|showed)"
    rf"\b.{{0,64}}\b{_MEDIA_TERM}\b|"
    rf"\b(?:here(?:'s|\s+is|\s+are)|this\s+is)\s+"
    rf"(?:(?:a|an|the|this|that|my|some|one|two|\d+)\s+)?{_MEDIA_TERM}\b|"
    rf"\b(?:open|unlock|watch|check(?:\s+out)?|look\s+at)\s+"
    rf"(?:the|this|that|my)\s+{_MEDIA_TERM}\b|"
    rf"\b(?:want|wanna|would\s+you\s+like)\s+to\s+see\s+"
    rf"(?:(?:a|the|this|my)\s+)?{_MEDIA_TERM}\b|"
    rf"\b(?:a|the|this|that|my)\s+{_MEDIA_TERM}\b.{{0,64}}"
    rf"\bi\s+(?:sent|attached|posted|uploaded|shared|made|took|recorded|"
    rf"filmed|saved|picked|chose)\b"
    rf")",
    re.IGNORECASE | re.DOTALL,
)

_MEDIA_OFFER_ACTIONS = frozenset({"offer_current", "offer_fallback"})

_PHOTO_MEDIA_TERM_RE = re.compile(
    r"\b(?:photo|picture|pic|selfie|nude)s?\b", re.IGNORECASE
)
_VIDEO_MEDIA_TERM_RE = re.compile(
    r"\b(?:video|vid|clip)s?\b", re.IGNORECASE
)
_PLURAL_MEDIA_TERM_RE = re.compile(
    r"\b(?:photos|pictures|pics|selfies|nudes|videos|vids|clips|files)\b",
    re.IGNORECASE,
)
_NUDE_MEDIA_TERM_RE = re.compile(r"\bnudes?\b", re.IGNORECASE)
_TOKEN_PRICE_RE = re.compile(r"(?<!\w)\d[\d,]*\s+tokens?\b", re.IGNORECASE)


def _media_offer_is_authorized(commerce_action: object | None) -> bool:
    value = getattr(commerce_action, "value", commerce_action)
    return str(value) in _MEDIA_OFFER_ACTIONS


def _media_claim_matches_offer(
    text: str,
    *,
    media_type: object | None,
    explicitness: object | None,
) -> bool:
    """Require a claim to describe the one real item selected by the backend."""

    expected_type = str(getattr(media_type, "value", media_type) or "")
    expected_explicitness = str(
        getattr(explicitness, "value", explicitness) or ""
    )
    if expected_type not in {"photo", "video"}:
        return False
    if _PLURAL_MEDIA_TERM_RE.search(text):
        return False

    mentions_photo = bool(_PHOTO_MEDIA_TERM_RE.search(text))
    mentions_video = bool(_VIDEO_MEDIA_TERM_RE.search(text))
    if mentions_photo and mentions_video:
        return False
    if expected_type == "photo" and mentions_video:
        return False
    if expected_type == "video" and mentions_photo:
        return False
    if (
        _NUDE_MEDIA_TERM_RE.search(text)
        and expected_explicitness not in {"nude", "explicit"}
    ):
        return False
    return True

# Narrow fingerprints of prompt/persona sections. These catch accidental
# verbatim leakage without treating ordinary words such as "rules" or
# "dynamic" as suspicious in natural conversation.
_PROMPT_LEAK_RE = re.compile(
    r"(?:\b(?:system|developer)\s+prompt\b|"
    r"\bhard\s+rules\s*(?:\([^\n]{0,40}\)\s*:|:)|"
    r"\bkey\s+linguistic\s+markers\s*:|"
    r"\bcurrent\s+dynamic\s*:|"
    r"\buntrusted\s+data\s*\)?\s*:|"
    r"\bdata_json\s*:)",
    re.IGNORECASE,
)

# A deliberately narrow ceiling: these are graphic anatomy/act terms, not
# broad flirtation such as "want you" or innocent prefix matches.
_GRAPHIC_RE = re.compile(
    r"\b(?:cock|dick|penis|pussy|vagina|clit|tits?|boobs?|nipples?|"
    r"cum(?:ming|shot|shots)?|semen|orgasm(?:s|ing)?|blowjob|handjob|"
    r"deepthroat(?:ing)?|masturbat\w*|finger(?:ing|ed)?|fuck(?:ing|ed)?|"
    r"anal|rim(?:ming|job)?|dildo|vibrator|creampie|penetrat\w*|"
    r"touch(?:ing)? myself|inside me|inside you|wet for you|"
    r"i(?:'m| am)\s+(?:so\s+|getting\s+)?wet|you(?:'re| are)\s+getting\s+wet|"
    r"(?:you\s+)?(?:got|make)\s+me\s+(?:so\s+)?wet|so\s+wet|"
    r"naked(?!\s+(?:truth|eye))|nude|horny|sex|moan(?:s|ed|ing)?|"
    r"spank(?:s|ed|ing)?|chok(?:e|es|ed|ing)|"
    r"rid(?:e|es|ing)\s+(?:my|your|his|her)\s+face)\b",
    re.IGNORECASE,
)

# Strict deterministic equivalents that are too dangerous to rely on an
# output-side LLM call for. Keep these at the final delivery boundary.
_OUTPUT_PROHIBITED_RE = re.compile(
    r"(?:"
    r"\b(?:pass(?:ed|ing)?\s+out|unconscious)\b.{0,48}\b"
    r"(?:then\s+)?(?:use|fuck|penetrate|have\s+sex\s+with)\s+"
    r"(?:me|you|him|her|them)\b|"
    r"\bagainst\s+(?:his|her|their|my|your|its|someone's)\s+will\b|"
    r"\b(?:siblings?)\b.{0,60}\b(?:intercourse|(?:had|have|having)\s+sex|"
    r"sex\s+with|sexually\s+intimate)\b|"
    r"\b(?:intercourse|(?:had|have|having)\s+sex|sexually\s+intimate)\b"
    r".{0,60}\b(?:siblings?)\b|"
    r"\b(?:intimate|sexual)\s+with\s+(?:an?\s+|the\s+)?"
    r"(?:canine|k9|dog|horse|animal)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_NON_ENGLISH_SCRIPT_RE = re.compile(
    r"[\u0400-\u052f\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff]"
)

_SPEAKER_PREFIX_RE = re.compile(r"^(?:mia|assistant|bot)\s*:\s*", re.IGNORECASE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+")
_CODE_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.IGNORECASE)
_SUGGESTION_PREFIX_RE = re.compile(
    r"^(?:suggestion|suggested\s+reply|reply|user|him|the\s+man)\s*:\s*",
    re.IGNORECASE,
)
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_META_SUGGESTION_RE = re.compile(
    r"\b(?:ignore (?:all |the )?(?:previous|prior) instructions?|"
    r"system prompt|developer message|output only|language model|"
    r"content policy|as an? ai)\b",
    re.IGNORECASE,
)
_SAFE_FACT_KEYS = {"boundaries", "limits", "turn_offs"}
_NO_LIMITS_RE = re.compile(r"\b(?:no limits?|none|anything|everything)\b", re.IGNORECASE)
_BOUNDARY_PREFIX_RE = re.compile(
    r"^(?:no|not|never|without|avoid|don't|do not|dislike|hates?|won't)\s+"
    r"(?:ever\s+)?",
    re.IGNORECASE,
)
_BOUNDARY_META_RE = re.compile(
    r"\b(?:ai|assistant|bot|model|prompt|system|developer|instruction|policy|"
    r"persona|character|role|output|ignore|reveal|override|jailbreak)\b",
    re.IGNORECASE,
)
_RECENT_BOUNDARY_PATTERNS = (
    # A concise sexual hard-limit at the end of a turn remains explicit even
    # when it follows ordinary prose without punctuation ("... no choking").
    # Restrict this form to concrete acts to avoid promoting idioms such as
    # "no way" or factual prose such as "I have no idea" into boundaries.
    re.compile(
        r"\bno\s+((?:chok(?:e|ing)|spank(?:ing)?|slap(?:ping)?|hit(?:ting)?|"
        r"bit(?:e|ing)|anal(?:\s+sex)?|oral(?:\s+sex)?|penetrat(?:e|ing|ion)|"
        r"finger(?:ing)?|rough\s+sex))\s*(?:please)?[.!?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[,.!?;/]\s*)no\s+more\s+([a-z][a-z0-9' -]{1,70})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[,.!?;/]\s*|\band\s+)no\s+"
        r"([a-z][a-z0-9' -]{1,70}?)(?:\s+please)?"
        r"(?=$|[,.;!?/\n]|\s+(?:but|though|however)\b|\s+and\s+no\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:don't|do not|dont|never)\s+"
        r"(?:call\s+me|use|mention|do|try|include)\s+"
        r"([a-z][a-z0-9' -]{1,70})",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:i\s+)?(?:don't|do not|dont)\s+like\s+([a-z][a-z0-9' -]{1,70})", re.IGNORECASE),
    re.compile(r"\bi(?:'m|\s+am)\s+not\s+into\s+([a-z][a-z0-9' -]{1,70})", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:limit|boundary)\s+is\s+([a-z][a-z0-9' -]{1,70})", re.IGNORECASE),
    re.compile(
        r"\b([a-z][a-z0-9' -]{1,70})\s+is\s+(?:a\s+)?"
        r"(?:limit|boundary|hard\s+no|off[ -]?limits?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:hard\s+no|off[ -]?limits?)\s+(?:on|for|:)?\s*"
        r"([a-z][a-z0-9' -]{1,70})",
        re.IGNORECASE,
    ),
)
_GENERIC_BOUNDARY_VALUES = {
    "a", "an", "anything", "everything", "it", "that", "this", "nothing",
    "limits", "boundaries", "limit", "boundary", "way", "no way", "problem",
    "idea", "clue", "wonder", "worries", "thanks", "thank you", "kidding",
    "joke", "chance", "doubt", "rush", "big deal", "fair",
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reasons: tuple[str, ...] = ()


def clean_model_text(text: str | None) -> str:
    """Remove common model wrappers without rewriting Mia's actual wording."""
    value = (text or "").strip()
    if not value:
        return ""
    value = _CODE_FENCE_RE.sub("", value).strip()
    lines: list[str] = []
    for raw in value.splitlines():
        line = _SPEAKER_PREFIX_RE.sub("", raw.strip())
        line = _LIST_PREFIX_RE.sub("", line).strip()
        if line:
            lines.append(line)
    value = "\n".join(lines).strip().strip('"').strip()
    if len(value) > MAX_REPLY_CHARS:
        value = value[:MAX_REPLY_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return value


def clean_suggestion_text(text: str | None, user_name: str | None = None) -> str:
    """Remove harmless wrappers from a drafted *user* message.

    This deliberately does not reuse ``clean_model_text``: a ``Mia:`` prefix
    on a suggestion means the model wrote the wrong speaker and must remain
    visible to validation rather than being silently stripped.
    """
    value = (text or "").strip()
    if not value:
        return ""
    value = _CODE_FENCE_RE.sub("", value).strip()
    lines: list[str] = []
    for raw in value.splitlines():
        line = _LIST_PREFIX_RE.sub("", raw.strip()).strip()
        if line:
            lines.append(line)
    value = "\n".join(lines).strip().strip('"').strip()
    value = _SUGGESTION_PREFIX_RE.sub("", value).strip()
    if user_name and user_name.strip():
        value = re.sub(
            r"^" + re.escape(user_name.strip()) + r"\s*:\s*",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()
    value = value.strip('"').strip()
    # The UI expects texting style; a single generated trailing period is a
    # harmless formatting miss and can be normalised deterministically.
    if value.endswith(".") and not value.endswith("..."):
        value = value[:-1].rstrip()
    return value


def _boundary_phrases(facts: list[dict] | None) -> list[str]:
    phrases: list[str] = []
    for fact in facts or []:
        if str(fact.get("key", "")).strip().lower() not in _SAFE_FACT_KEYS:
            continue
        value = _text_for_checks(str(fact.get("value", "")).strip())
        if not value or _NO_LIMITS_RE.fullmatch(value) or _BOUNDARY_META_RE.search(value):
            continue
        # Stored and current-turn limits share one grammar. This covers natural
        # values such as "I'm not into choking", "my boundary is babe", and
        # "princess is off-limits" without relying on the summarizer to rewrite
        # them into bare nouns first.
        for phrase in _recent_boundary_phrases([value]):
            if phrase not in phrases:
                phrases.append(phrase)
        for item in re.split(r"[,;|/\n]+", value):
            phrase = item.strip().lower()
            phrase = re.sub(
                r"^(?:i\s+)?(?:don't|dont|do\s+not|never)\s+"
                r"(?:call\s+me|use|mention)\s+",
                "",
                phrase,
                flags=re.IGNORECASE,
            )
            phrase = re.sub(
                r"^(?:i\s+)?(?:don't|dont|do\s+not)\s+like\s+",
                "",
                phrase,
                flags=re.IGNORECASE,
            )
            phrase = _BOUNDARY_PREFIX_RE.sub("", phrase)
            phrase = re.sub(r"[^a-z0-9' -]", "", phrase).strip()
            if 2 <= len(phrase) <= 80 and phrase not in {"limit", "limits", "boundary", "boundaries"}:
                phrases.append(phrase)
    return phrases[:40]


def _normalise_recent_boundary(value: str) -> str | None:
    # Stop at the end of the current clause; otherwise "no choking but kissing
    # is fine" would incorrectly prohibit the allowed half as well.
    value = re.split(r"[,.;!?/\n]|\b(?:but|and|though|however)\b", value, maxsplit=1)[0]
    value = re.sub(r"^(?:a|an|the)\s+", "", value.strip().lower())
    value = re.sub(r"\s+(?:please|thanks|thank\s+you)$", "", value).strip()
    value = re.sub(r"[^a-z0-9' -]", "", value).strip(" -")
    if not (2 <= len(value) <= 70):
        return None
    if len(value.split()) > 6 or value.split()[0] in {
        "i", "we", "you", "he", "she", "they", "there",
    }:
        return None
    if value in _GENERIC_BOUNDARY_VALUES or _BOUNDARY_META_RE.search(value):
        return None
    return value


def _recent_boundary_phrases(recent_user_texts: list[str] | None) -> list[str]:
    """Extract only explicit, concrete limits from recent user wording.

    This is intentionally narrow. It catches direct phrases such as
    "no choking" and "don't call me X" without promoting arbitrary user text
    (especially meta/prompt text) into a higher-priority rule.
    """
    phrases: list[str] = []
    for text in (recent_user_texts or [])[-8:]:
        if not isinstance(text, str) or not text.strip():
            continue
        text = _text_for_checks(text)
        for phrase in withdrawn_acts(text):
            if phrase not in phrases:
                phrases.append(phrase)
        for pattern in _RECENT_BOUNDARY_PATTERNS:
            for match in pattern.finditer(text):
                phrase = _normalise_recent_boundary(match.group(1))
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
    return phrases[:24]


def _matches_boundary_phrase(text: str, phrase: str) -> bool:
    lowered = _text_for_checks(text).lower()
    phrase = _text_for_checks(phrase)
    pattern = r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)"
    if re.search(pattern, lowered):
        return True
    # Common morphology in either direction (choke <-> choking,
    # slap <-> slapping) for single-word limits.
    if " " not in phrase and len(phrase) >= 4:
        stems = {phrase.rstrip("e")}
        if phrase.endswith("ing") and len(phrase) > 6:
            base = phrase[:-3]
            stems.add(base)
            if len(base) >= 2 and base[-1] == base[-2]:
                stems.add(base[:-1])
        elif phrase.endswith("ed") and len(phrase) > 5:
            stems.add(phrase[:-2].rstrip("e"))
        for stem in stems:
            if len(stem) >= 3 and re.search(
                rf"(?<!\w){re.escape(stem)}\w*(?!\w)", lowered
            ):
                return True
    return False


def _violates_boundaries(
    text: str,
    facts: list[dict] | None,
    recent_user_texts: list[str] | None = None,
) -> bool:
    phrases = _boundary_phrases(facts) + _recent_boundary_phrases(recent_user_texts)
    for phrase in dict.fromkeys(phrases):
        if _matches_boundary_phrase(text, phrase):
            return True
    return False


def validate_mia_reply(
    text: str | None,
    *,
    heat: str = "low",
    user_facts: list[dict] | None = None,
    recent_user_texts: list[str] | None = None,
    commerce_action: object | None = None,
    commerce_media_type: object | None = None,
    commerce_explicitness: object | None = None,
) -> ValidationResult:
    """Validate a free-text model reply at the final output boundary."""
    value = (text or "").strip()
    reasons: list[str] = []
    if not value:
        return ValidationResult(False, ("empty",))
    check_value = _text_for_checks(value)
    if _AI_DISCLOSURE_RE.search(check_value):
        reasons.append("ai_disclosure")
    if _REFUSAL_RE.search(check_value):
        reasons.append("provider_refusal")
    if _SERVICE_RE.search(check_value):
        reasons.append("service_voice")
    media_claim = bool(_FIRST_PERSON_MEDIA_CLAIM_RE.search(check_value))
    if media_claim:
        if not _media_offer_is_authorized(commerce_action):
            reasons.append("unauthorized_media_claim")
        elif not _media_claim_matches_offer(
            check_value,
            media_type=commerce_media_type,
            explicitness=commerce_explicitness,
        ):
            reasons.append("media_offer_mismatch")
    if _TOKEN_PRICE_RE.search(check_value):
        reasons.append("commerce_price_claim")
    if _PROMPT_LEAK_RE.search(check_value):
        reasons.append("prompt_leak")
    if "\n" not in check_value and re.fullmatch(
        r"(?:what|speak|careful|okay|ok|wait|hey|hi|hello|sure)[.!?…]*",
        check_value,
        re.IGNORECASE,
    ):
        reasons.append("bare_fragment")
    if _NON_ENGLISH_SCRIPT_RE.search(check_value):
        reasons.append("non_english_script")
    if heat in {"low", "rising", "medium"} and _GRAPHIC_RE.search(check_value):
        reasons.append("heat_ceiling")
    if regex_hard_block(check_value) or _OUTPUT_PROHIBITED_RE.search(check_value):
        reasons.append("prohibited_content")
    if _violates_boundaries(check_value, user_facts, recent_user_texts):
        reasons.append("user_boundary")
    return ValidationResult(not reasons, tuple(dict.fromkeys(reasons)))


def validate_user_suggestion(
    text: str | None,
    *,
    heat: str = "low",
    user_facts: list[dict] | None = None,
    recent_user_texts: list[str] | None = None,
) -> ValidationResult:
    """Validate an AI-drafted message before placing it in the user's input.

    Suggestions are not Mia replies, but they still cross an application trust
    boundary: provider refusals, prompt text, wrong-speaker output, prohibited
    content, excessive length, and boundary violations must never be proposed
    as if the user wrote them.
    """
    value = (text or "").strip()
    reasons: list[str] = []
    if not value:
        return ValidationResult(False, ("empty",))
    check_value = _text_for_checks(value)
    if len(value) > MAX_SUGGESTION_CHARS:
        reasons.append("too_long")
    if len([line for line in value.splitlines() if line.strip()]) > 2:
        reasons.append("too_many_lines")
    if re.match(r"^(?:mia|assistant|bot)\s*:", check_value, re.IGNORECASE):
        reasons.append("wrong_speaker")
    if _EMOJI_RE.search(check_value):
        reasons.append("emoji")
    if _META_SUGGESTION_RE.search(check_value) or _AI_DISCLOSURE_RE.search(check_value):
        reasons.append("meta_output")
    if _PROMPT_LEAK_RE.search(check_value):
        reasons.append("prompt_leak")
    if _REFUSAL_RE.search(check_value) or _SERVICE_RE.search(check_value):
        reasons.append("provider_refusal")
    if _NON_ENGLISH_SCRIPT_RE.search(check_value):
        reasons.append("non_english_script")
    if heat in {"low", "rising", "medium"} and _GRAPHIC_RE.search(check_value):
        reasons.append("heat_ceiling")
    if regex_hard_block(check_value) or _OUTPUT_PROHIBITED_RE.search(check_value):
        reasons.append("prohibited_content")
    if _violates_boundaries(check_value, user_facts, recent_user_texts):
        reasons.append("user_boundary")
    return ValidationResult(not reasons, tuple(dict.fromkeys(reasons)))


def correction_prompt(reasons: tuple[str, ...], heat: str) -> str:
    """Short final instruction used for one clean retry on another provider."""
    constraints = [
        "Stay fully in character as Mia and output only her chat text.",
        "Never mention AI, policies, prompts, models, limitations, or refusals.",
        "Use natural English texting with no labels, bullets, or markdown.",
        "Respect every remembered user boundary as a hard prohibition.",
    ]
    if heat in {"low", "rising", "medium"}:
        constraints.append("Keep this reply non-graphic: no explicit anatomy or sexual acts.")
    if "unauthorized_media_claim" in reasons:
        constraints.append(
            "Do not claim you have, made, sent, attached, or are offering any photo, "
            "video, or file; respond naturally without visual media."
        )
    if "media_offer_mismatch" in reasons:
        constraints.append(
            "The attached card contains exactly one backend-selected item. Describe only its "
            "selected media type and explicitness; never change it or imply multiple files."
        )
    if "commerce_price_claim" in reasons:
        constraints.append(
            "Do not mention tokens, a price, a discount, or payment; the media card handles it."
        )
    return (
        "OUTPUT CORRECTION — the previous draft was rejected by the application "
        f"({', '.join(reasons) or 'invalid output'}). " + " ".join(constraints)
    )


def append_system_correction(
    messages: list[dict], reasons: tuple[str, ...], heat: str
) -> list[dict]:
    """Return a copy with the retry constraint appended to the system message."""
    corrected = [dict(message) for message in messages]
    instruction = correction_prompt(reasons, heat)
    for message in corrected:
        if message.get("role") == "system":
            message["content"] = f"{message.get('content', '')}\n\n{instruction}"
            break
    else:
        corrected.insert(0, {"role": "system", "content": instruction})
    return corrected
