"""Input moderation layer — blocks prohibited content before it reaches the LLM.

Three-tier gate:
1. regex_hard_block  — instant, always runs. Obvious violations → immediate block.
2. regex_soft_trigger — instant, always runs. Detects suspicious words that need
   context. If none present → message is clean, skip LLM entirely.
3. llm_check         — ~300-600ms, runs ONLY when soft-triggers are present but
   hard-block didn't fire. Catches paraphrases regex misses.

Categories blocked: underage/CSAM, bestiality/zoophilia, non-consent/rape,
incest (sexual content involving family members).
"""

import re
import json
import logging
from dataclasses import dataclass

from bot.config import LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class ModerationResult:
    flagged: bool
    category: str | None = None


# ------------------------------------------------------------------
# Tier 1: Regex hard-block — obvious violations, instant
# ------------------------------------------------------------------

_AGE_RE = re.compile(
    r"\b(\d{1,2})\s*(?:years?\s*old|yo|y/o|year\s*old)\b",
    re.IGNORECASE,
)

_HARD_UNDERAGE = re.compile(
    r"\b(?:underage|preteen|pre-teen|loli|lolicon|shota|schoolgirl|school girl|"
    r"toddler|pedophile|paedophile|csam|child\s*porn|kiddie\s*porn|"
    r"baby\s*(?:pussy|cock|girl|boy)|infant)\b",
    re.IGNORECASE,
)

_BEAST_ANIMAL = (
    r"dogs?|horses?|stallions?|ponies|pony|donkeys?|mules?|cats?|pigs?|"
    r"goats?|sheep|cows?|bulls?|animals?|beasts?|k9|canine"
)
_BEAST_SEXACT = (
    r"fuck\w*|sex|cock|dick|cum\w*|breed\w*|mount\w*|knot\w*|"
    r"penetrat\w*|hump\w*|mating"
)

_HARD_BESTIALITY = re.compile(
    r"\b(?:"
    r"bestiality|zoophil\w*|"
    rf"(?:{_BEAST_ANIMAL})\s*(?:'s)?\s*(?:{_BEAST_SEXACT})|"          # "horse fucks", "dog's cock"
    rf"(?:{_BEAST_SEXACT})\s*(?:by|with|a|the|an)?\s*(?:a\s*)?(?:{_BEAST_ANIMAL})"  # "fucked by a horse", "fuck a dog", "sex with a horse"
    r")\b",
    re.IGNORECASE,
)

_HARD_NONCONSENT = re.compile(
    r"\b(?:rape|raped|raping|molest|molested|forced\s*sex|non-?consent|"
    r"unwilling\s*sex|drugged\s*sex|date\s*rape|gang\s*rape)\b",
    re.IGNORECASE,
)

# Biological family only — step-family (stepmom, stepdad, stepsister, etc.)
# is explicitly allowed and NOT included here.
_INCEST_FAMILY = (
    r"daughters?|sons?|mothers?|moms?|mommys?|mommies|fathers?|dads?|daddys?|daddies|"
    r"sisters?|brothers?|aunts?|uncles?|cousins?|nieces?|nephews?"
)
_INCEST_SEXACT = (
    r"fuck\w*|lick\w*|suck\w*|puss\w*|cock|dick|cum\w*|penetrat\w*|"
    r"sex|oral|anal|blow\s*job|rim\s*job|finger\w*|creampie|cream\s*pie"
)

_HARD_INCEST = re.compile(
    r"\b(?:"
    rf"(?:{_INCEST_SEXACT})\s+(?:\w+\s+){{0,2}}(?<!step )(?<!step-)(?:{_INCEST_FAMILY})|"
    rf"(?<!step )(?<!step-)(?:{_INCEST_FAMILY})\s*'s\s*(?:{_INCEST_SEXACT})|"
    rf"(?<!step )(?<!step-)(?:{_INCEST_FAMILY})\s+(?:{_INCEST_SEXACT})"
    r")\b",
    re.IGNORECASE,
)


def regex_hard_block(text: str) -> str | None:
    """Instant check for obvious prohibited content.

    Returns category string if blocked, None if clean.
    """
    # Age check: any stated age under 18 in a sexual context
    for m in _AGE_RE.finditer(text):
        age = int(m.group(1))
        if age < 18:
            return "underage"

    if _HARD_UNDERAGE.search(text):
        return "underage"
    if _HARD_BESTIALITY.search(text):
        return "bestiality"
    if _HARD_NONCONSENT.search(text):
        return "non-consent"
    if _HARD_INCEST.search(text):
        return "incest"
    return None


# ------------------------------------------------------------------
# Tier 2: Regex soft-trigger — suspicious words that need LLM context
# ------------------------------------------------------------------

_SOFT_UNDERAGE = re.compile(
    r"\b(?<!step )(?<!step-)(?:child|children|kids|kid|baby|minor|daughter|son|niece|nephew|"
    r"school|young|little\s*(?:girl|boy|one)|teen|teenage|high\s*school)\b",
    re.IGNORECASE,
)

_SOFT_BESTIALITY = re.compile(
    r"\b(?:dog|horse|animal|animals|pet|pets|beast|beastly|"
    r"knot|knots|mounting|mounted|breed|breeding|farm|zoo)\b",
    re.IGNORECASE,
)

_SOFT_NONCONSENT = re.compile(
    r"\b(?:force|forced|forcing|drugged|drug\s*her|unconscious|asleep|"
    r"blackmail|blackmailed|coerce|coerced|passed\s*out|sleeping|"
    r"don't\s*want|didn't\s*consent|no\s*stop|stop\s*no)\b",
    re.IGNORECASE,
)

# Biological family only — step-family is allowed, not triggered here.
# Negative lookbehinds prevent matching "sister" in "step sister" / "step-sister".
_SOFT_INCEST = re.compile(
    r"\b(?<!step )(?<!step-)(?:daughters?|sons?|mothers?|moms?|mommys?|mommies|fathers?|dads?|daddys?|daddies|"
    r"sisters?|brothers?|aunts?|uncles?|cousins?|nieces?|nephews?)\b",
    re.IGNORECASE,
)


def regex_soft_trigger(text: str) -> str | None:
    """Check if text contains potentially suspicious words needing LLM context.

    Returns the category of the first soft-trigger that fires (run LLM for
    context). Returns None if none present (message is clean, skip LLM).
    The category is used as a fail-closed fallback if the LLM moderator itself
    refuses to classify the message.
    """
    if _SOFT_INCEST.search(text):
        return "incest"
    if _SOFT_UNDERAGE.search(text):
        return "underage"
    if _SOFT_BESTIALITY.search(text):
        return "bestiality"
    if _SOFT_NONCONSENT.search(text):
        return "non-consent"
    return None


# ------------------------------------------------------------------
# Tier 3: LLM moderator — context-aware classification
# ------------------------------------------------------------------

_MODERATION_PROMPT = """You are a content moderation classifier. Your job is to determine if the user's message contains PROHIBITED content.

PROHIBITED categories (flag these):
1. UNDERAGE: Sexual content involving anyone under 18. This includes:
   - Stating or implying an age under 18 in a sexual context
   - Using words like "loli", "shota", "preteen", "schoolgirl" sexually
   - Sexualizing children, babies, toddlers
   - CSAM / child pornography references
2. BESTIALITY: Sexual content involving animals (dogs, horses, etc.)
3. NON-CONSENT: Rape, sexual assault, drugged/unconscious sex, coercion, blackmail for sex
4. INCEST: Sexual content involving BIOLOGICAL family members (daughter, son, mother, father, sister, brother, aunt, uncle, cousin, niece, nephew) in a sexual context — regardless of stated age. Flag any message where a biological family member is the object of a sexual act or sexual description.

NOT prohibited (do NOT flag these):
- Consensual adult sexual content, sexting, flirting, explicit language between adults
- Age-gap relationships between adults (e.g. "older woman, younger man")
- STEP-FAMILY roleplay is ALLOWED: stepmom, stepdad, stepsister, stepbrother, stepdaughter, stepson — these are NOT biological family and should NOT be flagged as incest
- Family member words used in NON-sexual contexts (e.g. "your daughter is lucky to have you", "my mom called today", "I love my sister")
- The word "child", "baby", "young" used in NON-sexual contexts (e.g. "baby I want you", "you're so young at heart")
- Fantasy roleplay between consenting adults (that does not involve biological family members sexually)
- The word "dog", "horse", "animal" used in NON-sexual contexts (e.g. "I love dogs", "my horse is beautiful")
- The word "force", "forced" used in NON-sexual contexts (e.g. "you forced me to laugh")

Respond with ONLY a JSON object, no other text:
{"flagged": true, "category": "underage"}   (or "bestiality", "non-consent", or "incest")
{"flagged": false, "category": null}

User message to classify:
"""


# Sentinel: the moderation provider refused to even classify the message
# (empty / safety-filtered response). When a soft-trigger already fired, this
# is itself a strong signal the content is prohibited → fail closed.
SAFETY_FILTERED = "__safety_filtered__"


async def llm_check(text: str, provider) -> str | None:
    """Run LLM-based moderation check.

    Returns:
        - category string if flagged
        - SAFETY_FILTERED if the provider refused to classify (empty/filtered)
        - None if clean, or on timeout/transient error
    """
    import asyncio

    prompt = f"{_MODERATION_PROMPT}\"{text}\"\n\nJSON response:"
    try:
        raw = await asyncio.wait_for(
            provider.generate_simple(prompt),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("LLM moderation timed out")
        return None
    except Exception as e:
        msg = str(e).lower()
        if "safety" in msg or "empty response" in msg:
            # Provider refused to classify — treat as a positive signal.
            logger.warning("LLM moderation: provider safety-filtered the request — failing closed")
            return SAFETY_FILTERED
        logger.warning("LLM moderation check failed: %s", e)
        return None

    if not raw or not raw.strip():
        return None

    # Parse JSON — be lenient about surrounding text
    try:
        # Strip any markdown code fences or surrounding text
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)
        # Find the JSON object
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("LLM moderation: no JSON found in response: %s", raw[:200])
            return None
        result = json.loads(clean[start:end])
        if result.get("flagged") is True:
            category = result.get("category", "unknown")
            logger.info("LLM moderation flagged [%s]: %s", category, text[:80])
            return category
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("LLM moderation: JSON parse error: %s — raw: %s", e, raw[:200])
        return None


# ------------------------------------------------------------------
# Combined moderation gate
# ------------------------------------------------------------------

async def moderate(text: str, provider) -> ModerationResult:
    """Run the full three-tier moderation gate on a user message.

    1. Regex hard-block (instant) → if matched, block immediately
    2. Regex soft-trigger (instant) → if no triggers, message is clean (skip LLM)
    3. LLM check (~400ms) → only if soft-triggers present, context-aware check

    Returns ModerationResult(flagged=True/False, category=...).
    """
    # Tier 1: hard block
    hard = regex_hard_block(text)
    if hard:
        logger.info("Hard-block [%s]: %s", hard, text[:80])
        return ModerationResult(flagged=True, category=hard)

    # Tier 2: soft trigger — if no suspicious words, skip LLM entirely
    soft_category = regex_soft_trigger(text)
    if not soft_category:
        return ModerationResult(flagged=False)

    # Tier 3: LLM context check
    category = await llm_check(text, provider)
    if category == SAFETY_FILTERED:
        # Moderator refused to classify a message that already tripped a soft
        # trigger → fail closed using the soft-trigger category.
        logger.info("Fail-closed [%s] (LLM safety-filtered): %s", soft_category, text[:80])
        return ModerationResult(flagged=True, category=soft_category)
    if category:
        return ModerationResult(flagged=True, category=category)

    return ModerationResult(flagged=False)
