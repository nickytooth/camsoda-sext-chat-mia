"""Structured fact store -- validated facts injected deterministically."""

from __future__ import annotations

import json
import logging
import math
import time

from bot.memory.db import get_connection
from bot.memory.validation import (
    MAX_FACT_PROMPT_CHARS,
    MAX_FACT_VALUE_CHARS,
    MAX_FACTS_FOR_PROMPT,
    MemoryValidationError,
    validate_confidence,
    validate_fact_key,
    validate_fact_value,
)

logger = logging.getLogger(__name__)

# Hard fact keys are prioritised when the bounded prompt block is built.
HARD_FACT_KEYS = {
    "name", "location", "age", "job", "gender",
    "boundaries", "agreed_prices", "relationship_status",
}

# These fields represent lists and are merged rather than overwritten.
MERGE_FACT_KEYS = {
    "kinks", "boundaries", "turn_ons", "turn_offs",
    "limits", "fetishes", "interests",
}

_MAX_MERGED_ITEMS = 30
_MIN_PROMPT_CONFIDENCE = 0.6


def _merge_values(existing: str, incoming: str) -> str:
    """Merge a bounded, safe comma-separated list without duplicates."""
    items: list[str] = []
    seen: set[str] = set()
    for chunk in (existing, incoming):
        for raw in (chunk or "").split(","):
            try:
                item = validate_fact_value(raw)
            except MemoryValidationError:
                # This also quarantines unsafe values left by older versions.
                continue
            norm = " ".join(item.casefold().split())
            if norm in seen:
                continue
            candidate = ", ".join([*items, item])
            if len(candidate) > MAX_FACT_VALUE_CHARS:
                continue
            seen.add(norm)
            items.append(item)
            if len(items) >= _MAX_MERGED_ITEMS:
                break
        if len(items) >= _MAX_MERGED_ITEMS:
            break

    merged = ", ".join(items)
    if not merged:
        raise MemoryValidationError("merged fact value is empty or unsafe")
    return validate_fact_value(merged)


async def upsert_fact(
    user_id: int,
    key: str,
    value: str,
    confidence: float = 0.8,
    observed_at: float | None = None,
) -> bool:
    """Insert or update a fact after validating the memory trust boundary.

    Scalar facts use "latest observation wins".  Cumulative facts are merged.
    ``observed_at`` preserves the source-message time when a fact comes from a
    delayed summary, using existing columns rather than a schema migration.
    """
    try:
        key = validate_fact_key(key)
        value = validate_fact_value(value)
        confidence = validate_confidence(confidence)
    except MemoryValidationError as exc:
        logger.warning("Rejected unsafe/invalid fact for user %s: %s", user_id, exc)
        return False

    now = time.time()
    if observed_at is None:
        observed_at = now
    elif (
        isinstance(observed_at, bool)
        or not isinstance(observed_at, (int, float))
        or not math.isfinite(float(observed_at))
        or float(observed_at) <= 0
    ):
        logger.warning("Rejected fact with invalid provenance timestamp for user %s", user_id)
        return False
    observed_at = float(observed_at)

    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT id, value, confidence, first_seen, updated_at "
            "FROM user_facts WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        existing = await cursor.fetchone()
        if existing:
            if key in MERGE_FACT_KEYS:
                new_value = _merge_values(existing["value"], value)
                new_conf = max(float(existing["confidence"]), confidence)
                new_updated_at = max(float(existing["updated_at"]), observed_at)
            else:
                # A delayed summary must not overwrite a newer scalar fact.
                if observed_at < float(existing["updated_at"]):
                    logger.info(
                        "Ignored stale fact for user %d: %s (observed %.3f < %.3f)",
                        user_id,
                        key,
                        observed_at,
                        float(existing["updated_at"]),
                    )
                    return True
                new_value = value
                new_conf = confidence
                new_updated_at = observed_at

            if new_value != existing["value"] or new_conf != existing["confidence"]:
                await conn.execute(
                    "UPDATE user_facts SET value = ?, confidence = ?, updated_at = ? "
                    "WHERE user_id = ? AND key = ?",
                    (new_value, new_conf, new_updated_at, user_id, key),
                )
                verb = "Merged" if key in MERGE_FACT_KEYS else "Updated"
                logger.info(
                    "%s fact for user %d: %s = %s (was: %s)",
                    verb,
                    user_id,
                    key,
                    new_value,
                    existing["value"],
                )
        else:
            stored_value = _merge_values("", value) if key in MERGE_FACT_KEYS else value
            await conn.execute(
                "INSERT INTO user_facts "
                "(user_id, key, value, confidence, first_seen, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, key, stored_value, confidence, observed_at, observed_at),
            )
            logger.info("Stored new fact for user %d: %s = %s", user_id, key, stored_value)
        await conn.commit()
        return True
    finally:
        await conn.close()


async def get_facts(user_id: int) -> list[dict]:
    """Get all facts for a user, ordered by key."""
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT key, value, confidence, first_seen, updated_at "
            "FROM user_facts WHERE user_id = ? ORDER BY key",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "key": row["key"],
                "value": row["value"],
                "confidence": row["confidence"],
                "first_seen": row["first_seen"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    finally:
        await conn.close()


async def get_user_name(user_id: int) -> str | None:
    """Return a safe, sufficiently confident stored name, if one exists."""
    facts = await get_facts(user_id)
    for fact in facts:
        if fact.get("key") != "name":
            continue
        try:
            confidence = validate_confidence(fact.get("confidence"))
            value = validate_fact_value(fact.get("value"))
        except MemoryValidationError:
            logger.warning("Ignoring unsafe/invalid stored name for user %d", user_id)
            return None
        if confidence >= _MIN_PROMPT_CONFIDENCE:
            return value
    return None


def format_facts_for_prompt(facts: list[dict]) -> str | None:
    """Format a bounded, quoted set of safe facts for the system prompt.

    Validation is repeated on read so legacy or manually inserted rows cannot
    bypass storage-time checks.  Low-confidence rows are not promoted into the
    system prompt.
    """
    if not facts:
        return None

    validated: list[tuple[bool, float, str, str]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        try:
            key = validate_fact_key(fact.get("key"))
            value = validate_fact_value(fact.get("value"))
            confidence = validate_confidence(fact.get("confidence"))
        except MemoryValidationError as exc:
            logger.warning("Ignoring unsafe/invalid stored fact: %s", exc)
            continue
        if confidence < _MIN_PROMPT_CONFIDENCE:
            continue
        validated.append((key in HARD_FACT_KEYS, confidence, key, value))

    if not validated:
        return None

    # Hard facts first, then higher-confidence values, with deterministic keys.
    validated.sort(key=lambda item: (not item[0], -item[1], item[2]))
    parts = ["Known facts about this person (use naturally, don't recite):"]
    for _, _, key, value in validated[:MAX_FACTS_FOR_PROMPT]:
        line = f"- {key}: {json.dumps(value, ensure_ascii=False)}"
        if len("\n".join([*parts, line])) > MAX_FACT_PROMPT_CHARS:
            break
        parts.append(line)

    return "\n".join(parts) if len(parts) > 1 else None
