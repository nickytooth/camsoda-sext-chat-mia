"""Structured fact store — hard facts injected deterministically every turn."""

import time
import logging
from bot.memory.db import get_connection

logger = logging.getLogger(__name__)

# Hard fact keys that are always injected into the prompt
HARD_FACT_KEYS = {
    "name", "location", "age", "job", "gender",
    "boundaries", "agreed_prices", "relationship_status",
}

# Soft, cumulative keys: these describe a growing LIST (e.g. kinks), not a
# single scalar. They are accumulated/merged across summaries instead of being
# overwritten — otherwise each summary batch wipes whatever the previous batch
# learned (a kink list should never shrink just because the latest 10 messages
# didn't re-mention an earlier kink). Scalar keys (name, age, location, ...)
# keep "latest value wins" semantics so conflicts resolve to the newest value.
MERGE_FACT_KEYS = {
    "kinks", "boundaries", "turn_ons", "turn_offs",
    "limits", "fetishes", "interests",
}

# Cap merged lists so the always-injected facts block can't bloat the prompt.
_MAX_MERGED_ITEMS = 30


def _merge_values(existing: str, incoming: str) -> str:
    """Union of comma-separated items, case-insensitively de-duplicated,
    preserving first-seen order and original casing. Capped to avoid bloat."""
    items: list[str] = []
    seen: set[str] = set()
    for chunk in (existing, incoming):
        for raw in (chunk or "").split(","):
            item = raw.strip()
            if not item:
                continue
            norm = " ".join(item.lower().split())
            if norm in seen:
                continue
            seen.add(norm)
            items.append(item)
    return ", ".join(items[:_MAX_MERGED_ITEMS])


async def upsert_fact(user_id: int, key: str, value: str, confidence: float = 0.8) -> None:
    """Insert or update a fact.

    Scalar keys use "latest value wins". Cumulative keys in MERGE_FACT_KEYS
    (e.g. kinks) are merged with the existing value so the list only grows.
    """
    now = time.time()
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT id, value, confidence FROM user_facts WHERE user_id = ? AND key = ?",
            (user_id, key),
        )
        existing = await cursor.fetchone()
        if existing:
            if key in MERGE_FACT_KEYS:
                new_value = _merge_values(existing["value"], value)
                new_conf = max(existing["confidence"], confidence)
            else:
                new_value = value
                new_conf = confidence
            if new_value != existing["value"] or new_conf != existing["confidence"]:
                await conn.execute(
                    "UPDATE user_facts SET value = ?, confidence = ?, updated_at = ? "
                    "WHERE user_id = ? AND key = ?",
                    (new_value, new_conf, now, user_id, key),
                )
                verb = "Merged" if key in MERGE_FACT_KEYS else "Updated"
                logger.info("%s fact for user %d: %s = %s (was: %s)", verb, user_id, key, new_value, existing["value"])
        else:
            # De-dup the incoming value too (LLM may repeat items in one batch).
            stored_value = _merge_values("", value) if key in MERGE_FACT_KEYS else value
            await conn.execute(
                "INSERT INTO user_facts (user_id, key, value, confidence, first_seen, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, key, stored_value, confidence, now, now),
            )
            logger.info("Stored new fact for user %d: %s = %s", user_id, key, stored_value)
        await conn.commit()
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
    """Extract the user's name from stored facts, or None if not set."""
    facts = await get_facts(user_id)
    for f in facts:
        if f["key"] == "name":
            return f["value"]
    return None


def format_facts_for_prompt(facts: list[dict]) -> str | None:
    """Format facts into a compact prompt section. Returns None if no facts."""
    if not facts:
        return None

    hard = []
    soft = []
    for f in facts:
        line = f"{f['key']}: {f['value']}"
        if f["key"] in HARD_FACT_KEYS:
            hard.append(line)
        else:
            soft.append(line)

    parts = ["Known facts about this person (use naturally, don't recite):"]
    if hard:
        parts.extend(f"- {h}" for h in hard)
    if soft:
        parts.extend(f"- {s}" for s in soft)
    return "\n".join(parts)
