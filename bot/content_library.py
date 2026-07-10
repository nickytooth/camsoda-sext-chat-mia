"""
Authored fantasy/story libraries + per-user "already shared" tracking.

The libraries (library/*.yaml) are the same for everyone. Which items Mia
has already told a given user is tracked in Postgres (shared_content), so the
"Hear a fantasy" / "Hear a story" cards never repeat until the pool is exhausted.
"""

import os
import time
import random
import logging
import yaml

from bot.config import FANTASIES_FILE, STORIES_FILE, TYLER_ARC_FILE
from bot.memory.db import get_connection

logger = logging.getLogger(__name__)

# kind -> (file path, top-level yaml key)
_SOURCES = {
    "fantasy": (FANTASIES_FILE, "fantasies"),
    "story": (STORIES_FILE, "stories"),
}

_cache: dict[str, list[dict]] = {}
_cache_mtime: dict[str, float] = {}


def _load(kind: str) -> list[dict]:
    """Load and cache a library file. Returns a list of {id, text, tags}.
    Cache is invalidated when the file's mtime changes."""
    src = _SOURCES.get(kind)
    if not src:
        return []
    path, key = src
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    if kind in _cache and _cache_mtime.get(kind) == mtime:
        return _cache[kind]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = [it for it in (data.get(key) or []) if it.get("id") and it.get("text")]
    except Exception as e:
        logger.warning("Failed to load %s library from %s: %s", kind, path, e)
        items = []
    _cache[kind] = items
    _cache_mtime[kind] = mtime
    return items


def get_examples(kind: str, n: int = 2) -> list[str]:
    """Return up to `n` authored item texts to use as STYLE few-shot examples.

    Used by dynamic generation (e.g. location/LTM-aware fantasies) so freshly
    generated content matches the authored length, rhythm and tone without being
    sent verbatim.
    """
    items = _load(kind)
    return [it["text"].strip() for it in items if it.get("text")][:n]


def library_size(kind: str) -> int:
    """How many authored items exist for this kind. Lets callers tell an
    EXHAUSTED rotation (all shared) apart from an EMPTY/missing library file."""
    return len(_load(kind))


# ---------------------------------------------------------------------------
# Tyler arc — slow background storyline, advanced by TOTAL messages sent
# ---------------------------------------------------------------------------

_arc_cache: dict = {"mtime": None, "events": []}


def get_arc_event(total_messages: int) -> dict | None:
    """The current Tyler-arc event: the LAST event whose `after_messages` has
    passed (events are cumulative). Returns {id, text, followup} or None when
    the file is missing/empty. `text` is the live just-happened phrasing for
    the one-time delivery; `followup` (may be None) is the background phrasing
    once she's told him — a live moment without one simply fades. Pure derive
    from the message count — no DB writes here; the caller tracks which event
    she has already told him about."""
    try:
        mtime = os.path.getmtime(TYLER_ARC_FILE)
    except OSError:
        return None
    if _arc_cache["mtime"] != mtime:
        try:
            with open(TYLER_ARC_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            events = [
                e for e in (data.get("events") or [])
                if e.get("id") and e.get("text") and e.get("after_messages") is not None
            ]
            events.sort(key=lambda e: e["after_messages"])
        except Exception as e:
            logger.warning("Failed to load Tyler arc from %s: %s", TYLER_ARC_FILE, e)
            events = []
        _arc_cache.update(mtime=mtime, events=events)

    current = None
    for event in _arc_cache["events"]:
        if total_messages >= event["after_messages"]:
            current = event
        else:
            break
    if not current:
        return None
    followup = (current.get("followup") or "").strip() or None
    return {"id": current["id"], "text": current["text"].strip(), "followup": followup}


async def _shared_ids(user_id: int, kind: str) -> set[str]:
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT item_id FROM shared_content WHERE user_id = ? AND kind = ?",
            (user_id, kind),
        )
        rows = await cursor.fetchall()
        return {row["item_id"] for row in rows}
    finally:
        await conn.close()


def _matches_tags(item: dict, preferred_tags: list[str] | None) -> bool:
    if not preferred_tags:
        return False
    item_tags = {str(t).lower() for t in (item.get("tags") or [])}
    wanted = {str(t).lower() for t in preferred_tags}
    return bool(item_tags & wanted)


async def pick_unshared(
    user_id: int, kind: str, preferred_tags: list[str] | None = None
) -> dict | None:
    """Return the NEXT library item this user hasn't been told yet, or None if the
    pool is exhausted (caller should then fall back to a personalised one).

    Order: file order (first to last). If `preferred_tags` is given, the earliest
    unshared item whose tags intersect them wins; otherwise it falls back to the
    earliest unshared item, so behaviour is unchanged when nothing matches.
    """
    items = _load(kind)
    if not items:
        return None
    shared = await _shared_ids(user_id, kind)
    candidates = [it for it in items if it["id"] not in shared]
    if not candidates:
        return None
    # Prefer the earliest unshared item that matches the current location/context.
    for it in candidates:
        if _matches_tags(it, preferred_tags):
            return it
    # No tag match (or none requested) — keep the deterministic sequential order.
    return candidates[0]


async def pick_from_library(
    user_id: int, kind: str, preferred_tags: list[str] | None = None
) -> dict | None:
    """Always return a library item for this kind, never improvising.

    Prefers items the user hasn't heard yet (and, within those, ones matching the
    current location/context tags); once the pool is exhausted it resets the
    user's shared history for this kind and starts the rotation over, so a
    story/fantasy request ALWAYS comes from the authored file.
    """
    item = await pick_unshared(user_id, kind, preferred_tags)
    if item:
        return item
    # Pool exhausted — reset and pick fresh from the full library.
    await reset_shared(user_id, kind)
    return await pick_unshared(user_id, kind, preferred_tags)


async def reset_shared(user_id: int, kind: str) -> None:
    """Clear which items of this kind the user has already been told."""
    conn = await get_connection()
    try:
        await conn.execute(
            "DELETE FROM shared_content WHERE user_id = ? AND kind = ?",
            (user_id, kind),
        )
        await conn.commit()
    finally:
        await conn.close()


async def mark_shared(user_id: int, kind: str, item_id: str) -> None:
    conn = await get_connection()
    try:
        await conn.execute(
            "INSERT INTO shared_content (user_id, kind, item_id, shared_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(user_id, kind, item_id) DO NOTHING",
            (user_id, kind, item_id, time.time()),
        )
        await conn.commit()
    finally:
        await conn.close()


async def shared_count(user_id: int, kind: str) -> int:
    """How many of this kind she's already told the user (for logging/UX)."""
    return len(await _shared_ids(user_id, kind))
