"""
Engagement tracking — message counts and last-seen timestamps.

Tracks NSFW message count per user and total messages for greeting logic.
Used by chat_engine for last-seen notes and already-greeted detection.
"""

import logging
import time
from datetime import datetime

from bot.memory.db import get_connection
from bot.time_context import TIMEZONE

logger = logging.getLogger(__name__)


async def track_message(user_id: int, classification: str) -> None:
    """Track a user message and its SFW/NSFW classification.

    Also counts ACTIVE chat days (distinct Miami-time calendar days on which
    the user messaged) — the Tyler arc advances by these, so her life moves
    every time he comes back on a new day, not by slow real-world weeks."""
    conn = await get_connection()
    try:
        now = time.time()
        today = datetime.now(TIMEZONE).date().isoformat()
        await conn.execute("""
            INSERT INTO engagement_state (user_id, nsfw_count, total_messages, last_message_at, first_message_at, active_days, last_active_date)
            VALUES (?, ?, 1, ?, ?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                nsfw_count = CASE WHEN ? = 'nsfw'
                    THEN engagement_state.nsfw_count + 1
                    ELSE engagement_state.nsfw_count END,
                total_messages = engagement_state.total_messages + 1,
                last_message_at = ?,
                first_message_at = COALESCE(engagement_state.first_message_at, ?),
                active_days = COALESCE(engagement_state.active_days, 0)
                    + CASE WHEN engagement_state.last_active_date IS DISTINCT FROM ? THEN 1 ELSE 0 END,
                last_active_date = ?
        """, (user_id, 1 if classification == "nsfw" else 0, now, now, today,
              classification, now, now, today, today))
        await conn.commit()
    finally:
        await conn.close()


async def set_last_arc_id(user_id: int, arc_id: str) -> None:
    """Remember which Tyler-arc event she has already told him about, so a
    fresh event gets the one-time 'tell him' treatment exactly once."""
    conn = await get_connection()
    try:
        await conn.execute(
            "UPDATE engagement_state SET last_arc_id = ? WHERE user_id = ?",
            (arc_id, user_id),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_engagement_state(user_id: int) -> dict | None:
    """Get engagement state for a user (used for greeting/last-seen logic)."""
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT * FROM engagement_state WHERE user_id = ?",
            (user_id,),
        )
        return await cursor.fetchone()
    finally:
        await conn.close()


async def record_reengage(user_id: int) -> None:
    """Record that we sent a re-engagement message."""
    conn = await get_connection()
    try:
        await conn.execute(
            "UPDATE engagement_state SET last_reengage_at = ? WHERE user_id = ?",
            (time.time(), user_id),
        )
        await conn.commit()
    finally:
        await conn.close()
