"""
Engagement tracking — message counts and last-seen timestamps.

Tracks NSFW message count per user and total messages for greeting logic.
Used by chat_engine for last-seen notes and already-greeted detection.
"""

import logging
import time
from bot.memory.db import get_connection

logger = logging.getLogger(__name__)


async def track_message(user_id: int, classification: str) -> None:
    """Track a user message and its SFW/NSFW classification."""
    conn = await get_connection()
    try:
        now = time.time()
        await conn.execute("""
            INSERT INTO engagement_state (user_id, nsfw_count, total_messages, last_message_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                nsfw_count = CASE WHEN ? = 'nsfw'
                    THEN engagement_state.nsfw_count + 1
                    ELSE engagement_state.nsfw_count END,
                total_messages = engagement_state.total_messages + 1,
                last_message_at = ?
        """, (user_id, 1 if classification == "nsfw" else 0, now, classification, now))
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
