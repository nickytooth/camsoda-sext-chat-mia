"""
Engagement tracking — message counts and last-seen timestamps.

Tracks NSFW message count per user and total messages for greeting logic.
Used by chat_engine for last-seen notes and already-greeted detection.
"""

import json
import logging
import time
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING

from bot.memory.db import get_connection
from bot.time_context import TIMEZONE

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bot.heat import HeatTurnResult


async def record_user_message(user_id: int) -> None:
    """Increment the monotonic lifetime count for one ingested user message.

    This deliberately does not update last-seen or NSFW analytics: those remain
    batch-level concerns in ``track_message``. Keeping the lifetime counter at
    ingestion means debounce batching and later STM summarization cannot make a
    long-running narrative arc move backwards or skip rapid-fire messages.
    """
    conn = await get_connection()
    try:
        await conn.execute(
            """
            INSERT INTO engagement_state (user_id, lifetime_user_messages)
            VALUES (?, 1)
            ON CONFLICT(user_id) DO UPDATE SET
                lifetime_user_messages =
                    COALESCE(engagement_state.lifetime_user_messages, 0) + 1
            """,
            (user_id,),
        )
        await conn.commit()
    finally:
        await conn.close()


async def track_message(
    user_id: int,
    classification: str,
) -> None:
    """Track a user message and its SFW/NSFW classification.

    Also retains active-day analytics. The Tyler arc uses the separate
    ingestion-time ``lifetime_user_messages`` counter, so this batch-level
    update cannot collapse rapid-fire messages into one progression step."""
    conn = await get_connection()
    try:
        now = time.time()
        today = datetime.now(TIMEZONE).date().isoformat()
        async with conn.transaction():
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


async def track_heat_batch(
    user_id: int,
    raw_messages: Iterable[str],
    *,
    now: float | None = None,
    timeout_seconds: int = 3600,
    commerce_decline: bool = False,
    suppress_progression: bool = False,
    direct_media_request: bool = False,
) -> tuple["HeatTurnResult", int]:
    """Atomically claim and persist one processed chat batch.

    The row lock makes ``total_messages`` the single source of truth for the
    batch number. Two workers can no longer both calculate ``N + 1`` and then
    let one guarded heat update disappear; each committed call receives a
    unique batch number and reduces from the latest durable heat state.
    """
    from bot.heat import HeatState, advance_heat

    effective_now = time.time() if now is None else float(now)
    today = datetime.fromtimestamp(effective_now, TIMEZONE).date().isoformat()
    messages = tuple(str(message or "") for message in raw_messages)
    conn = await get_connection()
    try:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO engagement_state (user_id)
                VALUES (?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                (user_id,),
            )
            cursor = await conn.execute(
                "SELECT * FROM engagement_state WHERE user_id = ? FOR UPDATE",
                (user_id,),
            )
            previous = await cursor.fetchone()
            if previous is None:
                raise RuntimeError("engagement state row was not created")

            try:
                previous_total = int(previous["total_messages"] or 0)
            except (KeyError, TypeError, ValueError):
                previous_total = 0
            try:
                previous_heat_batch = int(previous["heat_last_batch"] or 0)
            except (KeyError, TypeError, ValueError):
                previous_heat_batch = 0
            batch_number = max(0, previous_total, previous_heat_batch) + 1
            heat_turn = advance_heat(
                HeatState.from_mapping(previous),
                messages,
                now=effective_now,
                batch_number=batch_number,
                timeout_seconds=timeout_seconds,
                commerce_decline=commerce_decline,
                suppress_progression=suppress_progression,
                direct_media_request=direct_media_request,
            )
            state = heat_turn.state
            await conn.execute(
                """
                UPDATE engagement_state
                SET nsfw_count = COALESCE(nsfw_count, 0) + ?,
                    total_messages = GREATEST(
                        COALESCE(total_messages, 0),
                        COALESCE(heat_last_batch, 0)
                    ) + 1,
                    last_message_at = ?,
                    first_message_at = COALESCE(first_message_at, ?),
                    active_days = COALESCE(active_days, 0)
                        + CASE WHEN last_active_date IS DISTINCT FROM ? THEN 1 ELSE 0 END,
                    last_active_date = ?,
                    heat_stage = ?,
                    heat_progress = ?,
                    heat_last_sexual_at = ?,
                    heat_updated_at = ?,
                    heat_last_batch = ?,
                    heat_last_signal = ?,
                    sexual_pause_active = ?,
                    heat_blocked_acts = ?
                WHERE user_id = ?
                """,
                (
                    1 if heat_turn.sexual_batch else 0,
                    effective_now,
                    effective_now,
                    today,
                    today,
                    state.stage,
                    state.progress,
                    state.last_sexual_at,
                    state.updated_at,
                    state.last_batch,
                    state.last_signal,
                    state.consent_paused,
                    json.dumps(list(state.blocked_acts)),
                    user_id,
                ),
            )
        await conn.commit()
        return heat_turn, batch_number
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
