import time
from bot.memory.db import get_connection


# Character bounds are deliberately enforced at the point where STM becomes
# model context.  They protect every caller without requiring provider-specific
# tokenizers, while retaining both the beginning and the most recent tail of a
# long message.
STM_CONTEXT_MAX_CHARS = 48_000
STM_MESSAGE_MAX_CHARS = 8_000
_TRUNCATION_MARKER = " [...truncated...] "


def _truncate_for_context(content: str, max_chars: int) -> str:
    if not isinstance(content, str):
        content = str(content or "")
    if len(content) <= max_chars:
        return content
    if max_chars <= len(_TRUNCATION_MARKER):
        return content[:max_chars]
    available = max_chars - len(_TRUNCATION_MARKER)
    head_chars = available // 2
    tail_chars = available - head_chars
    return content[:head_chars] + _TRUNCATION_MARKER + content[-tail_chars:]


async def add_message(
    user_id: int,
    role: str,
    content: str,
    mode: str = "sexting",
    tag: str | None = None,
) -> int:
    """Persist a message. `tag` marks special content (e.g. 'fantasy_card' /
    'story_card') so the summarizer can treat it as fiction, not real events.

    Returns the durable row ID so callers that coordinate a second-phase
    operation (such as finalizing a reserved media offer) can compensate the
    exact message if that operation fails.
    """
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "INSERT INTO messages (user_id, role, content, timestamp, mode, tag) "
            "VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
            (user_id, role, content, time.time(), mode, tag),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("Persisted message did not return an ID")
        await conn.commit()
        return int(row["id"])
    finally:
        await conn.close()


async def replace_assistant_message(
    message_id: int,
    user_id: int,
    content: str,
) -> bool:
    """Replace one exact assistant row during a failed two-phase delivery.

    The user/role predicates prevent a compensating commerce write from ever
    touching user-authored content or another account's message.
    """
    if type(message_id) is not int or message_id <= 0:
        return False
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "UPDATE messages SET content = ? "
            "WHERE id = ? AND user_id = ? AND role = 'assistant' RETURNING id",
            (content, message_id, user_id),
        )
        row = await cursor.fetchone()
        await conn.commit()
        return row is not None
    finally:
        await conn.close()


async def get_recent_messages(user_id: int, limit: int, mode: str = "sexting") -> list[dict]:
    if type(limit) is not int or limit <= 0:
        return []
    limit = min(limit, 100)
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT role, content, timestamp FROM messages "
            "WHERE user_id = ? AND mode = ? ORDER BY timestamp DESC, id DESC LIMIT ?",
            (user_id, mode, limit * 2),
        )
        rows = await cursor.fetchall()
        # Rows arrive newest-first. Spend the context budget on the most recent
        # turns, then restore chronological order for the chat API.
        remaining = STM_CONTEXT_MAX_CHARS
        selected = []
        for row in rows:
            if remaining <= 0:
                break
            allowance = min(STM_MESSAGE_MAX_CHARS, remaining)
            content = _truncate_for_context(row["content"], allowance)
            selected.append({
                "role": row["role"],
                "content": content,
                "timestamp": row["timestamp"],
            })
            remaining -= len(content)
        return list(reversed(selected))
    finally:
        await conn.close()


async def count_turns(user_id: int, mode: str | None = "sexting") -> int:
    """Count user turns. mode=None counts across all modes (shared memory)."""
    conn = await get_connection()
    try:
        if mode is None:
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE user_id = ? AND role = 'user'",
                (user_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE user_id = ? AND mode = ? AND role = 'user'",
                (user_id, mode),
            )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0
    finally:
        await conn.close()


async def count_pending_memory_turns(user_id: int, mode: str | None = "sexting") -> int:
    """Count user turns not yet summarized or safely deferred.

    This is separate from the monotonic engagement counter because memory
    extraction must consider only work still pending. The distinction prevents
    a deferred old batch from making a brand-new turn eligible for immediate
    summarization.
    """
    conn = await get_connection()
    try:
        if mode is None:
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM messages "
                "WHERE user_id = ? AND role = 'user' AND memory_deferred_at IS NULL",
                (user_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM messages "
                "WHERE user_id = ? AND mode = ? AND role = 'user' "
                "AND memory_deferred_at IS NULL",
                (user_id, mode),
            )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0
    finally:
        await conn.close()


async def get_oldest_messages(user_id: int, limit: int, mode: str | None = "sexting") -> list[dict]:
    """Get the oldest messages still pending memory extraction.

    Rows marked after a valid empty extraction remain in normal history and
    recent context, but are skipped here so later durable facts cannot starve.
    """
    conn = await get_connection()
    try:
        if mode is None:
            cursor = await conn.execute(
                "SELECT id, role, content, timestamp, mode, tag FROM messages "
                "WHERE user_id = ? AND memory_deferred_at IS NULL "
                "ORDER BY timestamp ASC, id ASC LIMIT ?",
                (user_id, limit * 2),
            )
        else:
            cursor = await conn.execute(
                "SELECT id, role, content, timestamp, mode, tag FROM messages "
                "WHERE user_id = ? AND mode = ? AND memory_deferred_at IS NULL "
                "ORDER BY timestamp ASC, id ASC LIMIT ?",
                (user_id, mode, limit * 2),
            )
        rows = await cursor.fetchall()
        return [
            {"id": row["id"], "role": row["role"], "content": row["content"],
             "timestamp": row["timestamp"], "mode": row["mode"], "tag": row["tag"]}
            for row in rows
        ]
    finally:
        await conn.close()


async def defer_messages_from_memory(message_ids: list[int]) -> None:
    """Persistently skip a valid-empty batch without deleting its content."""
    if not message_ids:
        return
    if len(message_ids) > 500 or any(
        type(message_id) is not int or message_id <= 0 for message_id in message_ids
    ):
        raise ValueError("message deferral IDs are invalid or exceed the batch bound")
    message_ids = list(dict.fromkeys(message_ids))
    deferred_at = time.time()
    conn = await get_connection()
    try:
        placeholders = ",".join("?" for _ in message_ids)
        await conn.execute(
            "UPDATE messages SET memory_deferred_at = ? "
            f"WHERE id IN ({placeholders}) AND memory_deferred_at IS NULL",
            (deferred_at, *message_ids),
        )
        await conn.commit()
    finally:
        await conn.close()


async def delete_messages_by_ids(message_ids: list[int]) -> None:
    if not message_ids:
        return
    conn = await get_connection()
    try:
        placeholders = ",".join("?" for _ in message_ids)
        await conn.execute(
            f"DELETE FROM messages WHERE id IN ({placeholders})",
            message_ids,
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_all_messages(user_id: int, mode: str = "sexting") -> list[dict]:
    """Get all messages for a user in a specific mode (for chat history API)."""
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT id, role, content, timestamp, mode FROM messages "
            "WHERE user_id = ? AND mode = ? ORDER BY timestamp ASC, id ASC",
            (user_id, mode),
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "mode": row["mode"],
            }
            for row in rows
        ]
    finally:
        await conn.close()
