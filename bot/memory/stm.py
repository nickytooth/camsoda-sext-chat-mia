import time
from bot.memory.db import get_connection


async def add_message(
    user_id: int,
    role: str,
    content: str,
    mode: str = "sexting",
    image_url: str | None = None,
) -> None:
    conn = await get_connection()
    try:
        await conn.execute(
            "INSERT INTO messages (user_id, role, content, timestamp, mode, image_url) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, role, content, time.time(), mode, image_url),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_recent_messages(user_id: int, limit: int, mode: str = "sexting") -> list[dict]:
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT role, content, timestamp FROM messages "
            "WHERE user_id = ? AND mode = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, mode, limit * 2),
        )
        rows = await cursor.fetchall()
        return [
            {"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
            for row in reversed(rows)
        ]
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


async def get_oldest_messages(user_id: int, limit: int, mode: str | None = "sexting") -> list[dict]:
    """Get oldest messages. mode=None spans all modes (shared memory)."""
    conn = await get_connection()
    try:
        if mode is None:
            cursor = await conn.execute(
                "SELECT id, role, content, timestamp, mode FROM messages "
                "WHERE user_id = ? ORDER BY timestamp ASC LIMIT ?",
                (user_id, limit * 2),
            )
        else:
            cursor = await conn.execute(
                "SELECT id, role, content, timestamp, mode FROM messages "
                "WHERE user_id = ? AND mode = ? ORDER BY timestamp ASC LIMIT ?",
                (user_id, mode, limit * 2),
            )
        rows = await cursor.fetchall()
        return [
            {"id": row["id"], "role": row["role"], "content": row["content"],
             "timestamp": row["timestamp"], "mode": row["mode"]}
            for row in rows
        ]
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
            "SELECT role, content, timestamp, image_url FROM messages "
            "WHERE user_id = ? AND mode = ? ORDER BY timestamp ASC, id ASC",
            (user_id, mode),
        )
        rows = await cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "image_url": row["image_url"],
            }
            for row in rows
        ]
    finally:
        await conn.close()
