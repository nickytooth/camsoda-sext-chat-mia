import re
import time
import logging
import numpy as np
from bot.memory.db import get_connection
from bot.memory.embeddings import embed_text, cosine_similarity
from bot.config import (
    LTM_TOP_K,
    LTM_SIMILARITY_WEIGHT,
    LTM_IMPORTANCE_WEIGHT,
    LTM_RECENCY_WEIGHT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrieval gating — skip embed+search for trivial messages
# ---------------------------------------------------------------------------

_CALLBACK_CUES = re.compile(
    r"\b(remember|you said|you told|last time|before|didn.t you|earlier|"
    r"you mentioned|we talked|you asked|you know|we discussed|you were)\b",
    re.IGNORECASE,
)

_GREETING_PATTERN = re.compile(
    r"^(h(ey|i|ello|ola)|yo|sup|what.?s up|gm|good (morning|evening|night)|"
    r"how are you|how.?s it going|what.?s good|wyd|hru|lol|haha|"
    r"ok(ay)?|k|sure|yeah|yep|nah|no|yes|bye|gn|ttyl|brb|omg|wow|damn|fr|ngl"
    r"|[\U0001f600-\U0001f64f\U0001f900-\U0001f9ff\u2764\u2665\u200d]+)$",
    re.IGNORECASE,
)

_turn_counter: dict[int, int] = {}
_last_message_time: dict[int, float] = {}
_LTM_EVERY_N_TURNS = 5
_GAP_THRESHOLD = 3600  # 1 hour


def clear_retrieval_state(user_id: int) -> None:
    """Drop the in-memory retrieval-gating counters for a user (called on reset),
    so a fresh conversation starts with clean turn/gap tracking."""
    _turn_counter.pop(user_id, None)
    _last_message_time.pop(user_id, None)


def should_retrieve(user_id: int, message: str) -> bool:
    """Decide whether LTM retrieval is worth the embed API call."""
    now = time.time()

    # Track turns
    _turn_counter[user_id] = _turn_counter.get(user_id, 0) + 1
    turn = _turn_counter[user_id]

    # Track gap
    last_time = _last_message_time.get(user_id, 0)
    _last_message_time[user_id] = now
    gap = now - last_time if last_time > 0 else 0

    # Always retrieve on callback cues ("remember", "you said", etc.)
    if _CALLBACK_CUES.search(message):
        logger.debug("LTM gate: FIRE (callback cue) user %d", user_id)
        return True

    # Skip pure greetings / emoji / very short trivial messages
    stripped = message.strip()
    if len(stripped) < 15 and _GREETING_PATTERN.match(stripped):
        # But still fire on the every-N fallback
        if turn % _LTM_EVERY_N_TURNS == 0:
            logger.debug("LTM gate: FIRE (every-N fallback on greeting) user %d", user_id)
            return True
        logger.debug("LTM gate: SKIP (greeting/trivial) user %d", user_id)
        return False

    # Fire on first message after a gap (user came back after >1h)
    if last_time > 0 and gap >= _GAP_THRESHOLD:
        logger.debug("LTM gate: FIRE (gap %.0fs) user %d", gap, user_id)
        return True

    # Fire on substantive messages (>30 chars)
    if len(stripped) > 30:
        logger.debug("LTM gate: FIRE (long message) user %d", user_id)
        return True

    # Every-N-turns fallback so nothing goes stale
    if turn % _LTM_EVERY_N_TURNS == 0:
        logger.debug("LTM gate: FIRE (every-%d turns) user %d", _LTM_EVERY_N_TURNS, user_id)
        return True

    logger.debug("LTM gate: SKIP (short, no cues) user %d", user_id)
    return False


async def store_memory(
    user_id: int, category: str, content: str, importance: int, embedding: np.ndarray,
    mode: str = "sexting",
) -> None:
    conn = await get_connection()
    try:
        await conn.execute(
            "INSERT INTO memories (user_id, category, content, importance, embedding, created_at, mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, category, content, importance, embedding.tobytes(), time.time(), mode),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_all_memories(user_id: int, mode: str | None = None) -> list[dict]:
    """Return a user's memories. `mode=None` spans all modes (back-compat);
    pass a mode to keep story and sexting long-term memory separate."""
    conn = await get_connection()
    try:
        if mode is None:
            cursor = await conn.execute(
                "SELECT id, category, content, importance, embedding, created_at, last_accessed "
                "FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT id, category, content, importance, embedding, created_at, last_accessed "
                "FROM memories WHERE user_id = ? AND mode = ? ORDER BY created_at DESC",
                (user_id, mode),
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
            results.append({
                "id": row["id"],
                "category": row["category"],
                "content": row["content"],
                "importance": row["importance"],
                "embedding": emb,
                "created_at": row["created_at"],
                "last_accessed": row["last_accessed"],
            })
        return results
    finally:
        await conn.close()


async def retrieve_relevant(
    user_id: int, message: str, top_k: int = LTM_TOP_K, mode: str | None = None
) -> list[dict]:
    memories = await get_all_memories(user_id, mode=mode)
    if not memories:
        return []

    query_embedding = await embed_text(message)
    now = time.time()
    max_age = max((now - m["created_at"]) for m in memories) or 1.0

    scored = []
    for mem in memories:
        if mem["embedding"] is None:
            continue

        sim = cosine_similarity(query_embedding, mem["embedding"])
        imp = mem["importance"] / 10.0
        recency = 1.0 - ((now - mem["created_at"]) / max_age)

        score = (
            sim * LTM_SIMILARITY_WEIGHT
            + imp * LTM_IMPORTANCE_WEIGHT
            + recency * LTM_RECENCY_WEIGHT
        )
        scored.append({"memory": mem, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    # Update last_accessed for retrieved memories
    if top:
        conn = await get_connection()
        try:
            for item in top:
                await conn.execute(
                    "UPDATE memories SET last_accessed = ? WHERE id = ?",
                    (now, item["memory"]["id"]),
                )
            await conn.commit()
        finally:
            await conn.close()

    return [item["memory"] for item in top]


async def find_similar_memory(
    user_id: int, embedding: np.ndarray, threshold: float = 0.85, mode: str | None = None
) -> dict | None:
    """Find an existing memory with cosine similarity above threshold."""
    memories = await get_all_memories(user_id, mode=mode)
    best_match = None
    best_sim = 0.0
    for mem in memories:
        if mem["embedding"] is None:
            continue
        sim = cosine_similarity(embedding, mem["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_match = mem
    if best_sim >= threshold and best_match:
        return best_match
    return None


async def update_memory(
    memory_id: int, content: str, importance: int, embedding: np.ndarray
) -> None:
    """Update an existing memory entry."""
    conn = await get_connection()
    try:
        await conn.execute(
            "UPDATE memories SET content = ?, importance = ?, embedding = ? WHERE id = ?",
            (content, importance, embedding.tobytes(), memory_id),
        )
        await conn.commit()
    finally:
        await conn.close()


async def count_memories(user_id: int, mode: str | None = None) -> int:
    conn = await get_connection()
    try:
        if mode is None:
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE user_id = ?",
                (user_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE user_id = ? AND mode = ?",
                (user_id, mode),
            )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0
    finally:
        await conn.close()


async def delete_memories_by_ids(memory_ids: list[int]) -> None:
    if not memory_ids:
        return
    conn = await get_connection()
    try:
        placeholders = ",".join("?" for _ in memory_ids)
        await conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        )
        await conn.commit()
    finally:
        await conn.close()
