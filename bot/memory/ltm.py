import re
import time
import logging
import math
import numpy as np
from bot.memory.db import get_connection
from bot.memory.embeddings import embed_text, cosine_similarity
from bot.memory.validation import (
    MEMORY_CATEGORIES,
    MemoryValidationError,
    validate_importance,
    validate_memory_content,
    validate_stored_memory,
)
from bot.config import (
    LTM_TOP_K,
    LTM_SIMILARITY_WEIGHT,
    LTM_IMPORTANCE_WEIGHT,
    LTM_RECENCY_WEIGHT,
)

logger = logging.getLogger(__name__)

# A high importance/recency score must not make an unrelated memory eligible.
# Both semantic similarity and the final blended relevance must clear a floor.
LTM_MIN_SIMILARITY = 0.30
LTM_MIN_RELEVANCE_SCORE = 0.35
_MAX_RETRIEVAL_TOP_K = 10

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


def _validated_embedding(embedding: np.ndarray) -> np.ndarray:
    try:
        vector = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("memory embedding is invalid") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("memory embedding must be a finite one-dimensional vector")
    return vector


def _safe_memory(memory: dict) -> dict | None:
    """Return a normalised copy or quarantine an unsafe legacy DB row."""
    try:
        category, content, importance = validate_stored_memory(
            memory.get("category"),
            memory.get("content"),
            memory.get("importance"),
        )
        created_at = float(memory.get("created_at"))
        if not math.isfinite(created_at) or created_at <= 0:
            raise ValueError("invalid created_at")
    except (MemoryValidationError, TypeError, ValueError) as exc:
        logger.warning("Ignoring unsafe/invalid memory row %s: %s", memory.get("id"), exc)
        return None
    safe = dict(memory)
    safe.update({
        "category": category,
        "content": content,
        "importance": importance,
        "created_at": created_at,
    })
    return safe


async def store_memory(
    user_id: int,
    category: str,
    content: str,
    importance: int,
    embedding: np.ndarray,
    mode: str = "sexting",
    created_at: float | None = None,
) -> None:
    category, content, importance = validate_stored_memory(
        category, content, importance
    )
    vector = _validated_embedding(embedding)
    if created_at is None:
        created_at = time.time()
    if (
        isinstance(created_at, bool)
        or not isinstance(created_at, (int, float))
        or not math.isfinite(float(created_at))
        or float(created_at) <= 0
    ):
        raise ValueError("memory created_at must be a positive finite timestamp")
    conn = await get_connection()
    try:
        await conn.execute(
            "INSERT INTO memories (user_id, category, content, importance, embedding, created_at, mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                category,
                content,
                importance,
                vector.tobytes(),
                float(created_at),
                mode,
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_all_memories(
    user_id: int,
    mode: str | None = None,
    limit: int | None = None,
    oldest: bool = False,
) -> list[dict]:
    """Return a user's memories. `mode=None` spans all modes (back-compat);
    pass a mode to keep story and sexting long-term memory separate.

    ``limit`` lets compaction bound the records materialised and sent to its
    LLM.  Existing callers retain the previous unbounded behaviour.
    """
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            return []
        limit = min(limit, 2_000)
    order = "ASC" if oldest else "DESC"
    limit_clause = " LIMIT ?" if limit is not None else ""
    conn = await get_connection()
    try:
        if mode is None:
            cursor = await conn.execute(
                "SELECT id, category, content, importance, embedding, created_at, last_accessed "
                f"FROM memories WHERE user_id = ? ORDER BY created_at {order}{limit_clause}",
                (user_id, limit) if limit is not None else (user_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT id, category, content, importance, embedding, created_at, last_accessed "
                f"FROM memories WHERE user_id = ? AND mode = ? ORDER BY created_at {order}{limit_clause}",
                (user_id, mode, limit) if limit is not None else (user_id, mode),
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
            except (TypeError, ValueError):
                emb = None
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
    if not isinstance(message, str) or not message.strip():
        return []
    if type(top_k) is not int or top_k <= 0:
        return []
    top_k = min(top_k, _MAX_RETRIEVAL_TOP_K)

    memories = await get_all_memories(user_id, mode=mode)
    if not memories:
        return []

    safe_memories = []
    for memory in memories:
        safe = _safe_memory(memory)
        if safe is not None and safe.get("embedding") is not None:
            safe_memories.append(safe)
    if not safe_memories:
        return []

    try:
        query_embedding = await embed_text(message)
    except Exception as exc:
        # Memory recall is optional; a provider failure must not break chat.
        logger.error("LTM query embedding failed for user %d: %s", user_id, exc)
        return []

    now = time.time()
    ages = [max(0.0, now - memory["created_at"]) for memory in safe_memories]
    max_age = max(ages) or 1.0

    scored = []
    for mem, age in zip(safe_memories, ages):
        sim = cosine_similarity(query_embedding, mem["embedding"])
        if sim < LTM_MIN_SIMILARITY:
            continue
        imp = mem["importance"] / 10.0
        recency = 1.0 - (age / max_age)

        score = (
            sim * LTM_SIMILARITY_WEIGHT
            + imp * LTM_IMPORTANCE_WEIGHT
            + recency * LTM_RECENCY_WEIGHT
        )
        if math.isfinite(score) and score >= LTM_MIN_RELEVANCE_SCORE:
            scored.append({"memory": mem, "score": score, "similarity": sim})

    scored.sort(key=lambda item: (item["score"], item["similarity"]), reverse=True)
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
        except Exception as exc:
            # last_accessed is bookkeeping, not a reason to lose a valid recall.
            logger.warning("Could not update LTM access timestamps: %s", exc)
        finally:
            await conn.close()

    return [item["memory"] for item in top]


async def find_similar_memory(
    user_id: int, embedding: np.ndarray, threshold: float = 0.85, mode: str | None = None
) -> dict | None:
    """Find an existing memory with cosine similarity above threshold."""
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        threshold = 0.85
    threshold = min(1.0, max(-1.0, float(threshold)))
    memories = await get_all_memories(user_id, mode=mode)
    best_match = None
    best_sim = -1.0
    for mem in memories:
        safe = _safe_memory(mem)
        if safe is None or safe["embedding"] is None:
            continue
        sim = cosine_similarity(embedding, safe["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_match = safe
    if best_sim >= threshold and best_match:
        return best_match
    return None


async def update_memory(
    memory_id: int, content: str, importance: int, embedding: np.ndarray
) -> None:
    """Update an existing memory entry."""
    content = validate_memory_content(content)
    importance = validate_importance(importance)
    vector = _validated_embedding(embedding)
    conn = await get_connection()
    try:
        await conn.execute(
            "UPDATE memories SET content = ?, importance = ?, embedding = ? WHERE id = ?",
            (content, importance, vector.tobytes(), memory_id),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_recent_by_category(
    user_id: int, categories: list[str], limit: int = 2, mode: str | None = None
) -> list[dict]:
    """Most recent memories of the given categories (e.g. 'thread' for open
    conversational loops, 'event' for things he mentioned doing). Cheap SQL
    only — no embeddings — used by re-engagement and gap follow-ups."""
    if not categories:
        return []
    categories = [
        category for category in categories
        if isinstance(category, str) and category in MEMORY_CATEGORIES
    ]
    if not categories:
        return []
    if type(limit) is not int or limit <= 0:
        return []
    limit = min(limit, _MAX_RETRIEVAL_TOP_K)
    conn = await get_connection()
    try:
        placeholders = ",".join("?" for _ in categories)
        if mode is None:
            cursor = await conn.execute(
                f"SELECT id, category, content, importance, created_at FROM memories "
                f"WHERE user_id = ? AND category IN ({placeholders}) "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, *categories, limit),
            )
        else:
            cursor = await conn.execute(
                f"SELECT id, category, content, importance, created_at FROM memories "
                f"WHERE user_id = ? AND mode = ? AND category IN ({placeholders}) "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, mode, *categories, limit),
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            safe = _safe_memory({
                "id": row["id"],
                "category": row["category"],
                "content": row["content"],
                "importance": row["importance"],
                "created_at": row["created_at"],
            })
            if safe is not None:
                results.append(safe)
        return results
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
    if len(memory_ids) > 500 or any(type(memory_id) is not int or memory_id <= 0 for memory_id in memory_ids):
        raise ValueError("memory deletion IDs are invalid or exceed the bounded batch size")
    memory_ids = list(dict.fromkeys(memory_ids))
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
