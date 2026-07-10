"""
PostgreSQL backend with a thin aiosqlite-compatible adapter.

The rest of the codebase was written against aiosqlite's
`conn = await get_connection(); cur = await conn.execute(sql, params);
row = await cur.fetchone(); await conn.commit(); await conn.close()` pattern
with `?` placeholders. To avoid rewriting every module, `get_connection()`
returns a small adapter over an asyncpg pool connection that:
  - rewrites `?` placeholders to `$1, $2, ...`
  - runs fetch() for SELECTs (and RETURNING) and execute() otherwise
  - exposes fetchone()/fetchall(), a no-op commit(), and close() (pool release)
asyncpg Records support `row["col"]` access, so callers are unchanged.
"""

import asyncio
import asyncpg

from bot.config import DATABASE_URL

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    mode TEXT NOT NULL DEFAULT 'sexting'
);
ALTER TABLE messages ADD COLUMN IF NOT EXISTS image_url TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tag TEXT;
CREATE TABLE IF NOT EXISTS memories (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 5,
    embedding BYTEA,
    created_at DOUBLE PRECISION NOT NULL,
    last_accessed DOUBLE PRECISION
);
ALTER TABLE memories ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'sexting';
CREATE TABLE IF NOT EXISTS sent_content (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    content_id TEXT NOT NULL,
    category TEXT NOT NULL,
    sent_at DOUBLE PRECISION NOT NULL,
    paid INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS user_facts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    first_seen DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS engagement_state (
    user_id BIGINT PRIMARY KEY,
    nsfw_count INTEGER NOT NULL DEFAULT 0,
    total_messages INTEGER NOT NULL DEFAULT 0,
    last_push_at DOUBLE PRECISION DEFAULT 0,
    last_selfie_at DOUBLE PRECISION DEFAULT 0,
    last_message_at DOUBLE PRECISION DEFAULT 0,
    last_reengage_at DOUBLE PRECISION DEFAULT 0
);
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS last_push_content_id TEXT;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS first_message_at DOUBLE PRECISION;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS active_days INTEGER DEFAULT 0;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS last_active_date TEXT;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS last_arc_id TEXT;
CREATE TABLE IF NOT EXISTS shared_content (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    kind TEXT NOT NULL,
    item_id TEXT NOT NULL,
    shared_at DOUBLE PRECISION NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_shared_content_uniq ON shared_content(user_id, kind, item_id);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_user_mode ON messages(user_id, mode);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user_mode ON memories(user_id, mode);
CREATE INDEX IF NOT EXISTS idx_sent_content_user ON sent_content(user_id);
CREATE INDEX IF NOT EXISTS idx_sent_content_paid ON sent_content(user_id, content_id, paid);
CREATE INDEX IF NOT EXISTS idx_user_facts ON user_facts(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_facts_key ON user_facts(user_id, key);
"""

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


def _convert_placeholders(sql: str) -> str:
    """Rewrite aiosqlite-style `?` placeholders to asyncpg-style `$1, $2, ...`."""
    out = []
    n = 0
    for ch in sql:
        if ch == "?":
            n += 1
            out.append(f"${n}")
        else:
            out.append(ch)
    return "".join(out)


class _Cursor:
    def __init__(self, rows: list):
        self._rows = rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None

    async def fetchall(self):
        return self._rows


class _ConnAdapter:
    """Mimics the slice of the aiosqlite.Connection API the codebase uses."""

    def __init__(self, raw: asyncpg.Connection):
        self._raw = raw

    async def execute(self, sql: str, params=()):
        query = _convert_placeholders(sql)
        args = tuple(params) if params else ()
        head = sql.lstrip()[:6].upper()
        if head == "SELECT" or "RETURNING" in sql.upper():
            rows = await self._raw.fetch(query, *args)
            return _Cursor(list(rows))
        await self._raw.execute(query, *args)
        return _Cursor([])

    async def commit(self):
        # asyncpg autocommits outside an explicit transaction.
        pass

    async def close(self):
        await _pool.release(self._raw)


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    return _pool


async def get_connection() -> _ConnAdapter:
    pool = await _get_pool()
    raw = await pool.acquire()
    return _ConnAdapter(raw)


async def init_db() -> None:
    pool = await _get_pool()
    statements = [s.strip() for s in SCHEMA.split(";") if s.strip()]
    async with pool.acquire() as conn:
        for stmt in statements:
            await conn.execute(stmt)
