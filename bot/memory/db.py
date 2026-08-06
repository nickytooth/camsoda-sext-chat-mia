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
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tag TEXT;
-- A valid empty memory extraction must not delete the transcript or retry the
-- same oldest batch forever.  The persistent marker advances the summarizer
-- cursor across restarts while all original message content remains intact.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS memory_deferred_at DOUBLE PRECISION;
-- Remove columns left by the retired user-photo/vision feature.  These
-- migrations are intentionally destructive: the application is text-only and
-- must not retain backend references to uploaded user media.
ALTER TABLE messages DROP COLUMN IF EXISTS image_url;
-- Old vision descriptions were appended to user text with this exact backend
-- marker. Remove media-only rows and preserve any caption that preceded it.
DELETE FROM messages WHERE content LIKE '[User sent a photo:%';
UPDATE messages
SET content = BTRIM(SPLIT_PART(content, '[User sent a photo:', 1))
WHERE content LIKE '%[User sent a photo:%';
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
    lifetime_user_messages BIGINT NOT NULL DEFAULT 0,
    last_push_at DOUBLE PRECISION DEFAULT 0,
    last_message_at DOUBLE PRECISION DEFAULT 0,
    last_reengage_at DOUBLE PRECISION DEFAULT 0
);
ALTER TABLE engagement_state DROP COLUMN IF EXISTS last_selfie_at;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS last_push_content_id TEXT;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS first_message_at DOUBLE PRECISION;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS active_days INTEGER DEFAULT 0;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS last_active_date TEXT;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS last_arc_id TEXT;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS lifetime_user_messages BIGINT NOT NULL DEFAULT 0;
-- Visual-commerce pacing is deliberately based on total_messages, which is
-- incremented once per processed debounce batch. Raw lifetime_user_messages
-- must never be used for these fields.
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS sales_snooze_until_batch BIGINT;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS sales_reask_pending BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS sales_reask_asked_at_batch BIGINT;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS last_proactive_media_batch BIGINT;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS last_generic_media_type TEXT;
-- Conversation heat is a durable state machine.  It advances once per
-- processed debounce batch (never once per raw message) and survives
-- summarisation, reconnects and process restarts.
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS heat_stage TEXT NOT NULL DEFAULT 'low';
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS heat_progress INTEGER NOT NULL DEFAULT 0;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS heat_last_sexual_at DOUBLE PRECISION DEFAULT 0;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS heat_updated_at DOUBLE PRECISION DEFAULT 0;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS heat_last_batch BIGINT NOT NULL DEFAULT 0;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS heat_last_signal TEXT NOT NULL DEFAULT 'neutral';
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS sexual_pause_active BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE engagement_state ADD COLUMN IF NOT EXISTS heat_blocked_acts TEXT NOT NULL DEFAULT '[]';
-- Safe one-time/backward-compatible lower-bound backfill. Existing deployments
-- may already have summarized old message rows, so retain the larger historic
-- batch counter; otherwise use the exact user rows that are still available.
INSERT INTO engagement_state (user_id, lifetime_user_messages)
SELECT user_id, COUNT(*)
FROM messages
WHERE role = 'user'
GROUP BY user_id
ON CONFLICT(user_id) DO UPDATE SET
    lifetime_user_messages = GREATEST(
        engagement_state.lifetime_user_messages,
        EXCLUDED.lifetime_user_messages
    );
UPDATE engagement_state
SET lifetime_user_messages = GREATEST(
    lifetime_user_messages,
    total_messages::BIGINT
);
CREATE TABLE IF NOT EXISTS shared_content (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    kind TEXT NOT NULL,
    item_id TEXT NOT NULL,
    shared_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS demo_wallets (
    user_id BIGINT PRIMARY KEY,
    balance INTEGER NOT NULL CHECK (balance >= 0),
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS media_offers (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    content_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    action TEXT NOT NULL,
    request_type TEXT,
    description TEXT,
    batch_number BIGINT NOT NULL,
    price_tokens INTEGER NOT NULL CHECK (price_tokens > 0),
    status TEXT NOT NULL DEFAULT 'reserved'
        CHECK (status IN ('reserved', 'delivered', 'cancelled')),
    created_at DOUBLE PRECISION NOT NULL,
    offered_at DOUBLE PRECISION
);
ALTER TABLE media_offers ADD COLUMN IF NOT EXISTS description TEXT;
CREATE TABLE IF NOT EXISTS demo_token_transactions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    idempotency_key TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('initial_credit', 'refill', 'unlock')),
    amount_tokens INTEGER NOT NULL,
    balance_after INTEGER NOT NULL CHECK (balance_after >= 0),
    offer_id BIGINT,
    content_id TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    UNIQUE (user_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS media_entitlements (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    content_id TEXT NOT NULL,
    source_offer_id BIGINT NOT NULL,
    unlock_transaction_id BIGINT NOT NULL,
    unlocked_at DOUBLE PRECISION NOT NULL,
    first_opened_at DOUBLE PRECISION,
    UNIQUE (user_id, content_id)
);
CREATE TABLE IF NOT EXISTS media_tag_affinity (
    user_id BIGINT NOT NULL,
    tag_group TEXT NOT NULL,
    tag_value TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (user_id, tag_group, tag_value)
);
-- Ephemeral but durable direct-request confirmation.  The LLM never owns this
-- state: it receives only a trusted action after the backend has validated and
-- normalized the requested media facets.
CREATE TABLE IF NOT EXISTS media_request_confirmations (
    user_id BIGINT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('pending', 'granted')),
    requested_type TEXT CHECK (requested_type IN ('photo', 'video')),
    tags_json TEXT NOT NULL DEFAULT '{}',
    explicitness TEXT CHECK (explicitness IN ('tease', 'suggestive', 'nude', 'explicit')),
    asked_at_batch BIGINT NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
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
CREATE INDEX IF NOT EXISTS idx_media_offers_user_delivery
    ON media_offers(user_id, status, offered_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_offers_user_content
    ON media_offers(user_id, content_id, status, batch_number DESC);
-- Repair any crash-era duplicate live reservations before enforcing the
-- one-reservation-per-item invariant on an upgraded demo database.
WITH ranked_live_reservations AS (
    SELECT id, ROW_NUMBER() OVER (
        PARTITION BY user_id, content_id ORDER BY id DESC
    ) AS reservation_rank
    FROM media_offers WHERE status = 'reserved'
)
UPDATE media_offers SET status = 'cancelled'
WHERE id IN (
    SELECT id FROM ranked_live_reservations WHERE reservation_rank > 1
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_media_offers_one_live_reservation
    ON media_offers(user_id, content_id) WHERE status = 'reserved';
CREATE INDEX IF NOT EXISTS idx_media_entitlements_user
    ON media_entitlements(user_id, unlocked_at DESC);
CREATE INDEX IF NOT EXISTS idx_demo_token_transactions_user
    ON demo_token_transactions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_request_confirmations_expiry
    ON media_request_confirmations(expires_at);
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

    def transaction(self):
        """Return asyncpg's transaction context for atomic multi-query work."""
        return self._raw.transaction()

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


async def close_pool() -> None:
    """Close the shared pool (primarily for clean process/test shutdown)."""

    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        await pool.close()


async def init_db() -> None:
    pool = await _get_pool()
    async with pool.acquire() as conn:
        # Let PostgreSQL parse the complete script. Splitting on semicolons is
        # not SQL-aware and breaks on semicolons inside comments or literals.
        await conn.execute(SCHEMA)
