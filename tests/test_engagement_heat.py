import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from bot.engagement import track_heat_batch


class _Cursor:
    def __init__(self, rows=()):
        self.rows = list(rows)

    async def fetchone(self):
        return self.rows[0] if self.rows else None


class _Transaction:
    def __init__(self, lock):
        self.lock = lock

    async def __aenter__(self):
        await self.lock.acquire()

    async def __aexit__(self, exc_type, exc, traceback):
        self.lock.release()
        return False


class _SharedEngagementStore:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.rows = {}

    def connection(self):
        return _FakeConnection(self)


class _FakeConnection:
    def __init__(self, store):
        self.store = store

    def transaction(self):
        return _Transaction(self.store.lock)

    async def execute(self, sql, params=()):
        collapsed = " ".join(sql.split())
        user_id = params[-1] if params else None
        if collapsed.startswith("INSERT INTO engagement_state"):
            user_id = params[0]
            self.store.rows.setdefault(
                user_id,
                {
                    "user_id": user_id,
                    "nsfw_count": 0,
                    "total_messages": 0,
                    "last_message_at": 0,
                    "first_message_at": None,
                    "active_days": 0,
                    "last_active_date": None,
                    "heat_stage": "low",
                    "heat_progress": 0,
                    "heat_last_sexual_at": 0,
                    "heat_updated_at": 0,
                    "heat_last_batch": 0,
                    "heat_last_signal": "neutral",
                    "sexual_pause_active": False,
                    "heat_blocked_acts": "[]",
                },
            )
            return _Cursor()
        if collapsed.startswith("SELECT * FROM engagement_state"):
            return _Cursor((dict(self.store.rows[params[0]]),))
        if collapsed.startswith("UPDATE engagement_state"):
            row = self.store.rows[user_id]
            row["nsfw_count"] += params[0]
            row["total_messages"] = max(
                row["total_messages"], row["heat_last_batch"]
            ) + 1
            row["last_message_at"] = params[1]
            row["first_message_at"] = row["first_message_at"] or params[2]
            if row["last_active_date"] != params[3]:
                row["active_days"] += 1
            row["last_active_date"] = params[4]
            row["heat_stage"] = params[5]
            row["heat_progress"] = params[6]
            row["heat_last_sexual_at"] = params[7]
            row["heat_updated_at"] = params[8]
            row["heat_last_batch"] = params[9]
            row["heat_last_signal"] = params[10]
            row["sexual_pause_active"] = params[11]
            row["heat_blocked_acts"] = params[12]
            return _Cursor()
        raise AssertionError(f"unexpected SQL: {collapsed}")

    async def commit(self):
        return None

    async def close(self):
        return None


class AtomicHeatPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_workers_receive_unique_batches_and_latest_state(self):
        store = _SharedEngagementStore()

        async def connection():
            return store.connection()

        with patch("bot.engagement.get_connection", new=AsyncMock(side_effect=connection)):
            results = await asyncio.gather(
                track_heat_batch(77, ["fuck me"], now=10),
                track_heat_batch(77, ["fuck me"], now=20),
            )

        self.assertEqual({batch for _, batch in results}, {1, 2})
        self.assertEqual(store.rows[77]["total_messages"], 2)
        self.assertEqual(store.rows[77]["heat_progress"], 2)
        self.assertEqual(store.rows[77]["heat_last_batch"], 2)
        self.assertEqual(store.rows[77]["nsfw_count"], 2)
        self.assertEqual(json.loads(store.rows[77]["heat_blocked_acts"]), [])

    async def test_heat_batch_counter_is_never_behind_persisted_heat_version(self):
        store = _SharedEngagementStore()
        connection = store.connection()
        await connection.execute(
            "INSERT INTO engagement_state (user_id) VALUES (?)",
            (88,),
        )
        store.rows[88]["total_messages"] = 2
        store.rows[88]["heat_last_batch"] = 5

        async def get_connection():
            return store.connection()

        with patch(
            "bot.engagement.get_connection",
            new=AsyncMock(side_effect=get_connection),
        ):
            result, batch = await track_heat_batch(88, ["hello"], now=10)

        self.assertEqual(batch, 6)
        self.assertEqual(result.state.last_batch, 6)
        self.assertEqual(store.rows[88]["total_messages"], 6)

    async def test_validated_media_request_is_persisted_as_high_in_one_batch(self):
        store = _SharedEngagementStore()

        async def get_connection():
            return store.connection()

        with patch(
            "bot.engagement.get_connection",
            new=AsyncMock(side_effect=get_connection),
        ):
            result, batch = await track_heat_batch(
                99,
                ["can you send me a picture?"],
                now=10,
                direct_media_request=True,
            )

        self.assertEqual(batch, 1)
        self.assertEqual((result.state.stage, result.state.progress), ("high", 3))
        self.assertEqual(store.rows[99]["heat_stage"], "high")
        self.assertEqual(store.rows[99]["heat_progress"], 3)


if __name__ == "__main__":
    unittest.main()
