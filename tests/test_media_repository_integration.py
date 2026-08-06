"""Opt-in PostgreSQL transaction tests.

Run with ``RUN_DB_INTEGRATION=1 python -m unittest
tests.test_media_repository_integration`` against the disposable demo DB.
"""

import asyncio
import os
import time
import unittest

from bot.media_repository import MediaRepository, MediaUnavailableError
from bot.memory.db import close_pool, init_db


@unittest.skipUnless(
    os.getenv("RUN_DB_INTEGRATION") == "1", "set RUN_DB_INTEGRATION=1 for PostgreSQL tests"
)
class MediaRepositoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await init_db()
        self.repository = MediaRepository()
        self.user_id = 8_500_000_000 + int(time.time() * 1000) % 100_000_000
        await self.repository.reset_user_commerce(self.user_id)

    async def asyncTearDown(self):
        try:
            await self.repository.reset_user_commerce(self.user_id)
        finally:
            await close_pool()

    async def test_double_unlock_debits_once_and_photo_video_prices_are_snapshots(self):
        self.assertEqual((await self.repository.get_wallet(self.user_id)).balance, 1000)
        photo = await self.repository.reserve_offer(
            self.user_id,
            "integration_photo",
            trigger="direct",
            action="offer_current",
            request_type="photo",
            description="integration photo",
            batch_number=1,
            price_tokens=5,
        )
        await self.repository.mark_offer_delivered(photo.offer_id, media_type="photo")
        first, second = await asyncio.gather(
            self.repository.unlock_offer(self.user_id, photo.offer_id, "photo-click-a"),
            self.repository.unlock_offer(self.user_id, photo.offer_id, "photo-click-b"),
        )
        self.assertEqual(sorted((first.charged_tokens, second.charged_tokens)), [0, 5])
        self.assertEqual((await self.repository.get_wallet(self.user_id)).balance, 995)

        with self.assertRaises(MediaUnavailableError):
            await self.repository.reserve_offer(
                self.user_id,
                "integration_photo",
                trigger="direct",
                action="offer_current",
                request_type="photo",
                description="must never be resold",
                batch_number=2,
                price_tokens=5,
            )

        video = await self.repository.reserve_offer(
            self.user_id,
            "integration_video",
            trigger="direct",
            action="offer_fallback",
            request_type="video",
            description="integration video",
            batch_number=3,
            price_tokens=10,
        )
        await self.repository.mark_offer_delivered(video.offer_id, media_type="video")
        await self.repository.unlock_offer(self.user_id, video.offer_id, "video-click")
        self.assertEqual((await self.repository.get_wallet(self.user_id)).balance, 985)
        self.assertEqual(len(await self.repository.list_entitlements(self.user_id)), 2)

    async def test_pending_media_confirmation_is_consumed_once(self):
        now = time.time()
        staged = await self.repository.stage_media_confirmation(
            self.user_id,
            requested_type="photo",
            tags={"body_focus": ("pussy",)},
            explicitness="nude",
            asked_at_batch=1,
            expires_at=now + 600,
            now=now,
        )
        self.assertEqual(staged.status, "pending")

        async def grant():
            return await self.repository.grant_media_confirmation(
                self.user_id,
                batch_number=2,
                max_batch_gap=4,
                granted_until=now + 3600,
                now=now + 1,
            )

        first, second = await asyncio.gather(grant(), grant())
        self.assertEqual(sum(value is not None for value in (first, second)), 1)
        persisted = await self.repository.get_media_confirmation(
            self.user_id, now=now + 2
        )
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.status, "granted")
        self.assertEqual(persisted.tags["body_focus"], ("pussy",))


if __name__ == "__main__":
    unittest.main()
