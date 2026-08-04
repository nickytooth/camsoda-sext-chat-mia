import unittest
from unittest.mock import AsyncMock, patch

from server.app import ConnectionManager, _record_card_user_turn


class ConnectionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_disconnect_cannot_remove_newer_socket(self):
        manager = ConnectionManager()
        old_socket = AsyncMock()
        new_socket = AsyncMock()

        await manager.connect(7, old_socket)
        await manager.connect(7, new_socket)

        old_socket.close.assert_awaited_once()
        self.assertIs(manager.active[7], new_socket)

        manager.disconnect(7, old_socket)
        self.assertIs(manager.active[7], new_socket)

        manager.disconnect(7, new_socket)
        self.assertNotIn(7, manager.active)

    async def test_card_request_advances_raw_and_processed_turn_once(self):
        with (
            patch("bot.engagement.record_user_message", new=AsyncMock()) as raw,
            patch("bot.engagement.track_message", new=AsyncMock()) as processed,
            patch("bot.router.classify_fast", return_value="nsfw"),
        ):
            await _record_card_user_turn(8, "tell me a fantasy")

        raw.assert_awaited_once_with(8)
        processed.assert_awaited_once_with(8, "nsfw")


if __name__ == "__main__":
    unittest.main()
