import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException, Response


server_app = importlib.import_module("server.app")


class _Catalog:
    def __init__(self):
        self.item = SimpleNamespace(
            id="mia_home_001",
            media_type="photo",
            preview_key="previews/mia_home_001.webp",
            poster_key=None,
            full_key="premium/mia_home_001.jpg",
            mime_type="image/jpeg",
            aspect_ratio=0.75,
            duration_seconds=None,
        )

    def get(self, content_id):
        return self.item if content_id == self.item.id else None


class _Service:
    def __init__(self):
        self.catalog = _Catalog()
        self.has_entitlement = AsyncMock(return_value=False)
        self.list_delivered_offers = AsyncMock(return_value=[])
        self.unlock_offer = AsyncMock()


class MediaApiContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.old_service = server_app.commerce_service
        self.old_catalog = server_app.media_catalog
        self.old_ready = server_app.commerce_delivery_ready
        self.service = _Service()
        server_app.commerce_service = self.service
        server_app.media_catalog = self.service.catalog
        server_app.commerce_delivery_ready = False

    def tearDown(self):
        server_app.commerce_service = self.old_service
        server_app.media_catalog = self.old_catalog
        server_app.commerce_delivery_ready = self.old_ready

    async def test_history_is_a_stable_mixed_feed_without_storage_keys(self):
        self.service.list_delivered_offers.return_value = [
            SimpleNamespace(
                offer_id=17,
                content_id="mia_home_001",
                price_tokens=5,
                offered_at=2.0,
            )
        ]
        messages = [
            {"id": 1, "role": "user", "content": "show me", "timestamp": 1.0},
            {"id": 2, "role": "assistant", "content": "maybe this", "timestamp": 3.0},
        ]
        with patch.object(
            server_app,
            "get_all_messages",
            new=AsyncMock(return_value=messages),
        ):
            response = Response()
            result = await server_app.get_history("sexting", response, user_id=9)

        self.assertEqual(
            [item["type"] for item in result["messages"]],
            ["text", "media_offer", "text"],
        )
        media = result["messages"][1]
        self.assertEqual(media["id"], "media-offer-17")
        self.assertEqual(media["offer"]["price_tokens"], 5)
        self.assertEqual(media["offer"]["preview_url"], "")
        wire_text = repr(result)
        self.assertNotIn("preview_key", wire_text)
        self.assertNotIn("premium/", wire_text)
        self.assertEqual(response.headers["cache-control"], "private, no-store")

    async def test_unlock_contract_contains_no_media_source(self):
        self.service.unlock_offer.return_value = SimpleNamespace(
            balance=995,
            content_id="mia_home_001",
            already_unlocked=False,
        )
        request = server_app.IdempotencyRequest(idempotency_key="unlock-123456")
        with (
            patch.object(server_app, "_require_media_delivery", return_value=None),
            patch.object(server_app.manager, "send_json", new=AsyncMock()) as send,
        ):
            result = await server_app.unlock_media_offer("17", request, user_id=9)

        self.assertEqual(result["balance"], 995)
        self.assertEqual(
            result["entitlement"],
            {"content_id": "mia_home_001", "media_type": "photo"},
        )
        self.assertNotIn("access_url", result)
        send.assert_awaited_once()

    async def test_unlock_success_survives_websocket_disconnect(self):
        self.service.unlock_offer.return_value = SimpleNamespace(
            balance=995,
            content_id="mia_home_001",
            already_unlocked=False,
        )
        request = server_app.IdempotencyRequest(idempotency_key="unlock-654321")
        with (
            patch.object(server_app, "_require_media_delivery", return_value=None),
            patch.object(
                server_app.manager,
                "send_json",
                new=AsyncMock(side_effect=RuntimeError("socket closed")),
            ),
        ):
            result = await server_app.unlock_media_offer("17", request, user_id=9)
        self.assertEqual(result["balance"], 995)

    async def test_locked_full_access_is_forbidden(self):
        with patch.object(server_app, "_require_media_delivery", return_value=None):
            with self.assertRaises(HTTPException) as caught:
                await server_app.get_media_access(
                    "mia_home_001", Response(), user_id=9
                )
        self.assertEqual(caught.exception.status_code, 403)

    def test_insufficient_balance_maps_to_payment_required(self):
        error = type("Insufficient", (RuntimeError,), {"code": "insufficient_tokens"})(
            "need more tokens"
        )
        mapped = server_app._commerce_http_error(error)
        self.assertEqual(mapped.status_code, 402)

    def test_zero_user_id_is_not_silently_mapped_to_default_user(self):
        with self.assertRaises(HTTPException) as caught:
            server_app._commerce_user_id(0)
        self.assertEqual(caught.exception.status_code, 422)

    async def test_normal_chat_reset_preserves_commerce_and_batch_state(self):
        connection = AsyncMock()
        transaction = AsyncMock()
        transaction.__aenter__.return_value = None
        transaction.__aexit__.return_value = False
        connection.transaction = Mock(return_value=transaction)
        old_engine = server_app.engine
        server_app.engine = None
        try:
            with patch.object(
                server_app,
                "get_connection",
                new=AsyncMock(return_value=connection),
            ):
                await server_app.reset_user(user_id=9)
        finally:
            server_app.engine = old_engine

        sql = "\n".join(str(call.args[0]) for call in connection.execute.await_args_list)
        self.assertNotIn("DELETE FROM engagement_state", sql)
        self.assertNotIn("DELETE FROM demo_wallets", sql)
        self.assertNotIn("DELETE FROM media_offers", sql)
        self.assertNotIn("DELETE FROM media_entitlements", sql)
        self.assertIn("DELETE FROM media_request_confirmations", sql)
        self.assertIn("UPDATE engagement_state", sql)
        self.assertNotIn("total_messages =", sql)


if __name__ == "__main__":
    unittest.main()
