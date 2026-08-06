import base64
import logging
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlparse

import httpx

from bot.media_delivery import (
    MediaDeliveryError,
    R2DeliveryConfig,
    create_signed_get_url,
    create_signed_head_url,
    r2_object_exists,
    validate_media_delivery_config,
)


class MediaDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.config = R2DeliveryConfig(
            account_id="account123",
            access_key_id="access123",
            secret_access_key="secret123",
            bucket_name="mia-private",
        )
        self.now = datetime(2026, 8, 4, 12, 30, tzinfo=timezone.utc)

    def test_presigned_get_is_scoped_and_short_lived(self):
        url, expires_at = create_signed_get_url(
            "premium/mia bar 001.jpg",
            expires_seconds=600,
            now=self.now,
            delivery_config=self.config,
        )
        parsed = urlparse(url)
        query = parse_qs(parsed.query)

        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "account123.r2.cloudflarestorage.com")
        self.assertEqual(parsed.path, "/mia-private/premium/mia%20bar%20001.jpg")
        self.assertEqual(query["X-Amz-Algorithm"], ["AWS4-HMAC-SHA256"])
        self.assertEqual(query["X-Amz-Expires"], ["600"])
        self.assertEqual(query["X-Amz-SignedHeaders"], ["host"])
        self.assertEqual(len(query["X-Amz-Signature"][0]), 64)
        self.assertEqual(expires_at, self.now.timestamp() + 600)

    def test_signing_is_deterministic_for_same_instant(self):
        first = create_signed_get_url(
            "previews/mia_bar_001.webp",
            expires_seconds=3600,
            now=self.now,
            delivery_config=self.config,
        )
        second = create_signed_get_url(
            "previews/mia_bar_001.webp",
            expires_seconds=3600,
            now=self.now,
            delivery_config=self.config,
        )
        self.assertEqual(first, second)

    def test_head_probe_has_a_distinct_method_signature(self):
        get_url, _ = create_signed_get_url(
            "previews/mia_bar_001.webp",
            expires_seconds=60,
            now=self.now,
            delivery_config=self.config,
        )
        head_url, _ = create_signed_head_url(
            "previews/mia_bar_001.webp",
            expires_seconds=60,
            now=self.now,
            delivery_config=self.config,
        )
        self.assertNotEqual(
            parse_qs(urlparse(get_url).query)["X-Amz-Signature"],
            parse_qs(urlparse(head_url).query)["X-Amz-Signature"],
        )

    def test_rejects_urls_and_path_traversal(self):
        for unsafe in (
            "../private.jpg",
            "premium/../private.jpg",
            "https://example.com/private.jpg",
            "premium\\private.jpg",
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(MediaDeliveryError):
                create_signed_get_url(
                    unsafe,
                    expires_seconds=600,
                    now=self.now,
                    delivery_config=self.config,
                )

    def test_rejects_invalid_expiry(self):
        for expiry in (0, -1, 604_801, 10.0, True):
            with self.subTest(expiry=expiry), self.assertRaises(MediaDeliveryError):
                create_signed_get_url(
                    "premium/item.jpg",
                    expires_seconds=expiry,
                    now=self.now,
                    delivery_config=self.config,
                )

    def test_startup_config_rejects_invalid_media_ttl(self):
        configured = {
            "R2_ACCOUNT_ID": "account123",
            "R2_ACCESS_KEY_ID": "access123",
            "R2_SECRET_ACCESS_KEY": "secret123",
            "R2_BUCKET_NAME": "mia-private",
            "R2_SIGNED_PHOTO_TTL_SECONDS": 0,
            "R2_SIGNED_VIDEO_TTL_SECONDS": 3600,
            "R2_SIGNED_PREVIEW_TTL_SECONDS": 3600,
        }
        with patch.multiple("bot.media_delivery.config", **configured):
            with self.assertRaises(MediaDeliveryError):
                validate_media_delivery_config()

    def test_httpx_info_filter_drops_presigned_url_but_keeps_normal_logs(self):
        logger = logging.getLogger("httpx")
        with self.assertNoLogs("httpx", level="INFO"):
            logger.info(
                "HTTP Request: %s %s",
                "HEAD",
                "https://signed.example/object?X-Amz-Signature=secret",
            )
        with self.assertLogs("httpx", level="INFO") as captured:
            logger.info("HTTP Request: %s %s", "HEAD", "https://example.com/object")
        self.assertNotIn("secret", "\n".join(captured.output))


class R2HeadProbeTests(unittest.IsolatedAsyncioTestCase):
    async def probe(self, *, status_code=200, headers=None):
        response = Mock(status_code=status_code, headers=httpx.Headers(headers or {}))
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False
        client.head.return_value = response
        with patch(
            "bot.media_delivery.create_signed_head_url",
            return_value=("https://signed.example/head", 0),
        ), patch("httpx.AsyncClient", return_value=client):
            result = await r2_object_exists("premium/item.jpg")
        client.head.assert_awaited_once_with("https://signed.example/head")
        return result

    async def test_success_normalizes_r2_custom_metadata(self):
        checksum = "ABCDEF12" * 8
        result = await self.probe(
            headers={
                "Content-Type": "Image/JPEG",
                "Content-Length": "123",
                "X-Amz-Meta-Sha256": checksum,
            }
        )

        self.assertEqual(
            result,
            {
                "exists": True,
                "content_type": "image/jpeg",
                "content_length": 123,
                "sha256": checksum.lower(),
            },
        )

    async def test_full_object_s3_checksum_is_converted_from_base64(self):
        digest = bytes(range(32))
        result = await self.probe(
            headers={
                "content-type": "video/mp4",
                "content-length": "456",
                "x-amz-checksum-sha256": base64.b64encode(digest).decode("ascii"),
                "x-amz-checksum-type": "FULL_OBJECT",
            }
        )

        self.assertEqual(result["sha256"], digest.hex())

    async def test_missing_or_unusable_headers_are_fail_closed_values(self):
        result = await self.probe(
            headers={
                "content-length": "not-a-number",
                "x-amz-checksum-sha256": "not-base64",
            }
        )

        self.assertEqual(result["content_type"], "")
        self.assertEqual(result["content_length"], 0)
        self.assertEqual(result["sha256"], "")

    async def test_composite_checksum_requires_custom_raw_checksum_metadata(self):
        digest = base64.b64encode(bytes(range(32))).decode("ascii")
        result = await self.probe(
            headers={
                "content-type": "video/mp4",
                "content-length": "456",
                "x-amz-checksum-sha256": digest,
                "x-amz-checksum-type": "COMPOSITE",
            }
        )

        self.assertEqual(result["sha256"], "")

    async def test_missing_object_remains_false(self):
        self.assertIs(await self.probe(status_code=404), False)


if __name__ == "__main__":
    unittest.main()
