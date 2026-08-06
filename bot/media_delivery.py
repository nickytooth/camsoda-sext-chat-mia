"""Private Cloudflare R2 media delivery.

The chat and commerce layers deal only in catalog IDs.  This module is the
single boundary that turns a validated R2 object key into a short-lived S3
Signature V4 GET URL.  The URL is returned only by API serializers after their
authorization checks; it is never persisted in chat history or shown as text.

R2 implements the S3 presigned-URL protocol.  Keeping the small signer here
avoids a heavyweight boto3 dependency in the latency-sensitive chat service.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping
from urllib.parse import quote, urlencode

from bot import config


class _DropPresignedUrlLogFilter(logging.Filter):
    """Prevent HTTP client INFO logs from exposing signed query credentials."""

    _filters_r2_presigned_urls = True

    def filter(self, record: logging.LogRecord) -> bool:
        return "x-amz-signature=" not in repr(record.args).lower()


# HTTPX logs the full request URL at INFO after every response.  Keep its
# ordinary request diagnostics, but drop records containing any presigned URL
# before handlers/formatters can expose the signature or access-key scope.
_httpx_logger = logging.getLogger("httpx")
if not any(
    getattr(existing, "_filters_r2_presigned_urls", False)
    for existing in _httpx_logger.filters
):
    _httpx_logger.addFilter(_DropPresignedUrlLogFilter())


class MediaDeliveryError(RuntimeError):
    """Raised when private media delivery is unavailable or unsafe."""


def _sign(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


def _validate_object_key(object_key: str) -> str:
    """Accept a catalog object key, never an arbitrary URL or traversal path."""
    if not isinstance(object_key, str):
        raise MediaDeliveryError("Invalid media object key")
    key = object_key.strip().lstrip("/")
    if not key or "\\" in key or any(part in ("", ".", "..") for part in key.split("/")):
        raise MediaDeliveryError("Invalid media object key")
    if "://" in key or any(ord(char) < 32 for char in key):
        raise MediaDeliveryError("Invalid media object key")
    return key


@dataclass(frozen=True)
class R2DeliveryConfig:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str

    @classmethod
    def from_environment(cls) -> "R2DeliveryConfig":
        values = cls(
            account_id=getattr(config, "R2_ACCOUNT_ID", ""),
            access_key_id=getattr(config, "R2_ACCESS_KEY_ID", ""),
            secret_access_key=getattr(config, "R2_SECRET_ACCESS_KEY", ""),
            bucket_name=getattr(config, "R2_BUCKET_NAME", ""),
        )
        if not all(
            (values.account_id, values.access_key_id, values.secret_access_key, values.bucket_name)
        ):
            raise MediaDeliveryError("Cloudflare R2 is not configured")
        if any("/" in value or "\\" in value for value in (values.account_id, values.bucket_name)):
            raise MediaDeliveryError("Invalid Cloudflare R2 configuration")
        return values


def _create_signed_url(
    method: str,
    object_key: str,
    *,
    expires_seconds: int,
    now: datetime | None = None,
    delivery_config: R2DeliveryConfig | None = None,
) -> tuple[str, float]:
    """Return ``(url, expires_at_unix)`` for one private R2 request.

    S3-compatible presigned URLs allow at most seven days.  Commerce uses much
    shorter lifetimes (10 minutes for photos and 60 minutes for videos).
    """
    normalized_method = str(method).upper()
    if normalized_method not in {"GET", "HEAD"}:
        raise MediaDeliveryError("Unsupported signed media method")
    if type(expires_seconds) is not int or not 1 <= expires_seconds <= 604_800:
        raise MediaDeliveryError("Invalid signed media lifetime")

    cfg = delivery_config or R2DeliveryConfig.from_environment()
    key = _validate_object_key(object_key)
    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    amz_date = instant.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = instant.strftime("%Y%m%d")
    region = "auto"
    service = "s3"
    scope = f"{date_stamp}/{region}/{service}/aws4_request"

    host = f"{cfg.account_id}.r2.cloudflarestorage.com"
    # Quote each path segment while preserving slashes. Bucket names are
    # constrained above and catalog keys are validated before signing.
    canonical_uri = "/" + quote(f"{cfg.bucket_name}/{key}", safe="/-_.~")
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{cfg.access_key_id}/{scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires_seconds),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = urlencode(sorted(query.items()), quote_via=quote, safe="-_.~")
    canonical_headers = f"host:{host}\n"
    canonical_request = "\n".join(
        (
            normalized_method,
            canonical_uri,
            canonical_query,
            canonical_headers,
            "host",
            "UNSIGNED-PAYLOAD",
        )
    )
    string_to_sign = "\n".join(
        (
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        )
    )
    date_key = _sign(("AWS4" + cfg.secret_access_key).encode("utf-8"), date_stamp)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, service)
    signing_key = _sign(service_key, "aws4_request")
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    url = f"https://{host}{canonical_uri}?{canonical_query}&X-Amz-Signature={signature}"
    return url, instant.timestamp() + expires_seconds


def create_signed_get_url(
    object_key: str,
    *,
    expires_seconds: int,
    now: datetime | None = None,
    delivery_config: R2DeliveryConfig | None = None,
) -> tuple[str, float]:
    """Return a short-lived GET source for a private photo/video derivative."""
    return _create_signed_url(
        "GET",
        object_key,
        expires_seconds=expires_seconds,
        now=now,
        delivery_config=delivery_config,
    )


def create_signed_head_url(
    object_key: str,
    *,
    expires_seconds: int = 60,
    now: datetime | None = None,
    delivery_config: R2DeliveryConfig | None = None,
) -> tuple[str, float]:
    """Return a short-lived HEAD URL used by deployment catalog validation."""
    return _create_signed_url(
        "HEAD",
        object_key,
        expires_seconds=expires_seconds,
        now=now,
        delivery_config=delivery_config,
    )


def r2_is_configured() -> bool:
    try:
        R2DeliveryConfig.from_environment()
    except MediaDeliveryError:
        return False
    return True


def _validated_ttl(value: object, *, label: str) -> int:
    if type(value) is not int or not 1 <= value <= 604_800:
        raise MediaDeliveryError(f"{label} must be between 1 and 604800 seconds")
    return value


def validate_media_delivery_config() -> R2DeliveryConfig:
    """Validate credentials and every TTL before commerce can take payment."""
    delivery_config = R2DeliveryConfig.from_environment()
    full_media_ttl("photo")
    full_media_ttl("video")
    preview_media_ttl()
    return delivery_config


def _head_content_length(headers: Mapping[str, str]) -> int:
    """Return a positive HEAD content length, or zero for invalid metadata."""
    raw_length = str(headers.get("content-length", "")).strip()
    try:
        length = int(raw_length)
    except (TypeError, ValueError):
        return 0
    return length if length > 0 else 0


def _head_sha256(headers: Mapping[str, str]) -> str:
    """Normalize an R2/S3 SHA-256 response header to lowercase hex.

    R2 custom metadata is exposed by its S3 API as ``x-amz-meta-*``.  Assets
    should therefore be uploaded with a raw hexadecimal checksum in
    ``x-amz-meta-sha256``.  We also accept ``x-amz-meta-checksum-sha256`` for
    upload-tool compatibility.

    A standard S3 ``x-amz-checksum-sha256`` response is Base64 encoded.  It is
    safe to compare with the catalog's raw file checksum only when S3 marks it
    as a full-object checksum (or omits the checksum type, as older single-part
    implementations do).  Composite multipart checksums deliberately fail
    closed and require the custom raw checksum metadata instead.
    """
    for header_name in ("x-amz-meta-sha256", "x-amz-meta-checksum-sha256"):
        checksum = str(headers.get(header_name, "")).strip()
        if checksum:
            return checksum.lower()

    encoded_checksum = str(headers.get("x-amz-checksum-sha256", "")).strip()
    checksum_type = str(headers.get("x-amz-checksum-type", "")).strip().upper()
    if not encoded_checksum or checksum_type not in {"", "FULL_OBJECT"}:
        return ""
    try:
        digest = base64.b64decode(encoded_checksum, validate=True)
    except (binascii.Error, ValueError):
        return ""
    return digest.hex() if len(digest) == hashlib.sha256().digest_size else ""


async def r2_object_exists(object_key: str) -> bool | dict[str, object]:
    """Probe one configured R2 key and return normalized HEAD metadata.

    Returning a mapping for existing objects makes startup catalog validation
    fail closed on MIME type, object size and checksum instead of checking only
    that a key exists.  Missing objects retain the historical ``False`` result.
    """
    import httpx

    url, _ = create_signed_head_url(object_key)
    async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
        response = await client.head(url)
    if response.status_code in {200, 204}:
        headers = response.headers
        return {
            "exists": True,
            "content_type": str(headers.get("content-type", "")).strip().lower(),
            "content_length": _head_content_length(headers),
            "sha256": _head_sha256(headers),
        }
    if response.status_code == 404:
        return False
    raise MediaDeliveryError(
        f"Cloudflare R2 object validation failed with HTTP {response.status_code}"
    )


def full_media_ttl(media_type: str) -> int:
    if media_type == "photo":
        return _validated_ttl(
            getattr(config, "R2_SIGNED_PHOTO_TTL_SECONDS", 600),
            label="R2_SIGNED_PHOTO_TTL_SECONDS",
        )
    if media_type == "video":
        return _validated_ttl(
            getattr(config, "R2_SIGNED_VIDEO_TTL_SECONDS", 3600),
            label="R2_SIGNED_VIDEO_TTL_SECONDS",
        )
    raise MediaDeliveryError("Unsupported media type")


def preview_media_ttl() -> int:
    # Locked previews are harmless derivatives but still live in the private
    # bucket.  A longer TTL prevents a card from going blank during a session.
    return _validated_ttl(
        getattr(config, "R2_SIGNED_PREVIEW_TTL_SECONDS", 3600),
        label="R2_SIGNED_PREVIEW_TTL_SECONDS",
    )
