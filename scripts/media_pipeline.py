"""Prepare private visual-commerce assets and publish them to Cloudflare R2.

The authored manifest contains only semantic metadata and a local source name.
This pipeline creates privacy-stripped canonical full files, genuinely separate
degraded WebP derivatives, deterministic catalog metadata and R2 uploads. The
runtime catalog is replaced only after every object in the resulting catalog
passes the same fail-closed validation used during backend startup.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MANIFEST = PROJECT_ROOT / ".private-media" / "manifest.yaml"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / ".private-media" / "originals"
DEFAULT_BUILD_DIR = PROJECT_ROOT / ".media-build"
DEFAULT_CATALOG = PROJECT_ROOT / "library" / "media_catalog.yaml"
DEFAULT_PUBLIC_DIR = PROJECT_ROOT / "frontend" / "public"

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")
_MANIFEST_FIELDS = frozenset(
    {
        "id",
        "source",
        "type",
        "status",
        "explicitness",
        "min_heat",
        "presentation",
        "tags",
        "preview_time_seconds",
        "poster_time_seconds",
    }
)
_IMAGE_FORMATS: Mapping[str, tuple[str, str]] = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
}
_MAX_SINGLE_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024
_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"})
_VIDEO_FINGERPRINT_FRACTIONS = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


class MediaPipelineError(RuntimeError):
    """Raised when preparation or publication cannot complete safely."""


@dataclass(frozen=True, slots=True)
class PreparedObject:
    key: str
    path: Path
    mime_type: str
    sha256: str
    role: str


@dataclass(frozen=True, slots=True)
class PreparedRun:
    run_dir: Path
    candidate_catalog: Path
    objects: tuple[PreparedObject, ...]
    base_catalog_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml_snapshot(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise MediaPipelineError(f"Cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MediaPipelineError(f"{label} root must be a mapping")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    payload, _ = _load_yaml_snapshot(path, label=label)
    return payload


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            dict(payload),
            sort_keys=False,
            allow_unicode=True,
            width=110,
        ),
        encoding="utf-8",
    )


def _resolve_source(source_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MediaPipelineError("Each manifest item needs a non-empty source")
    relative = Path(value.strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise MediaPipelineError(f"Source must stay under the source directory: {value}")
    root = source_root.resolve()
    unresolved = root / relative
    current = unresolved
    while current != root:
        if current.is_symlink():
            raise MediaPipelineError(f"Source paths cannot contain symlinks: {value}")
        current = current.parent
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MediaPipelineError(f"Source escapes the source directory: {value}") from exc
    if not candidate.is_file():
        raise MediaPipelineError(f"Source file does not exist: {value}")
    return candidate


def _fingerprint_opened_image(opened: Any) -> tuple[int, tuple[int, int, int]]:
    from PIL import ImageOps

    normalized = ImageOps.exif_transpose(opened).convert("RGB")
    image = normalized.convert("L").resize((9, 8))
    pixels = list(image.getdata())
    color_sample = normalized.resize((16, 16))
    color_pixels = list(color_sample.getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            bits = (bits << 1) | int(left > right)
    average_color = tuple(
        sum(pixel[channel] for pixel in color_pixels) // len(color_pixels)
        for channel in range(3)
    )
    return bits, average_color


def _photo_fingerprint(path: Path) -> tuple[int, tuple[int, int, int]] | None:
    try:
        from PIL import Image

        with Image.open(path) as opened:
            return _fingerprint_opened_image(opened)
    except Exception:
        return None


def _photo_fingerprint_bytes(data: bytes) -> tuple[int, tuple[int, int, int]]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as opened:
            return _fingerprint_opened_image(opened)
    except Exception as exc:
        raise MediaPipelineError("Could not fingerprint an extracted video frame") from exc


def _public_asset_index(
    public_dir: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
) -> tuple[set[str], tuple[tuple[int, tuple[int, int, int]], ...]]:
    exact: set[str] = set()
    photos: list[tuple[int, tuple[int, int, int]]] = []
    if not public_dir.is_dir():
        return exact, ()
    for path in public_dir.rglob("*"):
        if not path.is_file():
            continue
        exact.add(sha256_file(path))
        fingerprint = _photo_fingerprint(path)
        if fingerprint is not None:
            photos.append(fingerprint)
        elif path.suffix.lower() in _VIDEO_EXTENSIONS:
            photos.extend(
                _video_frame_fingerprints(path, ffmpeg=ffmpeg, ffprobe=ffprobe)
            )
    return exact, tuple(photos)


def _assert_private_source(
    source: Path,
    *,
    media_type: str,
    public_dir: Path,
    public_hashes: set[str],
    public_photo_fingerprints: Sequence[tuple[int, tuple[int, int, int]]],
    ffmpeg: str,
    ffprobe: str,
) -> None:
    try:
        source.resolve().relative_to(public_dir.resolve())
    except ValueError:
        pass
    else:
        raise MediaPipelineError(
            f"Paid source {source.name} is inside frontend/public and is not private"
        )
    if sha256_file(source) in public_hashes:
        raise MediaPipelineError(
            f"Paid source {source.name} is byte-for-byte identical to a public asset"
        )
    if media_type == "photo":
        fingerprint = _photo_fingerprint(source)
        fingerprints = (fingerprint,) if fingerprint is not None else ()
    else:
        fingerprints = _video_frame_fingerprints(
            source, ffmpeg=ffmpeg, ffprobe=ffprobe
        )
    for fingerprint in fingerprints:
        if any(
            (fingerprint[0] ^ public[0]).bit_count() <= 4
            and sum(abs(a - b) for a, b in zip(fingerprint[1], public[1])) <= 45
            for public in public_photo_fingerprints
        ):
            raise MediaPipelineError(
                f"Paid source {source.name} contains imagery already available publicly"
            )


def _degraded_photo(source_image: Any, target: Path) -> None:
    from PIL import Image, ImageFilter

    image = source_image.convert("RGB")
    image.thumbnail((420, 640), Image.Resampling.LANCZOS)
    tiny = image.resize(
        (max(18, image.width // 12), max(18, image.height // 12)),
        Image.Resampling.BILINEAR,
    )
    image = tiny.resize(image.size, Image.Resampling.BILINEAR)
    image = image.filter(ImageFilter.GaussianBlur(radius=max(7, min(image.size) / 28)))
    image = Image.blend(image, Image.new("RGB", image.size, "black"), 0.20)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="WEBP", quality=36, method=6, exif=b"")


def _prepare_photo(content_id: str, source: Path, run_dir: Path) -> tuple[dict[str, Any], list[PreparedObject]]:
    try:
        from PIL import Image, ImageOps

        with Image.open(source) as opened:
            if bool(getattr(opened, "is_animated", False)):
                raise MediaPipelineError(f"Animated photo sources are unsupported: {source.name}")
            source_format = str(opened.format or "").upper()
            if source_format not in _IMAGE_FORMATS:
                raise MediaPipelineError(
                    f"Photo {source.name} must be JPEG, PNG or WebP"
                )
            image = ImageOps.exif_transpose(opened)
            image.load()
    except MediaPipelineError:
        raise
    except Exception as exc:
        raise MediaPipelineError(f"Cannot decode photo {source.name}: {exc}") from exc

    extension, mime_type = _IMAGE_FORMATS[source_format]
    full_path = run_dir / "full" / f"{content_id}{extension}"
    preview_path = run_dir / "previews" / f"{content_id}.webp"
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if source_format == "JPEG":
        image.convert("RGB").save(
            full_path,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=True,
            exif=b"",
        )
    elif source_format == "PNG":
        normalized = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        normalized.save(full_path, format="PNG", optimize=True)
    else:
        image.convert("RGB").save(
            full_path,
            format="WEBP",
            quality=95,
            method=6,
            exif=b"",
        )

    _degraded_photo(image, preview_path)
    full_hash = sha256_file(full_path)
    preview_hash = sha256_file(preview_path)
    if full_hash == preview_hash:
        raise MediaPipelineError(f"Preview for {content_id} is not a separate derivative")
    generated = {
        "full_key": f"premium/{content_id}/{full_hash}{extension}",
        "preview_key": f"previews/{content_id}/{preview_hash}.webp",
        "poster_key": None,
        "mime_type": mime_type,
        "aspect_ratio": round(image.width / image.height, 6),
        "duration_seconds": None,
        "sha256": full_hash,
    }
    objects = [
        PreparedObject(generated["full_key"], full_path, mime_type, full_hash, "full"),
        PreparedObject(
            generated["preview_key"], preview_path, "image/webp", preview_hash, "preview"
        ),
    ]
    return generated, objects


def _run(command: Sequence[str], *, label: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise MediaPipelineError(f"Cannot run {label}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()[-1200:]
        raise MediaPipelineError(f"{label} failed: {detail}")
    return result


def _run_binary(command: Sequence[str], *, label: str) -> bytes:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise MediaPipelineError(f"Cannot run {label}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"unknown error")[-1200:].decode(
            "utf-8", errors="replace"
        )
        raise MediaPipelineError(f"{label} failed: {detail.strip()}")
    if not result.stdout:
        raise MediaPipelineError(f"{label} produced no frame data")
    return result.stdout


def _probe_video(path: Path, *, ffprobe: str) -> tuple[int, int, float]:
    result = _run(
        (
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ),
        label="ffprobe",
    )
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        duration = float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaPipelineError(f"Could not read video metadata for {path.name}") from exc
    if width <= 0 or height <= 0 or duration <= 0:
        raise MediaPipelineError(f"Invalid video dimensions or duration for {path.name}")
    return width, height, duration


def _video_frame_fingerprints(
    path: Path,
    *,
    ffmpeg: str,
    ffprobe: str,
) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    """Sample video imagery in memory; no extracted frame is written to disk."""
    _, _, duration = _probe_video(path, ffprobe=ffprobe)
    fingerprints: list[tuple[int, tuple[int, int, int]]] = []
    for fraction in _VIDEO_FINGERPRINT_FRACTIONS:
        # Container duration commonly extends a fraction beyond the last
        # decodable frame. Stay comfortably inside the media timeline.
        timestamp = min(
            duration * fraction,
            max(0.0, duration - max(0.15, duration * 0.05)),
        )
        frame = _run_binary(
            (
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                "scale=256:256:force_original_aspect_ratio=decrease",
                "-f",
                "image2pipe",
                "-c:v",
                "png",
                "pipe:1",
            ),
            label=f"video privacy fingerprint for {path.name}",
        )
        fingerprints.append(_photo_fingerprint_bytes(frame))
    if not fingerprints:
        raise MediaPipelineError(f"Could not fingerprint video {path.name}")
    return tuple(fingerprints)


def _frame_time(value: object, *, duration: float, fallback_fraction: float, field: str) -> float:
    if value is None:
        chosen = duration * fallback_fraction
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaPipelineError(f"{field} must be a number")
    else:
        chosen = float(value)
    if chosen < 0 or chosen >= duration:
        raise MediaPipelineError(f"{field} must be within the video duration")
    return min(chosen, max(0.0, duration - 0.05))


def _generate_video_derivative(
    full_path: Path,
    target: Path,
    *,
    timestamp: float,
    ffmpeg: str,
    blur: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    video_filter = (
        "scale=360:-2:force_original_aspect_ratio=decrease,"
        f"gblur=sigma={blur},eq=brightness=-0.16:saturation=0.65"
    )
    _run(
        (
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(full_path),
            "-frames:v",
            "1",
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            "libwebp",
            "-quality",
            "38",
            "-compression_level",
            "6",
            str(target),
        ),
        label="video derivative generation",
    )


def _prepare_video(
    content_id: str,
    source: Path,
    run_dir: Path,
    manifest_item: Mapping[str, Any],
    *,
    ffmpeg: str,
    ffprobe: str,
) -> tuple[dict[str, Any], list[PreparedObject]]:
    full_path = run_dir / "full" / f"{content_id}.mp4"
    preview_path = run_dir / "previews" / f"{content_id}.webp"
    poster_path = run_dir / "posters" / f"{content_id}.webp"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        (
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map_metadata",
            "-1",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(full_path),
        ),
        label="video normalization",
    )
    width, height, duration = _probe_video(full_path, ffprobe=ffprobe)
    preview_time = _frame_time(
        manifest_item.get("preview_time_seconds"),
        duration=duration,
        fallback_fraction=0.25,
        field="preview_time_seconds",
    )
    poster_time = _frame_time(
        manifest_item.get("poster_time_seconds"),
        duration=duration,
        fallback_fraction=0.55,
        field="poster_time_seconds",
    )
    _generate_video_derivative(
        full_path, preview_path, timestamp=preview_time, ffmpeg=ffmpeg, blur=20
    )
    _generate_video_derivative(
        full_path, poster_path, timestamp=poster_time, ffmpeg=ffmpeg, blur=16
    )
    full_hash = sha256_file(full_path)
    preview_hash = sha256_file(preview_path)
    poster_hash = sha256_file(poster_path)
    if len({full_hash, preview_hash, poster_hash}) != 3:
        raise MediaPipelineError(f"Video derivatives for {content_id} are not separate files")
    generated = {
        "full_key": f"premium/{content_id}/{full_hash}.mp4",
        "preview_key": f"previews/{content_id}/{preview_hash}.webp",
        "poster_key": f"posters/{content_id}/{poster_hash}.webp",
        "mime_type": "video/mp4",
        "aspect_ratio": round(width / height, 6),
        "duration_seconds": round(duration, 6),
        "sha256": full_hash,
    }
    objects = [
        PreparedObject(generated["full_key"], full_path, "video/mp4", full_hash, "full"),
        PreparedObject(
            generated["preview_key"], preview_path, "image/webp", preview_hash, "preview"
        ),
        PreparedObject(
            generated["poster_key"], poster_path, "image/webp", poster_hash, "poster"
        ),
    ]
    return generated, objects


def _merge_catalog(existing: Mapping[str, Any], generated_items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if existing.get("version") != 1 or not isinstance(existing.get("items"), list):
        raise MediaPipelineError("Existing runtime catalog must have version: 1 and an items list")
    merged = [dict(item) for item in existing["items"]]
    positions = {
        str(item.get("id")): index
        for index, item in enumerate(merged)
        if isinstance(item, dict)
    }
    for generated in generated_items:
        content_id = generated["id"]
        existing_index = positions.get(content_id)
        if existing_index is None:
            positions[content_id] = len(merged)
            merged.append(generated)
            continue
        previous = merged[existing_index]
        if str(previous.get("sha256", "")).lower() != generated["sha256"]:
            raise MediaPipelineError(
                f"Refusing to replace bytes for existing content ID {content_id}; use a new ID"
            )
        merged[existing_index] = generated
    return {"version": 1, "items": merged}


def prepare_assets(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    build_dir: Path = DEFAULT_BUILD_DIR,
    catalog_path: Path = DEFAULT_CATALOG,
    public_dir: Path = DEFAULT_PUBLIC_DIR,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> PreparedRun:
    manifest = _load_yaml(manifest_path, label="manifest")
    if set(manifest) - {"version", "items"}:
        raise MediaPipelineError("Manifest has unknown root fields")
    if manifest.get("version") != 1:
        raise MediaPipelineError("Manifest version must be 1")
    raw_items = manifest.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise MediaPipelineError("Manifest items must be a non-empty list")

    build_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=build_dir))
    public_hashes, public_fingerprints = _public_asset_index(
        public_dir,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    generated_items: list[dict[str, Any]] = []
    objects: list[PreparedObject] = []
    seen_ids: set[str] = set()

    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise MediaPipelineError(f"Manifest items[{index}] must be a mapping")
        unknown = sorted(set(raw_item) - _MANIFEST_FIELDS)
        if unknown:
            raise MediaPipelineError(
                f"Manifest items[{index}] has unknown/generated fields: {', '.join(unknown)}"
            )
        content_id = str(raw_item.get("id", ""))
        if not _ID_RE.fullmatch(content_id):
            raise MediaPipelineError(f"Manifest items[{index}].id is invalid")
        if content_id in seen_ids:
            raise MediaPipelineError(f"Duplicate manifest content ID: {content_id}")
        seen_ids.add(content_id)
        media_type = str(raw_item.get("type", ""))
        if media_type not in {"photo", "video"}:
            raise MediaPipelineError(f"Manifest item {content_id} type must be photo or video")
        source = _resolve_source(source_dir, raw_item.get("source"))
        _assert_private_source(
            source,
            media_type=media_type,
            public_dir=public_dir,
            public_hashes=public_hashes,
            public_photo_fingerprints=public_fingerprints,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )

        if media_type == "photo":
            generated, prepared_objects = _prepare_photo(content_id, source, run_dir)
        else:
            generated, prepared_objects = _prepare_video(
                content_id,
                source,
                run_dir,
                raw_item,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )
        authored = {
            key: raw_item[key]
            for key in (
                "id",
                "type",
                "status",
                "explicitness",
                "min_heat",
                "presentation",
                "tags",
            )
            if key in raw_item
        }
        authored.setdefault("status", "active")
        authored.update(generated)
        generated_items.append(authored)
        objects.extend(prepared_objects)

    existing, base_catalog_sha256 = _load_yaml_snapshot(
        catalog_path, label="runtime catalog"
    )
    candidate_payload = _merge_catalog(existing, generated_items)
    candidate_path = run_dir / "media_catalog.yaml"
    _write_yaml(candidate_path, candidate_payload)

    from bot.media_catalog import CatalogValidationError, load_media_catalog

    try:
        load_media_catalog(candidate_path)
    except CatalogValidationError as exc:
        raise MediaPipelineError(str(exc)) from exc
    return PreparedRun(
        run_dir,
        candidate_path,
        tuple(objects),
        base_catalog_sha256,
    )


def _head_object(client: Any, bucket_name: str, key: str) -> Mapping[str, Any] | None:
    try:
        return client.head_object(Bucket=bucket_name, Key=key)
    except Exception as exc:
        response = getattr(exc, "response", {})
        error = response.get("Error", {}) if isinstance(response, dict) else {}
        metadata = response.get("ResponseMetadata", {}) if isinstance(response, dict) else {}
        code = str(error.get("Code", ""))
        status = metadata.get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return None
        raise MediaPipelineError(f"R2 HEAD failed for {key}: {code or exc}") from exc


def _verify_prepared_head(obj: PreparedObject, head: Mapping[str, Any]) -> bool:
    metadata = head.get("Metadata") or {}
    try:
        content_length = int(head.get("ContentLength") or 0)
    except (TypeError, ValueError):
        return False
    return (
        content_length == obj.path.stat().st_size
        and str(head.get("ContentType") or "").lower() == obj.mime_type
        and str(metadata.get("sha256") or "").lower() == obj.sha256
    )


def _remote_sha256(client: Any, bucket_name: str, key: str) -> str:
    """Hash the bytes R2 actually stored instead of trusting custom metadata."""
    body: Any = None
    try:
        response = client.get_object(Bucket=bucket_name, Key=key)
        body = response.get("Body") if isinstance(response, Mapping) else None
        if body is None or not callable(getattr(body, "read", None)):
            raise MediaPipelineError(f"R2 GET returned no readable body for {key}")
        digest = hashlib.sha256()
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise MediaPipelineError(f"R2 GET returned non-binary data for {key}")
            digest.update(chunk)
        return digest.hexdigest()
    except MediaPipelineError:
        raise
    except Exception as exc:
        raise MediaPipelineError(f"R2 GET verification failed for {key}") from exc
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()


def _remote_object_matches(
    client: Any,
    bucket_name: str,
    obj: PreparedObject,
    head: Mapping[str, Any] | None,
) -> bool:
    return bool(
        head is not None
        and _verify_prepared_head(obj, head)
        and _remote_sha256(client, bucket_name, obj.key) == obj.sha256
    )


def _is_precondition_failed(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    metadata = response.get("ResponseMetadata", {}) if isinstance(response, dict) else {}
    return (
        str(error.get("Code", "")) in {"412", "PreconditionFailed"}
        or metadata.get("HTTPStatusCode") == 412
    )


def upload_prepared_objects(
    prepared: PreparedRun,
    *,
    client: Any,
    bucket_name: str,
) -> tuple[int, int]:
    uploaded = 0
    unchanged = 0
    try:
        client.head_bucket(Bucket=bucket_name)
    except Exception as exc:
        raise MediaPipelineError(
            "Could not access the configured R2 bucket; check bucket-scoped credentials"
        ) from exc
    for obj in prepared.objects:
        existing = _head_object(client, bucket_name, obj.key)
        if existing is not None:
            if _remote_object_matches(client, bucket_name, obj, existing):
                unchanged += 1
                continue
            raise MediaPipelineError(
                f"R2 key {obj.key} already exists with different bytes or metadata; use a new content ID"
            )
        object_size = obj.path.stat().st_size
        if object_size > _MAX_SINGLE_UPLOAD_BYTES:
            raise MediaPipelineError(
                f"R2 object {obj.key} exceeds the safe conditional single-upload limit"
            )
        try:
            with obj.path.open("rb") as body:
                client.put_object(
                    Bucket=bucket_name,
                    Key=obj.key,
                    Body=body,
                    ContentLength=object_size,
                    IfNoneMatch="*",
                    ContentType=obj.mime_type,
                    CacheControl="private, max-age=3600",
                    ContentDisposition="inline",
                    Metadata={
                        "sha256": obj.sha256,
                        "media-role": obj.role,
                        "pipeline-version": "1",
                    },
                )
        except Exception as exc:
            if _is_precondition_failed(exc):
                collided = _head_object(client, bucket_name, obj.key)
                if _remote_object_matches(client, bucket_name, obj, collided):
                    unchanged += 1
                    continue
                raise MediaPipelineError(
                    f"R2 conditional create collided with different content at {obj.key}"
                ) from exc
            raise MediaPipelineError(f"R2 upload failed for {obj.key}") from exc
        verified = _head_object(client, bucket_name, obj.key)
        if not _remote_object_matches(client, bucket_name, obj, verified):
            raise MediaPipelineError(f"R2 verification failed after uploading {obj.key}")
        uploaded += 1
    return uploaded, unchanged


def _catalog_probe(client: Any, bucket_name: str, key: str) -> bool | dict[str, Any]:
    head = _head_object(client, bucket_name, key)
    if head is None:
        return False
    metadata = head.get("Metadata") or {}
    return {
        "exists": True,
        "content_type": str(head.get("ContentType") or "").lower(),
        "content_length": int(head.get("ContentLength") or 0),
        "sha256": str(metadata.get("sha256") or "").lower(),
    }


@contextmanager
def _catalog_publish_lock(target: Path) -> Iterator[None]:
    """Hold an OS-managed cross-process lock; crashes cannot leave it locked."""
    lock_path = target.with_name(f".{target.name}.publish.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    acquired = False
    try:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise MediaPipelineError(
                "Another media publisher is updating the runtime catalog"
            ) from exc
        acquired = True
        yield
    finally:
        if acquired:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _install_catalog(
    candidate: Path,
    target: Path,
    *,
    expected_sha256: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with candidate.open("rb") as source, os.fdopen(handle, "wb") as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        with _catalog_publish_lock(target):
            try:
                current_sha256 = sha256_file(target)
            except OSError as exc:
                raise MediaPipelineError(
                    "Runtime catalog disappeared before publication; refusing a stale replace"
                ) from exc
            if current_sha256 != expected_sha256:
                raise MediaPipelineError(
                    "Runtime catalog changed after preparation; rerun publish against the latest catalog"
                )
            # The cross-process lock closes the check/replace race; the digest
            # comparison remains immediately adjacent to the atomic replace.
            os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_prepared(
    prepared: PreparedRun,
    *,
    client: Any,
    bucket_name: str,
    catalog_path: Path = DEFAULT_CATALOG,
) -> tuple[int, int]:
    uploaded, unchanged = upload_prepared_objects(
        prepared, client=client, bucket_name=bucket_name
    )
    from bot.media_catalog import load_media_catalog, validate_catalog_objects

    catalog = load_media_catalog(prepared.candidate_catalog)
    asyncio.run(
        validate_catalog_objects(
            catalog, lambda key: _catalog_probe(client, bucket_name, key)
        )
    )
    _install_catalog(
        prepared.candidate_catalog,
        catalog_path,
        expected_sha256=prepared.base_catalog_sha256,
    )
    return uploaded, unchanged


def create_r2_client() -> tuple[Any, str]:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise MediaPipelineError(
            "R2 publishing needs: pip install -r requirements-media.txt"
        ) from exc
    from bot.media_delivery import R2DeliveryConfig

    try:
        cfg = R2DeliveryConfig.from_environment()
    except Exception as exc:
        raise MediaPipelineError(str(exc)) from exc
    upload_access_key = os.getenv("R2_UPLOAD_ACCESS_KEY_ID", "").strip()
    upload_secret_key = os.getenv("R2_UPLOAD_SECRET_ACCESS_KEY", "").strip()
    if not upload_access_key or not upload_secret_key:
        raise MediaPipelineError(
            "Set separate bucket-scoped R2_UPLOAD_ACCESS_KEY_ID and "
            "R2_UPLOAD_SECRET_ACCESS_KEY for publishing"
        )
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{cfg.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=upload_access_key,
        aws_secret_access_key=upload_secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 4}),
    )
    return client, cfg.bucket_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "publish"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepared = prepare_assets(
            manifest_path=args.manifest,
            source_dir=args.source_dir,
            build_dir=args.build_dir,
            catalog_path=args.catalog,
            public_dir=args.public_dir,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
        if args.command == "prepare":
            print(f"Prepared {len(prepared.objects)} objects in {prepared.run_dir}")
            print(f"Candidate catalog: {prepared.candidate_catalog}")
            return 0
        client, bucket_name = create_r2_client()
        uploaded, unchanged = publish_prepared(
            prepared,
            client=client,
            bucket_name=bucket_name,
            catalog_path=args.catalog,
        )
        print(
            f"Published safely: {uploaded} uploaded, {unchanged} already identical; "
            f"catalog installed at {args.catalog}"
        )
        return 0
    except MediaPipelineError as exc:
        print(f"Media pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
