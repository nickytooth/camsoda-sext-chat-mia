import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from bot.media_catalog import CatalogValidationError, load_media_catalog
from scripts.media_pipeline import (
    MediaPipelineError,
    _catalog_publish_lock,
    prepare_assets,
    publish_prepared,
    upload_prepared_objects,
)


def manifest_item(source: str, *, content_id: str = "mia_private_001"):
    return {
        "id": content_id,
        "source": source,
        "type": "photo",
        "status": "active",
        "explicitness": "suggestive",
        "min_heat": "rising",
        "presentation": {
            "mode": "past_only",
            "periods": [],
            "current_description": None,
            "past_description": "a private test photo",
        },
        "tags": {
            "location": ["home"],
            "body_focus": ["full_body"],
            "activity": ["posing"],
            "outfit": ["casual"],
            "vibe": ["teasing"],
            "capture": ["selfie"],
        },
    }


class MissingObject(Exception):
    def __init__(self):
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class PreconditionFailed(Exception):
    def __init__(self):
        self.response = {
            "Error": {"Code": "PreconditionFailed"},
            "ResponseMetadata": {"HTTPStatusCode": 412},
        }


class FakeR2Client:
    def __init__(self, *, fail_on_put=None, race_identical_on_put=False):
        self.objects = {}
        self.fail_on_put = fail_on_put
        self.race_identical_on_put = race_identical_on_put
        self.put_count = 0

    def head_bucket(self, *, Bucket):
        return {"Bucket": Bucket}

    def head_object(self, *, Bucket, Key):
        try:
            return dict(self.objects[(Bucket, Key)]["head"])
        except KeyError as exc:
            raise MissingObject() from exc

    def get_object(self, *, Bucket, Key):
        try:
            body = self.objects[(Bucket, Key)]["body"]
        except KeyError as exc:
            raise MissingObject() from exc
        return {"Body": io.BytesIO(body)}

    def put_object(
        self,
        *,
        Bucket,
        Key,
        Body,
        ContentLength,
        IfNoneMatch,
        ContentType,
        CacheControl,
        ContentDisposition,
        Metadata,
    ):
        self.put_count += 1
        if self.fail_on_put == self.put_count:
            raise RuntimeError("synthetic upload failure")
        self.asserted_if_none_match = IfNoneMatch
        data = Body.read()
        self.asserted_cache_control = CacheControl
        self.asserted_content_disposition = ContentDisposition
        entry = {
            "head": {
                "ContentLength": ContentLength,
                "ContentType": ContentType,
                "Metadata": dict(Metadata),
            },
            "body": data,
        }
        if self.race_identical_on_put and self.put_count == 1:
            self.objects[(Bucket, Key)] = entry
            raise PreconditionFailed()
        if (Bucket, Key) in self.objects or IfNoneMatch != "*":
            raise PreconditionFailed()
        self.objects[(Bucket, Key)] = entry
        return {"ETag": "synthetic"}


class MediaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "private" / "originals"
        self.sources.mkdir(parents=True)
        self.public = self.root / "public"
        self.public.mkdir()
        self.build = self.root / "build"
        self.catalog = self.root / "media_catalog.yaml"
        self.catalog.write_text("version: 1\nitems: []\n", encoding="utf-8")
        self.manifest = self.root / "manifest.yaml"

    def tearDown(self):
        self.temporary.cleanup()

    def make_photo(self, name="private.png", *, color=(180, 30, 120)):
        from PIL import Image

        path = self.sources / name
        image = Image.new("RGB", (360, 640), color)
        for offset in range(0, 360, 24):
            for y in range(640):
                image.putpixel((offset, y), ((offset + y) % 255, y % 180, 120))
        image.save(path, format="PNG")
        return path

    def write_manifest(self, item):
        self.manifest.write_text(
            yaml.safe_dump({"version": 1, "items": [item]}, sort_keys=False),
            encoding="utf-8",
        )

    def prepare(self):
        return prepare_assets(
            manifest_path=self.manifest,
            source_dir=self.sources,
            build_dir=self.build,
            catalog_path=self.catalog,
            public_dir=self.public,
        )

    def test_photo_pipeline_generates_separate_content_addressed_derivative(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))

        prepared = self.prepare()
        catalog = load_media_catalog(prepared.candidate_catalog)
        item = catalog.require("mia_private_001")

        self.assertRegex(item.full_key, r"^premium/mia_private_001/[0-9a-f]{64}\.png$")
        self.assertRegex(
            item.preview_key, r"^previews/mia_private_001/[0-9a-f]{64}\.webp$"
        )
        self.assertEqual(len(prepared.objects), 2)
        self.assertNotEqual(prepared.objects[0].sha256, prepared.objects[1].sha256)

        from PIL import Image

        with Image.open(prepared.objects[1].path) as preview:
            self.assertLessEqual(preview.width, 420)
            self.assertLessEqual(preview.height, 640)
            self.assertEqual(preview.format, "WEBP")

    def test_public_source_or_copy_is_rejected(self):
        source = self.make_photo()
        shutil.copyfile(source, self.public / "already-public.png")
        self.write_manifest(manifest_item("private.png"))
        with self.assertRaisesRegex(MediaPipelineError, "public"):
            self.prepare()

    def test_manifest_cannot_inject_generated_storage_fields(self):
        self.make_photo()
        item = manifest_item("private.png")
        item["full_key"] = "public/bypass.png"
        self.write_manifest(item)
        with self.assertRaisesRegex(MediaPipelineError, "unknown/generated fields"):
            self.prepare()

    def test_publish_is_verified_atomic_and_idempotent(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))
        prepared = self.prepare()
        client = FakeR2Client()

        uploaded, unchanged = publish_prepared(
            prepared,
            client=client,
            bucket_name="private-bucket",
            catalog_path=self.catalog,
        )
        self.assertEqual((uploaded, unchanged), (2, 0))
        self.assertEqual(len(load_media_catalog(self.catalog).items), 1)

        uploaded, unchanged = upload_prepared_objects(
            prepared, client=client, bucket_name="private-bucket"
        )
        self.assertEqual((uploaded, unchanged), (0, 2))

    def test_partial_upload_failure_leaves_catalog_byte_for_byte_unchanged(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))
        prepared = self.prepare()
        before = self.catalog.read_bytes()
        client = FakeR2Client(fail_on_put=2)

        with self.assertRaisesRegex(MediaPipelineError, "upload failed"):
            publish_prepared(
                prepared,
                client=client,
                bucket_name="private-bucket",
                catalog_path=self.catalog,
            )

        self.assertEqual(self.catalog.read_bytes(), before)
        self.assertEqual(len(client.objects), 1)  # private orphan; never catalogued

    def test_remote_catalog_validation_failure_leaves_catalog_unchanged(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))
        prepared = self.prepare()
        before = self.catalog.read_bytes()
        client = FakeR2Client()

        with patch("scripts.media_pipeline._catalog_probe", return_value=False):
            with self.assertRaises(CatalogValidationError):
                publish_prepared(
                    prepared,
                    client=client,
                    bucket_name="private-bucket",
                    catalog_path=self.catalog,
                )

        self.assertEqual(self.catalog.read_bytes(), before)
        self.assertEqual(len(client.objects), 2)

    def test_catalog_compare_and_swap_rejects_a_stale_prepared_run(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))
        prepared = self.prepare()
        newer = b"version: 1\nitems: []\n# concurrent catalog update\n"
        self.catalog.write_bytes(newer)

        with self.assertRaisesRegex(MediaPipelineError, "changed after preparation"):
            publish_prepared(
                prepared,
                client=FakeR2Client(),
                bucket_name="private-bucket",
                catalog_path=self.catalog,
            )

        self.assertEqual(self.catalog.read_bytes(), newer)

    def test_cross_process_catalog_lock_fails_closed(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))
        prepared = self.prepare()
        before = self.catalog.read_bytes()

        with _catalog_publish_lock(self.catalog):
            with self.assertRaisesRegex(MediaPipelineError, "Another media publisher"):
                publish_prepared(
                    prepared,
                    client=FakeR2Client(),
                    bucket_name="private-bucket",
                    catalog_path=self.catalog,
                )

        self.assertEqual(self.catalog.read_bytes(), before)

    def test_conditional_create_handles_an_identical_race_without_overwrite(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))
        prepared = self.prepare()
        client = FakeR2Client(race_identical_on_put=True)

        uploaded, unchanged = publish_prepared(
            prepared,
            client=client,
            bucket_name="private-bucket",
            catalog_path=self.catalog,
        )

        self.assertEqual((uploaded, unchanged), (1, 1))
        self.assertEqual(client.asserted_if_none_match, "*")

    def test_existing_metadata_cannot_mask_different_remote_bytes(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))
        prepared = self.prepare()
        obj = prepared.objects[0]
        client = FakeR2Client()
        client.objects[("private-bucket", obj.key)] = {
            "head": {
                "ContentLength": obj.path.stat().st_size,
                "ContentType": obj.mime_type,
                "Metadata": {"sha256": obj.sha256},
            },
            "body": b"x" * obj.path.stat().st_size,
        }

        with self.assertRaisesRegex(MediaPipelineError, "different bytes or metadata"):
            upload_prepared_objects(
                prepared,
                client=client,
                bucket_name="private-bucket",
            )

    def test_existing_content_id_cannot_change_bytes(self):
        self.make_photo()
        self.write_manifest(manifest_item("private.png"))
        prepared = self.prepare()
        client = FakeR2Client()
        publish_prepared(
            prepared,
            client=client,
            bucket_name="private-bucket",
            catalog_path=self.catalog,
        )

        self.make_photo("replacement.png", color=(20, 160, 80))
        self.write_manifest(manifest_item("replacement.png"))
        with self.assertRaisesRegex(MediaPipelineError, "use a new ID"):
            self.prepare()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_reencoded_public_video_is_rejected_by_frame_fingerprints(self):
        public_video = self.public / "public-profile.mp4"
        private_copy = self.sources / "reencoded-private.mov"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x240:rate=12:duration=2",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(public_video),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(public_video),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "30",
                str(private_copy),
            ],
            check=True,
            capture_output=True,
        )
        item = manifest_item(
            "reencoded-private.mov", content_id="mia_public_video_copy_001"
        )
        item["type"] = "video"
        self.write_manifest(item)

        with self.assertRaisesRegex(MediaPipelineError, "already available publicly"):
            self.prepare()
        self.assertEqual(list(self.build.rglob("*.png")), [])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_video_pipeline_normalizes_full_and_generates_two_degraded_frames(self):
        source = self.sources / "private.mov"
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=12:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ]
        subprocess.run(command, check=True, capture_output=True)
        item = manifest_item("private.mov", content_id="mia_private_video_001")
        item["type"] = "video"
        item["preview_time_seconds"] = 0.2
        item["poster_time_seconds"] = 0.6
        self.write_manifest(item)

        prepared = self.prepare()
        catalog = load_media_catalog(prepared.candidate_catalog)
        video = catalog.require("mia_private_video_001")
        self.assertEqual(video.mime_type, "video/mp4")
        self.assertGreater(video.duration_seconds, 0)
        self.assertEqual(len(prepared.objects), 3)
        self.assertRegex(video.full_key, r"/[0-9a-f]{64}\.mp4$")
        self.assertRegex(video.preview_key, r"/[0-9a-f]{64}\.webp$")
        self.assertRegex(video.poster_key, r"/[0-9a-f]{64}\.webp$")
        self.assertEqual(len({obj.sha256 for obj in prepared.objects}), 3)


if __name__ == "__main__":
    unittest.main()
