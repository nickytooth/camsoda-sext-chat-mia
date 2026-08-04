import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from bot.media_catalog import (
    CatalogValidationError,
    load_media_catalog,
    validate_catalog_objects,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "media_catalog.yaml"
RUNTIME_CATALOG_PATH = PROJECT_ROOT / "library" / "media_catalog.yaml"


class MediaCatalogValidationTests(unittest.TestCase):
    def setUp(self):
        self.payload = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))

    def load_payload(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            return load_media_catalog(path)

    def test_runtime_catalog_is_valid_and_safe_by_default(self):
        catalog = load_media_catalog(RUNTIME_CATALOG_PATH)
        self.assertEqual(catalog.items, ())

    def test_fixture_catalog_is_valid_and_has_bar_photo_and_video(self):
        catalog = load_media_catalog(CATALOG_PATH)
        self.assertTrue(
            any(
                item.media_type == "photo" and "bar" in item.tags["location"]
                for item in catalog.active_items()
            )
        )
        self.assertTrue(any(item.media_type == "video" for item in catalog.active_items()))

    def test_unknown_tag_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["items"][0]["tags"]["location"].append("spaceship")
        with self.assertRaises(CatalogValidationError) as raised:
            self.load_payload(payload)
        self.assertIn("unknown values: spaceship", str(raised.exception))

    def test_duplicate_id_checksum_and_any_r2_key_are_rejected(self):
        for mutation, phrase in (
            (lambda a, b: b.update(id=a["id"]), "duplicate content IDs"),
            (lambda a, b: b.update(sha256=a["sha256"]), "duplicate full object checksums"),
            (
                lambda a, b: b.update(preview_key=a["full_key"]),
                "duplicate R2 object keys",
            ),
        ):
            with self.subTest(phrase=phrase):
                payload = copy.deepcopy(self.payload)
                mutation(payload["items"][0], payload["items"][1])
                with self.assertRaises(CatalogValidationError) as raised:
                    self.load_payload(payload)
                self.assertIn(phrase, str(raised.exception))

    def test_video_requires_separate_image_poster(self):
        video_index = next(
            index
            for index, item in enumerate(self.payload["items"])
            if item["type"] == "video"
        )
        for poster, phrase in (
            (None, "poster_key is required"),
            (self.payload["items"][video_index]["full_key"], "separate preview object"),
            ("posters/not-an-image.mp4", "supported image poster"),
        ):
            with self.subTest(poster=poster):
                payload = copy.deepcopy(self.payload)
                payload["items"][video_index]["poster_key"] = poster
                with self.assertRaises(CatalogValidationError) as raised:
                    self.load_payload(payload)
                self.assertIn(phrase, str(raised.exception))

    def test_inappropriate_full_mime_and_preview_extension_are_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["items"][0]["mime_type"] = "video/mp4"
        payload["items"][0]["preview_key"] = "previews/item.txt"
        with self.assertRaises(CatalogValidationError) as raised:
            self.load_payload(payload)
        message = str(raised.exception)
        self.assertIn("image/* for photo", message)
        self.assertIn("supported image preview", message)


class MediaCatalogRemoteValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_head_metadata_validates_mime_checksum_and_nonempty_objects(self):
        catalog = load_media_catalog(CATALOG_PATH)
        by_key = {}
        for item in catalog.items:
            by_key[item.full_key] = {
                "exists": True,
                "content_type": item.mime_type,
                "sha256": item.sha256,
                "content_length": 100,
            }
            by_key[item.preview_key] = {
                "exists": True,
                "content_type": "image/webp",
                "content_length": 50,
            }
            if item.poster_key:
                by_key[item.poster_key] = {
                    "exists": True,
                    "content_type": "image/webp",
                    "content_length": 50,
                }
        await validate_catalog_objects(catalog, by_key.__getitem__)

        broken = dict(by_key)
        first = catalog.items[0]
        broken[first.full_key] = {
            **broken[first.full_key],
            "sha256": "0" * 64,
        }
        with self.assertRaises(CatalogValidationError) as raised:
            await validate_catalog_objects(catalog, broken.__getitem__)
        self.assertIn("SHA-256 does not match", str(raised.exception))

    async def test_head_metadata_requires_positive_content_length(self):
        catalog = load_media_catalog(CATALOG_PATH)
        by_key = {}
        for item in catalog.items:
            by_key[item.full_key] = {
                "exists": True,
                "content_type": item.mime_type,
                "sha256": item.sha256,
                "content_length": 100,
            }
            by_key[item.preview_key] = {
                "exists": True,
                "content_type": "image/webp",
                "content_length": 50,
            }
            if item.poster_key:
                by_key[item.poster_key] = {
                    "exists": True,
                    "content_type": "image/webp",
                    "content_length": 50,
                }

        first = catalog.items[0]
        for invalid_length in (None, 0, -1, "invalid", True):
            with self.subTest(invalid_length=invalid_length):
                broken = copy.deepcopy(by_key)
                if invalid_length is None:
                    broken[first.full_key].pop("content_length")
                else:
                    broken[first.full_key]["content_length"] = invalid_length
                with self.assertRaises(CatalogValidationError) as raised:
                    await validate_catalog_objects(catalog, broken.__getitem__)
                self.assertIn("positive content length", str(raised.exception))

    async def test_boolean_probe_still_rejects_missing_preview(self):
        catalog = load_media_catalog(CATALOG_PATH)
        missing = catalog.items[0].preview_key
        with self.assertRaises(CatalogValidationError) as raised:
            await validate_catalog_objects(catalog, lambda key: key != missing)
        self.assertIn(missing, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
