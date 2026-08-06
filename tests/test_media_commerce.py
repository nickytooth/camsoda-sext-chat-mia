import dataclasses
import random
import unittest
from pathlib import Path

from bot.media_catalog import MediaCatalog, MediaPresentation, load_media_catalog
from bot.media_commerce import (
    CommerceAction,
    MediaCommerceService,
    _spoken_fallback_context,
    _spoken_item_description,
)
from bot.media_planner import classify_media_intent
from bot.media_repository import OfferRecord


CATALOG_PATH = Path(__file__).resolve().parent / "fixtures" / "media_catalog.yaml"
PRODUCTION_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "library" / "media_catalog.yaml"
)


class FakeRepository:
    def __init__(self):
        self.state = {"user_id": 1, "total_messages": 0}
        self.unlocked = set()
        self.history = []
        self.reserved = {}
        self.affinity = {}
        self.next_offer_id = 1
        self.cancelled_stale = 0

    async def cancel_stale_reservations(self, user_id, *, older_than_seconds=600):
        self.cancelled_stale += 1
        return 0

    async def get_engagement_state(self, user_id):
        return dict(self.state)

    async def unlocked_content_ids(self, user_id):
        return set(self.unlocked)

    async def delivered_offer_history(self, user_id):
        return list(self.history)

    async def get_tag_affinity(self, user_id):
        return dict(self.affinity)

    async def active_reserved_content_ids(self, user_id):
        return {row.content_id for row in self.reserved.values()}

    async def reserve_offer(
        self,
        user_id,
        content_id,
        *,
        trigger,
        action,
        request_type,
        description,
        batch_number,
        price_tokens,
    ):
        record = OfferRecord(
            offer_id=self.next_offer_id,
            user_id=user_id,
            content_id=content_id,
            trigger=trigger,
            action=action,
            request_type=request_type,
            description=description,
            batch_number=batch_number,
            price_tokens=price_tokens,
            status="reserved",
            created_at=float(batch_number),
            offered_at=None,
        )
        self.next_offer_id += 1
        self.reserved[record.offer_id] = record
        return record

    async def get_offer(self, offer_id, *, user_id=None, delivered_only=False):
        offer_id = int(offer_id)
        row = self.reserved.get(offer_id)
        if row is None:
            row = next((item for item in self.history if item.offer_id == offer_id), None)
        if row is None or (user_id is not None and row.user_id != user_id):
            return None
        if delivered_only and row.status != "delivered":
            return None
        return row

    async def mark_offer_delivered(self, offer_id, *, media_type=None):
        row = self.reserved.pop(int(offer_id), None)
        if row is None:
            return next(
                (item for item in self.history if item.offer_id == int(offer_id)), None
            )
        row = dataclasses.replace(row, status="delivered", offered_at=row.created_at)
        self.history.insert(0, row)
        if row.trigger == "proactive":
            self.state["last_proactive_media_batch"] = row.batch_number
        if row.request_type == "generic":
            self.state["last_generic_media_type"] = media_type
        if row.trigger in {"direct", "permission_reask"}:
            await self.clear_sales_pause(row.user_id)
        return row

    async def cancel_offer_reservation(self, offer_id):
        return self.reserved.pop(int(offer_id), None) is not None

    async def set_decline_snooze(self, user_id, *, until_batch, reask_pending=True):
        self.state.update(
            sales_snooze_until_batch=until_batch,
            sales_reask_pending=reask_pending,
            sales_reask_asked_at_batch=None,
        )

    async def mark_reask_asked(self, user_id, *, batch_number):
        self.state["sales_reask_pending"] = True
        self.state["sales_reask_asked_at_batch"] = batch_number

    async def clear_sales_pause(self, user_id):
        self.state.update(
            sales_snooze_until_batch=None,
            sales_reask_pending=False,
            sales_reask_asked_at_batch=None,
        )

    async def clear_stale_reask(self, user_id, *, asked_at_batch):
        if self.state.get("sales_reask_asked_at_batch") == asked_at_batch:
            await self.clear_sales_pause(user_id)

    async def set_last_generic_media_type(self, user_id, media_type):
        self.state["last_generic_media_type"] = media_type


class MediaIntentTests(unittest.TestCase):
    def test_catalog_copy_is_rendered_in_mias_first_person_voice(self):
        cases = {
            "a photo from her bed": "a photo from my bed",
            "a playful photo she took at home": "a playful photo I took at home",
            "Mia's private bathroom clip": "my private bathroom clip",
            "a photo of her": "a photo of me",
            "a clip with her dancing": "a clip with me dancing",
            "a photo of her in bed": "a photo of me in bed",
            "a clip with her on the couch": "a clip with me on the couch",
            "a photo of her body": "a photo of my body",
            "a photo of herself in bed": "a photo of me in bed",
            "Mia is posing in her bedroom": "I am posing in my bedroom",
            "Mia was posing in her bedroom": "I was posing in my bedroom",
            "Mia has saved this photo": "I have saved this photo",
            "she's posing in her bedroom": "I'm posing in my bedroom",
            "she hasn't shared this photo": "I haven't shared this photo",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(_spoken_item_description(raw), expected)

        self.assertEqual(
            _spoken_fallback_context("Tyler is asleep beside her"),
            "Tyler is asleep beside me",
        )
        self.assertEqual(
            _spoken_fallback_context("she is home getting ready"),
            "I am home getting ready",
        )
        self.assertEqual(
            _spoken_fallback_context(
                "customers and coworkers are around her at the bar"
            ),
            "customers and coworkers are around me at the bar",
        )

    def test_inventory_video_question_is_direct_but_story_mentions_are_not(self):
        inventory = classify_media_intent("do you have any videos?")
        self.assertTrue(inventory.requested)
        self.assertEqual(inventory.requested_type, "video")

        for text in ("I watched a video", "we watched videos last night"):
            with self.subTest(text=text):
                mention = classify_media_intent(text)
                self.assertFalse(mention.requested)
                self.assertEqual(mention.requested_type, "video")

    def test_required_aliases_normalize_to_controlled_values(self):
        cases = {
            "send a clip": ("video", None, None),
            "show me your tits": (None, "boobs", None),
            "your breasts in a photo": ("photo", "boobs", None),
            "show me that butt": (None, "ass", None),
            "a booty pic": ("photo", "ass", None),
            "send nudes": (None, None, "nude"),
            "show me you naked": (None, None, "nude"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                intent = classify_media_intent(text)
                self.assertTrue(intent.requested)
                self.assertEqual(intent.requested_type, expected[0])
                if expected[1]:
                    self.assertIn(expected[1], intent.tags["body_focus"])
                self.assertEqual(intent.explicitness, expected[2])

    def test_work_aliases_and_negative_media_command(self):
        for text in ("a photo at work", "a photo behind the bar"):
            with self.subTest(text=text):
                self.assertIn("bar", classify_media_intent(text).tags["location"])
        decline = classify_media_intent("don't send me a video")
        self.assertEqual(decline.decline_kind, "soft")

    def test_media_mentions_and_open_ended_show_me_are_not_requests(self):
        for text in (
            "that video was funny",
            "we took a photo",
            "I hate photos",
            "show me what you want",
            "that video is really good",
        ):
            with self.subTest(text=text):
                self.assertFalse(classify_media_intent(text).requested)

    def test_negative_want_is_a_decline_not_a_direct_request(self):
        intent = classify_media_intent("I don't want a photo")
        self.assertEqual(intent.decline_kind, "soft")
        self.assertTrue(intent.decline_global)

    def test_do_not_sell_content_is_a_global_soft_decline(self):
        intent = classify_media_intent("don't sell me content")
        self.assertEqual(intent.decline_kind, "soft")
        self.assertTrue(intent.decline_global)
        self.assertFalse(intent.requested)

    def test_cyrillic_media_request_is_blocked_in_english_only_runtime(self):
        intent = classify_media_intent("прати send nudes")
        self.assertFalse(intent.requested)
        self.assertTrue(intent.blocked_request)

    def test_cue_plus_visual_facet_is_a_request(self):
        for text in ("show me your ass", "I want nudes", "send nudes"):
            with self.subTest(text=text):
                self.assertTrue(classify_media_intent(text).requested)

    def test_polite_photo_requests_are_direct_media_requests(self):
        for text in (
            "can I have a picture?",
            "could I have a photo, please?",
        ):
            with self.subTest(text=text):
                intent = classify_media_intent(text)
                self.assertTrue(intent.requested)
                self.assertEqual(intent.requested_type, "photo")

    def test_generic_request_accepts_punctuation_emoji_and_politeness(self):
        for text in (
            "show me!!!",
            "show me 😏",
            "show me, please 😏",
        ):
            with self.subTest(text=text):
                intent = classify_media_intent(text)
                self.assertTrue(intent.requested)
                self.assertIsNone(intent.requested_type)

    def test_exact_explicit_request_and_affirmative_aliases(self):
        request = classify_media_intent("but I wanna see your pussy so bad babe")
        self.assertTrue(request.requested)
        self.assertIn("pussy", request.tags["body_focus"])

        for text in ("now!", "do it", "send it"):
            with self.subTest(text=text):
                intent = classify_media_intent(text)
                self.assertTrue(intent.affirmative)
                self.assertFalse(intent.requested)


class MediaCommercePlanningTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.repository = FakeRepository()
        self.service = MediaCommerceService(
            load_media_catalog(CATALOG_PATH),
            repository=self.repository,
            random_source=random.Random(7),
        )

    async def deliver(self, decision):
        self.assertIsNotNone(decision.offer)
        return await self.service.mark_offer_delivered(decision.offer.offer_id)

    async def plan_batch(
        self,
        messages,
        *,
        batch_number,
        heat="low",
        period="bar_shift",
    ):
        intent = await self.service.classify_media_turn(
            1,
            messages,
            batch_number=batch_number,
        )
        return await self.service.plan_commerce_turn(
            1,
            messages[-1] if messages else "",
            batch_number=batch_number,
            heat=heat,
            period=period,
            intent=intent,
        )

    async def test_first_direct_request_immediately_offers_at_low_heat(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "can you send me a picture?",
            batch_number=1,
            heat="low",
            period="bar_shift",
        )

        self.assertEqual(decision.action, CommerceAction.OFFER_CURRENT)
        self.assertIsNotNone(decision.offer)
        self.assertEqual(decision.offer.media_type, "photo")
        self.assertEqual(decision.offer.trigger, "direct")

    async def test_inventory_video_question_immediately_offers(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "do you have any videos?",
            batch_number=1,
            heat="low",
            period="bar_shift",
        )

        self.assertIsNotNone(decision.offer)
        self.assertEqual(decision.offer.media_type, "video")
        self.assertEqual(decision.offer.request_type, "video")
        self.assertEqual(decision.offer.trigger, "direct")

    async def test_contextual_requests_require_recent_offer_or_unlock(self):
        first = await self.service.plan_commerce_turn(
            1,
            "send me a photo",
            batch_number=1,
            heat="low",
            period="bar_shift",
        )
        await self.deliver(first)

        for text, requested_type in (
            ("another one", None),
            ("what about a video?", "video"),
        ):
            with self.subTest(source="offer", text=text):
                recent = await self.service.classify_media_turn(
                    1,
                    [text],
                    batch_number=9,
                )
                stale = await self.service.classify_media_turn(
                    1,
                    [text],
                    batch_number=10,
                )
                self.assertTrue(recent.requested)
                self.assertEqual(recent.requested_type, requested_type)
                self.assertFalse(stale.requested)

        self.repository.history.clear()
        self.repository.state["last_media_unlock_batch"] = 20
        recent_unlock = await self.service.classify_media_turn(
            1,
            ["show me another"],
            batch_number=28,
        )
        stale_unlock = await self.service.classify_media_turn(
            1,
            ["show me another"],
            batch_number=29,
        )
        self.assertTrue(recent_unlock.requested)
        self.assertFalse(stale_unlock.requested)

    async def test_bare_affirmative_only_accepts_immediate_permission_reask(self):
        first = await self.service.plan_commerce_turn(
            1,
            "send me a photo",
            batch_number=1,
            heat="low",
            period="bar_shift",
        )
        await self.deliver(first)

        recent_offer_yes = await self.service.classify_media_turn(
            1,
            ["yes"],
            batch_number=2,
        )
        self.assertTrue(recent_offer_yes.affirmative)
        self.assertFalse(recent_offer_yes.requested)

        self.repository.state.update(
            sales_snooze_until_batch=20,
            sales_reask_pending=True,
            sales_reask_asked_at_batch=20,
        )
        immediate = await self.service.classify_media_turn(
            1,
            ["yes"],
            batch_number=21,
        )
        late = await self.service.classify_media_turn(
            1,
            ["yes"],
            batch_number=22,
        )
        self.assertTrue(immediate.requested)
        self.assertTrue(immediate.affirmative)
        self.assertFalse(late.requested)

        accepted = await self.service.plan_commerce_turn(
            1,
            "yes",
            batch_number=21,
            heat="high",
            period="bar_shift",
            intent=immediate,
        )
        self.assertIsNotNone(accepted.offer)
        self.assertEqual(accepted.offer.trigger, "permission_reask")

    async def test_latest_decisive_raw_batch_command_wins(self):
        declined = await self.service.classify_media_turn(
            1,
            ["send me a photo", "haha", "no, not now", "thanks"],
            batch_number=1,
        )
        self.assertFalse(declined.requested)
        self.assertEqual(declined.decline_kind, "soft")

        requested = await self.service.classify_media_turn(
            1,
            ["not now", "wait", "actually send me a video", "please"],
            batch_number=2,
        )
        self.assertTrue(requested.requested)
        self.assertEqual(requested.requested_type, "video")

        decision = await self.service.plan_commerce_turn(
            1,
            "please",
            batch_number=2,
            heat="low",
            period="bar_shift",
            intent=requested,
        )
        self.assertIsNotNone(decision.offer)
        self.assertEqual(decision.offer.trigger, "direct")

    async def test_raw_lifetime_count_does_not_unlock_proactive_offer(self):
        self.repository.state["lifetime_user_messages"] = 200
        decision = await self.service.plan_commerce_turn(
            1, "one processed batch", batch_number=1, heat="high", period="bar_shift"
        )
        self.assertEqual(decision.action, CommerceAction.NONE)

    async def test_proactive_minimum_and_cooldown_are_delivered_batch_based(self):
        before = await self.service.plan_commerce_turn(
            1, "flirting", batch_number=7, heat="high", period="bar_shift"
        )
        self.assertEqual(before.action, CommerceAction.NONE)
        first = await self.service.plan_commerce_turn(
            1, "flirting", batch_number=8, heat="high", period="bar_shift"
        )
        self.assertEqual(first.action, CommerceAction.OFFER_CURRENT)
        await self.deliver(first)
        cooldown = await self.service.plan_commerce_turn(
            1, "more flirting", batch_number=15, heat="high", period="bar_shift"
        )
        self.assertEqual(cooldown.action, CommerceAction.NONE)
        ready = await self.service.plan_commerce_turn(
            1, "more flirting", batch_number=16, heat="high", period="bar_shift"
        )
        self.assertIn(
            ready.action,
            {
                CommerceAction.OFFER_CURRENT,
                CommerceAction.OFFER_SAVED,
                CommerceAction.OFFER_FALLBACK,
            },
        )

    async def test_low_heat_never_gets_proactive_offer(self):
        decision = await self.service.plan_commerce_turn(
            1, "hello", batch_number=100, heat="low", period="bar_shift"
        )
        self.assertEqual(decision.action, CommerceAction.NONE)

    async def test_confirmed_decline_context_matches_visible_commerce_state(self):
        self.assertFalse(
            await self.service.is_confirmed_decline(1, "not now", batch_number=1)
        )
        self.assertTrue(
            await self.service.is_confirmed_decline(
                1, "don't send me photos", batch_number=1
            )
        )

        offer = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=1, heat="rising", period="bar_shift"
        )
        await self.deliver(offer)
        self.assertTrue(
            await self.service.is_confirmed_decline(1, "not now", batch_number=2)
        )

    async def test_medium_or_cooling_heat_never_gets_proactive_offer(self):
        for heat in ("medium", "cooling"):
            with self.subTest(heat=heat):
                decision = await self.service.plan_commerce_turn(
                    1, "hello", batch_number=100, heat=heat, period="bar_shift"
                )
                self.assertEqual(decision.action, CommerceAction.NONE)

    async def test_bar_period_prefers_unlocked_bar_inventory(self):
        decision = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=1, heat="rising", period="bar_shift"
        )
        self.assertEqual(decision.offer.content_id, "mia_bar_001")
        self.assertEqual(decision.action, CommerceAction.OFFER_CURRENT)
        self.assertEqual(decision.item_locations, ("bar", "bathroom"))
        self.assertEqual(decision.current_locations, ("bar", "stockroom"))
        self.assertIn("my bar shift", decision.offered_item_description)
        self.assertIn("her bar shift", decision.offer.description)
        self.assertNotEqual(decision.offered_item_description, decision.offer.description)

    async def test_exact_past_only_match_is_a_saved_offer_with_spoken_copy(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "send me a pic from your pussy",
            batch_number=1,
            heat="high",
            period="evening_pregame",
        )

        self.assertEqual(decision.action, CommerceAction.OFFER_SAVED)
        self.assertEqual(decision.offer.content_id, "mia_private_nude_001")
        self.assertIn("from my bedroom", decision.offered_item_description)
        self.assertIn("from her bedroom", decision.offer.description)
        self.assertEqual(decision.current_context, "")
        self.assertIn("special moment", decision.brief)
        self.assertNotIn("Tyler", decision.brief)

    async def test_live_past_only_match_uses_grounded_blocker_and_fallback(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "show me a photo of your pussy right now",
            batch_number=1,
            heat="high",
            period="evening_pregame",
        )

        self.assertEqual(decision.action, CommerceAction.OFFER_FALLBACK)
        self.assertEqual(decision.offer.content_id, "mia_private_nude_001")
        self.assertIn("Tyler is on the couch", decision.current_context)
        self.assertIn("I cannot capture", decision.current_context)
        self.assertNotIn(" she ", f" {decision.current_context.lower()} ")
        self.assertIn("from my bedroom", decision.offered_item_description)

    async def test_live_fallback_when_alone_never_invents_a_person(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "show me a photo of your pussy right now",
            batch_number=1,
            heat="high",
            period="morning_home",
        )

        self.assertEqual(decision.action, CommerceAction.OFFER_FALLBACK)
        self.assertIn("I do not have a fresh version", decision.current_context)
        for invented in ("Tyler", "friend", "girl", "customer", "coworker"):
            self.assertNotIn(invented, decision.current_context)

    async def test_split_production_pussy_request_selects_exact_saved_photo(self):
        repository = FakeRepository()
        service = MediaCommerceService(
            load_media_catalog(PRODUCTION_CATALOG_PATH),
            repository=repository,
            random_source=random.Random(7),
        )
        intent = await service.classify_media_turn(
            1,
            ["send me a pic", "from your pussy"],
            batch_number=1,
        )
        decision = await service.plan_commerce_turn(
            1,
            "from your pussy",
            batch_number=1,
            heat="low",
            period="evening_pregame",
            intent=intent,
        )

        self.assertEqual(decision.offer.content_id, "mia_bedroom_001")
        self.assertEqual(decision.action, CommerceAction.OFFER_SAVED)
        self.assertEqual(decision.current_context, "")

    async def test_exact_alternate_video_beats_unrelated_requested_photo(self):
        catalog = load_media_catalog(CATALOG_PATH)
        club_video = catalog.require("mia_club_clip_001")
        pussy_video = dataclasses.replace(
            club_video,
            tags={**club_video.tags, "body_focus": ("pussy", "full_body")},
        )
        repository = FakeRepository()
        repository.unlocked.add("mia_private_nude_001")
        service = MediaCommerceService(
            MediaCatalog(
                pussy_video if item.id == pussy_video.id else item
                for item in catalog.items
            ),
            repository=repository,
            random_source=random.Random(7),
        )

        decision = await service.plan_commerce_turn(
            1,
            "send me a photo of your pussy",
            batch_number=1,
            heat="high",
            period="morning_home",
        )

        self.assertEqual(decision.offer.content_id, "mia_club_clip_001")
        self.assertEqual(decision.offer.media_type, "video")
        self.assertEqual(decision.action, CommerceAction.OFFER_FALLBACK)

    async def test_exact_alternate_photo_beats_unrelated_requested_video(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "send me a video behind the bar",
            batch_number=1,
            heat="high",
            period="morning_home",
        )

        self.assertEqual(decision.offer.content_id, "mia_bar_001")
        self.assertEqual(decision.offer.media_type, "photo")
        self.assertEqual(decision.action, CommerceAction.OFFER_FALLBACK)

    async def test_exact_requested_type_remains_ahead_of_exact_alternate_type(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "send me a photo of your boobs",
            batch_number=1,
            heat="high",
            period="club_night",
        )

        self.assertEqual(decision.offer.content_id, "mia_bar_001")
        self.assertEqual(decision.offer.media_type, "photo")

    async def test_current_location_breaks_tie_between_saved_exact_items(self):
        catalog = load_media_catalog(CATALOG_PATH)
        base = catalog.require("mia_home_pose_001")
        past = MediaPresentation(
            mode="past_only",
            periods=(),
            current_description=None,
            past_description="a saved playful full-body photo",
        )
        club = dataclasses.replace(
            base,
            id="aaa_club_saved",
            presentation=past,
            tags={**base.tags, "location": ("club",)},
        )
        home = dataclasses.replace(
            base,
            id="zzz_home_saved",
            presentation=past,
            tags={**base.tags, "location": ("home",)},
        )
        service = MediaCommerceService(
            MediaCatalog((club, home)),
            repository=FakeRepository(),
            random_source=random.Random(7),
        )

        decision = await service.plan_commerce_turn(
            1,
            "send me a photo",
            batch_number=1,
            heat="low",
            period="morning_home",
        )

        self.assertEqual(decision.offer.content_id, "zzz_home_saved")
        self.assertEqual(decision.action, CommerceAction.OFFER_SAVED)

    async def test_explicit_requested_location_beats_current_location(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "send a photo behind the bar",
            batch_number=1,
            heat="rising",
            period="morning_home",
        )
        self.assertEqual(decision.offer.content_id, "mia_bar_001")
        self.assertEqual(decision.action, CommerceAction.OFFER_SAVED)
        self.assertEqual(decision.current_context, "")
        self.assertNotIn("current context", decision.brief.lower())

    async def test_requested_body_from_other_location_beats_wrong_current_body(self):
        decision = await self.service.plan_commerce_turn(
            1,
            "send a photo of your boobs at home",
            batch_number=1,
            heat="rising",
            period="morning_home",
        )
        self.assertEqual(decision.offer.content_id, "mia_bar_001")
        self.assertEqual(decision.action, CommerceAction.OFFER_FALLBACK)

    async def test_unlocked_current_item_is_excluded_and_exact_saved_item_needs_no_excuse(self):
        self.repository.unlocked.add("mia_bar_001")
        decision = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=1, heat="rising", period="bar_shift"
        )
        self.assertEqual(decision.offer.content_id, "mia_home_pose_001")
        self.assertEqual(decision.action, CommerceAction.OFFER_SAVED)
        self.assertIn("saved", decision.brief.lower())
        self.assertEqual(decision.item_locations, ("home", "bedroom"))
        self.assertEqual(decision.current_locations, ("bar", "stockroom"))
        self.assertEqual(decision.current_context, "")

    async def test_fallback_explains_both_location_and_requested_type_mismatch(self):
        self.repository.unlocked.add("mia_club_clip_001")
        decision = await self.service.plan_commerce_turn(
            1, "send a video", batch_number=1, heat="rising", period="bar_shift"
        )
        self.assertEqual(decision.offer.media_type, "photo")
        self.assertEqual(decision.action, CommerceAction.OFFER_FALLBACK)
        self.assertIn("I do not have that as a video", decision.brief)
        self.assertIn("exactly that as a photo", decision.brief)

    async def test_generic_request_starts_photo_then_alternates_video(self):
        first = await self.service.plan_commerce_turn(
            1, "I want to see you", batch_number=1, heat="rising", period="bar_shift"
        )
        self.assertEqual(first.offer.media_type, "photo")
        await self.deliver(first)
        second = await self.service.plan_commerce_turn(
            1, "show me", batch_number=2, heat="rising", period="bar_shift"
        )
        self.assertEqual(second.offer.media_type, "video")

    async def test_proactive_photo_does_not_change_first_generic_request_type(self):
        proactive = await self.service.plan_commerce_turn(
            1, "flirting", batch_number=8, heat="high", period="bar_shift"
        )
        await self.deliver(proactive)
        generic = await self.service.plan_commerce_turn(
            1, "I want to see you", batch_number=9, heat="high", period="bar_shift"
        )
        self.assertEqual(generic.offer.media_type, "photo")

    async def test_new_similar_pool_is_used_before_repeating_exact_item(self):
        first = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=1, heat="rising", period="bar_shift"
        )
        await self.deliver(first)
        second = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=2, heat="rising", period="bar_shift"
        )
        self.assertEqual(second.offer.content_id, "mia_home_pose_001")

    async def test_direct_request_bypasses_repeat_cooldown(self):
        for batch, content_id in ((1, "mia_bar_001"), (2, "mia_home_pose_001")):
            record = OfferRecord(
                offer_id=batch,
                user_id=1,
                content_id=content_id,
                trigger="direct",
                action="offer_current",
                request_type="photo",
                description="test photo",
                batch_number=batch,
                price_tokens=5,
                status="delivered",
                created_at=float(batch),
                offered_at=float(batch),
            )
            self.repository.history.insert(0, record)
            self.repository.next_offer_id = 3
        direct = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=3, heat="rising", period="bar_shift"
        )
        self.assertIsNotNone(direct.offer)

    async def test_cancelled_reservation_never_enters_rotation_history(self):
        first = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=1, heat="rising", period="bar_shift"
        )
        await self.service.cancel_offer_reservation(first.offer.offer_id)
        self.assertEqual(self.repository.history, [])
        again = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=2, heat="rising", period="bar_shift"
        )
        self.assertEqual(again.offer.content_id, first.offer.content_id)
        self.assertGreaterEqual(self.repository.cancelled_stale, 2)

    async def test_soft_decline_reasks_after_30_to_40_batches_without_card(self):
        offer = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=1, heat="rising", period="bar_shift"
        )
        await self.deliver(offer)
        decline = await self.service.plan_commerce_turn(
            1, "not now", batch_number=2, heat="rising", period="bar_shift"
        )
        self.assertEqual(decline.action, CommerceAction.REACT_TO_DECLINE)
        self.assertIsNone(decline.offer)
        # The user's refusal is authoritative and is durable before Mia's
        # cosmetic reaction is generated/finalized.
        threshold = self.repository.state["sales_snooze_until_batch"]
        await self.service.mark_commerce_action_delivered(decline)
        self.assertEqual(
            self.repository.state["sales_snooze_until_batch"], threshold
        )
        self.assertGreaterEqual(threshold - 2, 30)
        self.assertLessEqual(threshold - 2, 40)
        early = await self.service.plan_commerce_turn(
            1, "flirting", batch_number=threshold - 1, heat="high", period="bar_shift"
        )
        self.assertEqual(early.action, CommerceAction.NONE)
        reask = await self.service.plan_commerce_turn(
            1, "flirting", batch_number=threshold, heat="high", period="bar_shift"
        )
        self.assertEqual(reask.action, CommerceAction.ASK_PERMISSION_AGAIN)
        self.assertIsNone(reask.offer)
        await self.service.mark_commerce_action_delivered(reask)
        accepted = await self.service.plan_commerce_turn(
            1, "yes please", batch_number=threshold + 1, heat="high", period="bar_shift"
        )
        self.assertIsNotNone(accepted.offer)
        self.assertEqual(accepted.offer.trigger, "permission_reask")

    async def test_decline_pause_survives_reaction_cancellation_and_replay(self):
        offer = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=1, heat="rising", period="bar_shift"
        )
        await self.deliver(offer)
        decline = await self.service.plan_commerce_turn(
            1, "not now", batch_number=2, heat="rising", period="bar_shift"
        )
        threshold = self.repository.state["sales_snooze_until_batch"]

        self.assertTrue(await self.service.cancel_commerce_action(decline))
        self.assertEqual(
            self.repository.state["sales_snooze_until_batch"], threshold
        )
        self.assertTrue(await self.service.mark_commerce_action_delivered(decline))
        self.assertTrue(await self.service.mark_commerce_action_delivered(decline))
        self.assertEqual(
            self.repository.state["sales_snooze_until_batch"], threshold
        )

    async def test_global_do_not_sell_decline_snoozes_without_a_recent_offer(self):
        for text in ("don't sell me content",):
            with self.subTest(text=text):
                self.repository.state.clear()
                decline = await self.service.plan_commerce_turn(
                    1,
                    text,
                    batch_number=20,
                    heat="rising",
                    period="bar_shift",
                )
                self.assertEqual(decline.action, CommerceAction.REACT_TO_DECLINE)
                self.assertEqual(decline.decline_kind, "soft")
                self.assertIsNone(decline.offer)
                await self.service.mark_commerce_action_delivered(decline)
                threshold = self.repository.state["sales_snooze_until_batch"]
                self.assertGreaterEqual(threshold - 20, 30)
                self.assertLessEqual(threshold - 20, 40)
                self.assertTrue(self.repository.state["sales_reask_pending"])

    async def test_stale_context_not_now_can_never_trigger_a_proactive_card(self):
        self.repository.history.append(
            OfferRecord(
                offer_id=1,
                user_id=1,
                content_id="mia_bar_001",
                trigger="direct",
                action="offer_current",
                request_type="photo",
                description="old photo",
                batch_number=1,
                price_tokens=5,
                status="delivered",
                created_at=1.0,
                offered_at=1.0,
            )
        )
        decision = await self.service.plan_commerce_turn(
            1, "not now", batch_number=10, heat="high", period="bar_shift"
        )
        self.assertEqual(decision.action, CommerceAction.NONE)
        self.assertIsNone(decision.offer)

    async def test_hard_decline_snoozes_exactly_100_batches(self):
        offer = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=10, heat="rising", period="bar_shift"
        )
        await self.deliver(offer)
        decline = await self.service.plan_commerce_turn(
            1,
            "never ask me again",
            batch_number=11,
            heat="high",
            period="bar_shift",
        )
        await self.service.mark_commerce_action_delivered(decline)
        self.assertEqual(self.repository.state["sales_snooze_until_batch"], 111)

    async def test_direct_request_bypasses_and_clears_pause_on_delivery(self):
        self.repository.state.update(
            sales_snooze_until_batch=100,
            sales_reask_pending=True,
            sales_reask_asked_at_batch=None,
        )
        direct = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=3, heat="rising", period="bar_shift"
        )
        self.assertIsNotNone(direct.offer)
        await self.deliver(direct)
        self.assertFalse(self.repository.state["sales_reask_pending"])
        self.assertIsNone(self.repository.state["sales_snooze_until_batch"])

    async def test_direct_request_clears_pause_even_when_no_item_exists(self):
        self.repository.state.update(
            sales_snooze_until_batch=100,
            sales_reask_pending=True,
            sales_reask_asked_at_batch=None,
        )
        self.repository.unlocked.update(item.id for item in self.service.catalog.items)
        decision = await self.service.plan_commerce_turn(
            1, "send a photo", batch_number=3, heat="rising", period="bar_shift"
        )
        self.assertEqual(
            decision.action, CommerceAction.MEDIA_REQUEST_UNAVAILABLE
        )
        self.assertIsNone(decision.offer)
        self.assertFalse(self.repository.state["sales_reask_pending"])
        self.assertIsNone(self.repository.state["sales_snooze_until_batch"])

    async def test_bare_no_after_permission_check_creates_a_fresh_snooze(self):
        self.repository.state.update(
            sales_snooze_until_batch=40,
            sales_reask_pending=True,
            sales_reask_asked_at_batch=40,
        )
        decline = await self.service.plan_commerce_turn(
            1, "no", batch_number=41, heat="high", period="bar_shift"
        )
        self.assertEqual(decline.action, CommerceAction.REACT_TO_DECLINE)
        self.assertIsNone(decline.offer)
        await self.service.mark_commerce_action_delivered(decline)
        self.assertGreaterEqual(
            self.repository.state["sales_snooze_until_batch"] - 41, 30
        )

    async def test_final_no_blocks_permission_reask_affirmative(self):
        for text in ("yes, no", "yes, but no"):
            with self.subTest(text=text):
                self.repository.state.update(
                    sales_snooze_until_batch=40,
                    sales_reask_pending=True,
                    sales_reask_asked_at_batch=40,
                )
                intent = await self.service.classify_media_turn(
                    1,
                    [text],
                    batch_number=41,
                )
                self.assertFalse(intent.requested)
                self.assertFalse(intent.affirmative)
                self.assertEqual(intent.decline_kind, "soft")

                decision = await self.service.plan_commerce_turn(
                    1,
                    text,
                    batch_number=41,
                    heat="high",
                    period="bar_shift",
                    intent=intent,
                )
                self.assertEqual(decision.action, CommerceAction.REACT_TO_DECLINE)
                self.assertIsNone(decision.offer)

    async def test_unrelated_reask_answer_keeps_sales_paused(self):
        self.repository.state.update(
            sales_snooze_until_batch=40,
            sales_reask_pending=True,
            sales_reask_asked_at_batch=40,
        )
        decision = await self.service.plan_commerce_turn(
            1, "how was work?", batch_number=41, heat="high", period="bar_shift"
        )
        self.assertEqual(decision.action, CommerceAction.NONE)
        self.assertTrue(self.repository.state["sales_reask_pending"])
        self.assertGreaterEqual(self.repository.state["sales_snooze_until_batch"], 71)

    async def test_hard_decline_extends_an_existing_soft_pause(self):
        self.repository.state.update(
            sales_snooze_until_batch=50,
            sales_reask_pending=True,
            sales_reask_asked_at_batch=None,
        )
        decline = await self.service.plan_commerce_turn(
            1,
            "never ask me again",
            batch_number=20,
            heat="high",
            period="bar_shift",
        )
        self.assertEqual(decline.action, CommerceAction.REACT_TO_DECLINE)
        await self.service.mark_commerce_action_delivered(decline)
        self.assertEqual(self.repository.state["sales_snooze_until_batch"], 120)


if __name__ == "__main__":
    unittest.main()
