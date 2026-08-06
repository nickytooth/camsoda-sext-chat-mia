import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from bot.chat_engine import ChatEngine, ChatResponse
from bot.heat import HeatState, HeatTurnResult
from bot.media_commerce import CommerceAction, CommerceDecision, MediaOffer
from bot.memory.stm import add_message, replace_assistant_message
from bot.output_guard import validate_mia_reply
from bot.persona import Persona


class StubProvider:
    async def generate(self, messages, temperature=None):
        return "unused"

    async def generate_simple(self, prompt):
        return "unused"


def make_engine(commerce_service=None) -> ChatEngine:
    provider = StubProvider()
    persona = Persona({"general": {"name": "Mia", "age": 26}})
    return ChatEngine(
        persona=persona,
        nsfw_persona=persona,
        nsfw_provider=provider,
        classifier_provider=provider,
        fallback_provider=provider,
        commerce_service=commerce_service,
    )


def offer_payload(offer_id=41):
    return {
        "offer_id": offer_id,
        "content_id": "mia_bar_001",
        "media_type": "photo",
        "price_tokens": 5,
        "aspect_ratio": 0.75,
        "duration_seconds": None,
        "explicitness": "suggestive",
        "description": "behind the bar during her shift",
        # Kept inside the trusted planner decision. It controls immediate
        # offer pacing but is intentionally absent from the browser payload.
        "trigger": "direct",
        # These fields must never cross the chat boundary.
        "full_key": "premium/mia_bar_001.jpg",
        "preview_key": "previews/mia_bar_001.webp",
        "access_url": "https://example.com/private.jpg",
    }


class MessageCompensationStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_returns_row_id_and_compensation_is_user_role_scoped(self):
        connection = AsyncMock()
        insert_cursor = AsyncMock()
        insert_cursor.fetchone.return_value = {"id": 731}
        replace_cursor = AsyncMock()
        replace_cursor.fetchone.return_value = {"id": 731}
        connection.execute.side_effect = [insert_cursor, replace_cursor]

        with patch(
            "bot.memory.stm.get_connection",
            new=AsyncMock(return_value=connection),
        ):
            message_id = await add_message(9, "assistant", "teaser")
            replaced = await replace_assistant_message(
                message_id, 9, "neutral fallback"
            )

        self.assertEqual(message_id, 731)
        self.assertTrue(replaced)
        insert_sql = connection.execute.await_args_list[0].args[0]
        replace_sql = connection.execute.await_args_list[1].args[0]
        self.assertIn("RETURNING id", insert_sql)
        self.assertIn("user_id = ?", replace_sql)
        self.assertIn("role = 'assistant'", replace_sql)
        self.assertIn("RETURNING id", replace_sql)


class CommerceAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_rejects_unbacked_media_claim_but_allows_offer_claim(self):
        primary = StubProvider()
        primary.generate = AsyncMock(return_value="here's a photo i picked for you")
        fallback = StubProvider()
        fallback.generate = AsyncMock(return_value="you have my full attention rn")
        persona = Persona({"general": {"name": "Mia", "age": 26}})
        engine = ChatEngine(
            persona=persona,
            nsfw_persona=persona,
            nsfw_provider=primary,
            classifier_provider=fallback,
            fallback_provider=fallback,
        )
        prompt = [
            {"role": "system", "content": "stay in character"},
            {"role": "user", "content": "talk to me"},
        ]

        without_offer = await engine._generate_with_fallback(
            primary,
            prompt,
            heat="high",
        )
        self.assertEqual(without_offer, "you have my full attention rn")
        fallback.generate.assert_awaited_once()

        primary.generate.reset_mock()
        primary.generate.return_value = "here's a photo i picked for you"
        fallback.generate.reset_mock()
        with_offer = await engine._generate_with_fallback(
            primary,
            prompt,
            heat="high",
            commerce_action="offer_current",
            commerce_media_type="photo",
            commerce_explicitness="suggestive",
        )
        self.assertEqual(with_offer, "here's a photo i picked for you")
        fallback.generate.assert_not_awaited()

    async def test_generation_retries_when_offer_copy_changes_the_real_location(self):
        primary = StubProvider()
        primary.generate = AsyncMock(
            return_value="i have a video. one from my bedroom, just for you"
        )
        fallback = StubProvider()
        fallback.generate = AsyncMock(
            return_value="i have a video from my bathroom, just for you"
        )
        persona = Persona({"general": {"name": "Mia", "age": 26}})
        engine = ChatEngine(
            persona=persona,
            nsfw_persona=persona,
            nsfw_provider=primary,
            classifier_provider=fallback,
            fallback_provider=fallback,
        )
        prompt = [
            {"role": "system", "content": "stay in character"},
            {"role": "user", "content": "show me the video"},
        ]

        result = await engine._generate_with_fallback(
            primary,
            prompt,
            heat="high",
            commerce_action="offer_fallback",
            commerce_media_type="video",
            commerce_explicitness="nude",
            commerce_media_description="a synthetic test clip from her bathroom",
            commerce_media_locations=("bathroom",),
        )

        self.assertEqual(result, "i have a video from my bathroom, just for you")
        fallback.generate.assert_awaited_once()
        correction = fallback.generate.await_args.args[0][0]["content"]
        self.assertIn("setting from the trusted commerce brief", correction)

    async def test_generation_retries_third_person_offer_copy_as_first_person(self):
        primary = StubProvider()
        primary.generate = AsyncMock(
            return_value="here's a photo she took from her bed"
        )
        fallback = StubProvider()
        fallback.generate = AsyncMock(
            return_value="here's a photo I took from my bed"
        )
        persona = Persona({"general": {"name": "Mia", "age": 26}})
        engine = ChatEngine(
            persona=persona,
            nsfw_persona=persona,
            nsfw_provider=primary,
            classifier_provider=fallback,
            fallback_provider=fallback,
        )

        result = await engine._generate_with_fallback(
            primary,
            [
                {"role": "system", "content": "stay in character"},
                {"role": "user", "content": "show me the photo"},
            ],
            heat="high",
            commerce_action="offer_saved",
            commerce_media_type="photo",
            commerce_explicitness="suggestive",
            commerce_media_description="a photo I took from my bed",
            commerce_media_locations=("bedroom",),
        )

        self.assertEqual(result, "here's a photo I took from my bed")
        fallback.generate.assert_awaited_once()
        correction = fallback.generate.await_args.args[0][0]["content"]
        self.assertIn("I, me, and my", correction)

    def test_batch_number_uses_processed_turns_not_raw_message_lifetime(self):
        self.assertEqual(
            ChatEngine._next_batch_number(
                {"total_messages": 7, "lifetime_user_messages": 200}
            ),
            8,
        )
        self.assertEqual(
            ChatEngine._next_batch_number(
                {"total_messages": 0, "lifetime_user_messages": 200}
            ),
            1,
        )

        class AsyncpgRecordLike:
            def __getitem__(self, key):
                return {"total_messages": 7, "lifetime_user_messages": 200}[key]

        self.assertEqual(
            ChatEngine._next_batch_number(AsyncpgRecordLike()),
            8,
        )

    async def test_two_hundred_raw_messages_in_one_debounce_are_one_processed_turn(self):
        engine = make_engine()
        engine._pending[7] = [f"raw {index}" for index in range(200)]
        engine._last_activity[7] = 95.0
        engine._process_sexting = AsyncMock(
            return_value=ChatResponse(messages=["one reply"])
        )
        callback = AsyncMock()
        clock = [100.0]

        async def fake_sleep(delay):
            clock[0] += delay

        with (
            patch("bot.config.SEXTING_DEBOUNCE_SECONDS", 5.0),
            patch("bot.chat_engine.time.time", side_effect=lambda: clock[0]),
            patch("bot.chat_engine.asyncio.sleep", side_effect=fake_sleep),
        ):
            await engine._batch_collect(7, callback)

        engine._process_sexting.assert_awaited_once()
        combined = engine._process_sexting.await_args.args[1]
        self.assertEqual(len(combined.splitlines()), 200)
        callback.assert_awaited_once()

    async def test_planner_receives_batch_heat_and_period_and_offer_is_whitelisted(self):
        service = type("Service", (), {})()
        service.plan_commerce_turn = AsyncMock(
            return_value={
                "action": "offer_current",
                "brief": "a real current bar photo",
                "offer": offer_payload(),
            }
        )
        engine = make_engine(service)

        turn = await engine._plan_commerce_turn(
            9,
            "show me a photo",
            batch_number=8,
            heat="rising",
            period="bar_shift",
        )

        service.plan_commerce_turn.assert_awaited_once_with(
            9,
            "show me a photo",
            batch_number=8,
            heat="rising",
            period="bar_shift",
        )
        self.assertEqual(turn.action, "offer_current")
        self.assertEqual(turn.media_offer["price_tokens"], 5)
        self.assertNotIn("full_key", turn.media_offer)
        self.assertNotIn("preview_key", turn.media_offer)
        self.assertNotIn("access_url", turn.media_offer)
        self.assertNotIn("trigger", turn.media_offer)

    async def test_adapter_accepts_the_production_decision_dataclasses(self):
        service = type("Service", (), {})()
        service.plan_commerce_turn = AsyncMock(
            return_value=CommerceDecision(
                action=CommerceAction.OFFER_FALLBACK,
                brief="customers are around, so offer an older bedroom photo",
                offer=MediaOffer(
                    offer_id=42,
                    content_id="fixture_bedroom_photo_001",
                    media_type="photo",
                    price_tokens=5,
                    aspect_ratio=0.75,
                    duration_seconds=None,
                    explicitness="suggestive",
                    description="a synthetic bedroom fixture photo",
                    trigger="direct",
                    action="offer_fallback",
                    request_type="photo",
                ),
                user_id=9,
                batch_number=8,
            )
        )
        engine = make_engine(service)

        turn = await engine._plan_commerce_turn(
            9,
            "show me a photo",
            batch_number=8,
            heat="rising",
            period="bar_shift",
        )

        self.assertEqual(turn.action, "offer_fallback")
        self.assertEqual(turn.media_offer["offer_id"], 42)
        self.assertEqual(
            turn.media_offer["content_id"], "fixture_bedroom_photo_001"
        )

    async def test_adapter_accepts_saved_offer_without_expanding_public_payload(self):
        saved = offer_payload(43)
        saved.update(
            content_id="fixture_saved_photo_001",
            description="a private photo from her bed",
            action="offer_saved",
        )
        service = type("Service", (), {})()
        service.plan_commerce_turn = AsyncMock(
            return_value={
                "action": "offer_saved",
                "offered_item_description": "a private photo from my bed",
                "offer": saved,
            }
        )
        engine = make_engine(service)

        turn = await engine._plan_commerce_turn(
            9,
            "show me a photo",
            batch_number=8,
            heat="high",
            period="evening_pregame",
        )

        self.assertEqual(turn.action, "offer_saved")
        self.assertEqual(turn.media_offer["offer_id"], 43)
        self.assertEqual(
            turn.media_offer["description"], "a private photo from her bed"
        )
        self.assertNotIn("action", turn.media_offer)
        self.assertNotIn("trigger", turn.media_offer)

    async def test_non_offer_actions_never_attach_a_card(self):
        for action in (
            "react_to_decline",
            "ask_permission_again",
            "media_request_unavailable",
            "acknowledge_unlock",
        ):
            with self.subTest(action=action):
                service = type("Service", (), {})()
                service.plan_commerce_turn = AsyncMock(
                    return_value={
                        "action": action,
                        "brief": "trusted text-only action",
                        "offer": offer_payload(),
                    }
                )
                engine = make_engine(service)

                turn = await engine._plan_commerce_turn(
                    9,
                    "not now",
                    batch_number=9,
                    heat="low",
                    period="bar_shift",
                )

                self.assertEqual(turn.action, action)
                self.assertIsNone(turn.media_offer)

    async def test_retired_confirmation_actions_fail_closed(self):
        for action in ("ask_media_confirmation", "cancel_media_confirmation"):
            with self.subTest(action=action):
                service = type("Service", (), {})()
                service.plan_commerce_turn = AsyncMock(
                    return_value={
                        "action": action,
                        "brief": "obsolete confirmation action",
                        "offer": offer_payload(),
                    }
                )
                engine = make_engine(service)

                turn = await engine._plan_commerce_turn(
                    9,
                    "show me a photo",
                    batch_number=9,
                    heat="high",
                    period="bar_shift",
                )

                self.assertEqual(turn.action, "none")
                self.assertIsNone(turn.media_offer)

    async def test_offer_fails_closed_if_price_or_safe_metadata_is_invalid(self):
        service = type("Service", (), {})()
        invalid = offer_payload()
        invalid["price_tokens"] = 1
        invalid["description"] = "fetch it from https://example.com/private.jpg"
        service.plan_commerce_turn = AsyncMock(
            return_value={
                "action": "offer_current",
                "brief": "an invalid offer",
                "offer": invalid,
            }
        )
        engine = make_engine(service)

        with self.assertLogs("bot.chat_engine", level="ERROR"):
            turn = await engine._plan_commerce_turn(
                9,
                "show me a photo",
                batch_number=8,
                heat="rising",
                period="bar_shift",
            )

        self.assertEqual(turn.action, "none")
        self.assertIsNone(turn.media_offer)

    async def test_offer_description_rejects_storage_paths_and_signed_query_fields(self):
        for unsafe in (
            "premium/mia/private.jpg",
            r"C:\private\mia.jpg",
            "file:/private/mia.jpg",
            "/home/mia/private.jpg",
            "~/private/mia.jpg",
            "../private/mia.jpg",
            r"\\server\share\mia.jpg",
            "X-Amz-Credential=temporary",
            "X-Amz-Signature=temporary",
        ):
            with self.subTest(unsafe=unsafe):
                service = type("Service", (), {})()
                invalid = offer_payload()
                invalid["description"] = unsafe
                service.plan_commerce_turn = AsyncMock(
                    return_value={
                        "action": "offer_current",
                        "brief": "an invalid offer",
                        "offer": invalid,
                    }
                )
                engine = make_engine(service)

                with self.assertLogs("bot.chat_engine", level="ERROR"):
                    turn = await engine._plan_commerce_turn(
                        9,
                        "show me a photo",
                        batch_number=8,
                        heat="rising",
                        period="bar_shift",
                    )

                self.assertEqual(turn.action, "none")
                self.assertIsNone(turn.media_offer)


class CommerceTurnPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.service = type("Service", (), {})()

        async def plan(user_id, text, *, batch_number, heat, period):
            self.events.append(("plan", batch_number, period))
            return {
                "action": "offer_current",
                "brief": "a playful mirror photo from behind the bar",
                "offer": offer_payload(),
            }

        async def finalize(offer_id):
            self.events.append(("finalize", offer_id))
            return offer_payload(int(offer_id))

        async def cancel(offer_id):
            self.events.append(("cancel", offer_id))
            return True

        async def finalize_action(decision):
            self.events.append(("finalize_action", decision["action"]))
            return True

        async def cancel_action(decision):
            self.events.append(("cancel_action", decision["action"]))
            return True

        self.service.plan_commerce_turn = plan
        self.service.mark_offer_delivered = finalize
        self.service.cancel_offer_reservation = cancel
        self.service.mark_commerce_action_delivered = finalize_action
        self.service.cancel_commerce_action = cancel_action
        self.engine = make_engine(self.service)

    def _patch_turn_dependencies(self, *, generated="i picked this one just for you"):
        state = {
            "last_message_at": 0,
            "total_messages": 7,
            "lifetime_user_messages": 200,
            "last_arc_id": None,
        }
        stm = [
            {
                "role": "user",
                "content": "show me a sexy photo",
                "timestamp": 1.0,
            }
        ]

        async def persist(*args, **kwargs):
            self.events.append(("persist", args[2]))
            return 731

        async def replace(message_id, user_id, content):
            self.events.append(("replace", message_id, content))
            return True

        self.engine._generate_with_fallback = AsyncMock(return_value=generated)
        return (
            patch("bot.chat_engine.maybe_summarize", new=AsyncMock(return_value=False)),
            patch("bot.chat_engine.maybe_compact", new=AsyncMock(return_value=False)),
            patch(
                "bot.chat_engine.get_engagement_state",
                new=AsyncMock(return_value=state),
            ),
            patch(
                "bot.chat_engine.get_recent_messages", new=AsyncMock(return_value=stm)
            ),
            patch(
                "bot.chat_engine.track_heat_batch",
                new=AsyncMock(
                    return_value=(
                        HeatTurnResult(
                            state=HeatState(
                                stage="rising",
                                progress=1,
                                last_sexual_at=1,
                                updated_at=1,
                                last_batch=8,
                                last_signal="sexual",
                            ),
                            response_heat="rising",
                            policy="normal",
                            sexual_batch=True,
                        ),
                        8,
                    )
                ),
            ),
            patch("bot.chat_engine.get_time_period", return_value="bar_shift"),
            patch(
                "bot.chat_engine.mood_for_message",
                return_value={"mood": "warm", "intensity": 1},
            ),
            patch("bot.chat_engine.get_arc_event", return_value=None),
            patch("bot.chat_engine.should_retrieve", return_value=False),
            patch("bot.chat_engine.get_facts", new=AsyncMock(return_value=[])),
            patch("bot.chat_engine.get_user_name", new=AsyncMock(return_value=None)),
            patch(
                "bot.chat_engine.build_prompt",
                new=AsyncMock(
                    return_value=[
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "show me a sexy photo"},
                    ]
                ),
            ),
            patch("bot.chat_engine.add_message", new=persist),
            patch("bot.chat_engine.replace_assistant_message", new=replace),
        )

    async def test_direct_current_offer_is_one_bubble_with_card_after_persistence(self):
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(
                generated=(
                    "ohhh, you really wanna cross that line?\n"
                    "you've got me tempted now\n"
                    "here's a photo i picked for you"
                )
            ):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(
                23, "show me a sexy photo"
            )

        self.assertEqual(self.events[0], ("plan", 8, "bar_shift"))
        persist_index = next(
            index for index, event in enumerate(self.events) if event[0] == "persist"
        )
        finalize_index = next(
            index for index, event in enumerate(self.events) if event[0] == "finalize"
        )
        self.assertLess(persist_index, finalize_index)
        self.assertEqual(response.commerce_action, "offer_current")
        self.assertEqual(response.media_offer["offer_id"], 41)
        self.assertNotIn("trigger", response.media_offer)
        self.assertEqual(len(response.messages), 1)
        self.assertIn("cross that line", response.messages[0])
        self.assertIn("here's a photo", response.messages[0])

    async def test_direct_saved_offer_is_one_bubble_with_card(self):
        saved_offer = offer_payload(43)
        saved_offer.update(
            content_id="mia_saved_bedroom_001",
            description="a private photo from her bed",
        )

        async def plan_saved(user_id, text, *, batch_number, heat, period):
            self.events.append(("plan", batch_number, period))
            return {
                "action": "offer_saved",
                "brief": "offer the exact saved photo without a current excuse",
                "current_context": "",
                "offered_item_description": "a private photo from my bed",
                "offer": saved_offer,
            }

        async def finalize_saved(offer_id):
            self.events.append(("finalize", offer_id))
            return saved_offer

        self.service.plan_commerce_turn = plan_saved
        self.service.mark_offer_delivered = finalize_saved
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(
                generated=(
                    "ohhh, you really wanna see it?\n"
                    "i kept this private photo from my bed for a special moment"
                )
            ):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(
                23, "show me a sexy photo"
            )

        self.assertEqual(response.commerce_action, "offer_saved")
        self.assertEqual(response.media_offer["offer_id"], 43)
        self.assertEqual(len(response.messages), 1)
        visible = response.messages[0]
        self.assertIn("kept this private photo", visible)
        self.assertNotIn("Tyler", visible)
        self.assertNotIn("can't take", visible)

    async def test_proactive_saved_offer_is_also_one_bubble_with_card(self):
        saved_offer = offer_payload(44)
        saved_offer.update(
            content_id="mia_saved_bedroom_001",
            description="a private photo from her bed",
            trigger="proactive",
        )

        async def plan_saved(user_id, text, *, batch_number, heat, period):
            self.events.append(("plan", batch_number, period))
            return {
                "action": "offer_saved",
                "brief": "offer the saved photo confidently",
                "current_context": "",
                "offered_item_description": "a private photo from my bed",
                "offer": saved_offer,
            }

        async def finalize_saved(offer_id):
            self.events.append(("finalize", offer_id))
            return saved_offer

        self.service.plan_commerce_turn = plan_saved
        self.service.mark_offer_delivered = finalize_saved
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(
                generated=(
                    "i've been thinking about you\n"
                    "so i kept this private photo from my bed\n"
                    "maybe tonight is the special moment"
                )
            ):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(23, "i missed you")

        self.assertEqual(response.commerce_action, "offer_saved")
        self.assertEqual(response.media_offer["offer_id"], 44)
        self.assertEqual(len(response.messages), 1)
        self.assertIn("private photo from my bed", response.messages[0])

    async def test_direct_fallback_offer_is_two_bubbles_with_card(self):
        fallback_offer = offer_payload(42)
        fallback_offer.update(
            content_id="mia_bathroom_001",
            description="a private photo from her bathroom",
        )

        async def plan_fallback(user_id, text, *, batch_number, heat, period):
            self.events.append(("plan", batch_number, period))
            return {
                "action": "offer_fallback",
                "brief": "customers are around; offer the real bathroom alternative",
                "current_context": "customers are around at the bar",
                "offered_item_description": fallback_offer["description"],
                "offer": fallback_offer,
            }

        async def finalize_fallback(offer_id):
            self.events.append(("finalize", offer_id))
            return fallback_offer

        self.service.plan_commerce_turn = plan_fallback
        self.service.mark_offer_delivered = finalize_fallback
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(
                generated=(
                    "ohhh, you really wanna go there?\n"
                    "here's an old one i picked for you"
                )
            ):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(
                23, "show me a sexy photo"
            )

        self.assertEqual(response.commerce_action, "offer_fallback")
        self.assertIsNotNone(response.media_offer)
        self.assertEqual(response.media_offer["offer_id"], 42)
        self.assertNotIn("trigger", response.media_offer)
        self.assertEqual(len(response.messages), 2)
        visible = "\n".join(response.messages)
        self.assertIn("customers are around at the bar", visible)
        self.assertIn("private photo from my bathroom", visible)

    async def test_permission_reask_is_committed_only_after_question_is_persisted(self):
        async def plan_reask(user_id, text, *, batch_number, heat, period):
            self.events.append(("plan", batch_number, period))
            return {
                "action": "ask_permission_again",
                "brief": "the sales snooze expired; ask softly once",
                "offer": None,
            }

        self.service.plan_commerce_turn = plan_reask
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(
                generated="do you maybe wanna see me now?"
            ):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(23, "i missed you")

        persist_index = next(
            index for index, event in enumerate(self.events) if event[0] == "persist"
        )
        finalize_index = next(
            index
            for index, event in enumerate(self.events)
            if event[0] == "finalize_action"
        )
        self.assertLess(persist_index, finalize_index)
        self.assertEqual(response.commerce_action, "ask_permission_again")
        self.assertIsNone(response.media_offer)

    async def test_failed_reask_state_is_replaced_before_delivery(self):
        async def plan_reask(user_id, text, *, batch_number, heat, period):
            self.events.append(("plan", batch_number, period))
            return {
                "action": "ask_permission_again",
                "brief": "ask softly once",
                "offer": None,
            }

        async def fail_finalize_action(decision):
            self.events.append(("finalize_action_failed", decision["action"]))
            return False

        self.service.plan_commerce_turn = plan_reask
        self.service.mark_commerce_action_delivered = fail_finalize_action
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(
                generated="do you maybe wanna see me now?"
            ):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(23, "i missed you")

        self.assertIsNone(response.commerce_action)
        self.assertIsNone(response.media_offer)
        self.assertNotEqual(response.messages, ["do you maybe wanna see me now?"])
        replacement = next(event for event in self.events if event[0] == "replace")
        self.assertEqual("\n".join(response.messages), replacement[2])

    async def test_failed_generation_cancels_reservation_and_returns_no_card(self):
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(generated=""):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(
                23, "show me a sexy photo"
            )

        self.assertIn(("cancel", "41"), self.events)
        self.assertFalse(any(event[0] == "finalize" for event in self.events))
        self.assertIsNone(response.media_offer)
        self.assertIsNone(response.commerce_action)

    async def test_failed_teaser_persistence_cancels_reservation(self):
        patches = list(self._patch_turn_dependencies())
        patches[-2] = patch(
            "bot.chat_engine.add_message",
            new=AsyncMock(side_effect=RuntimeError("database down")),
        )

        with self.assertRaises(RuntimeError):
            with ExitStack() as stack:
                for turn_patch in patches:
                    stack.enter_context(turn_patch)
                await self.engine._process_sexting(23, "show me a sexy photo")

        self.assertIn(("cancel", "41"), self.events)
        self.assertFalse(any(event[0] == "finalize" for event in self.events))

    async def test_failed_offer_finalize_cancels_and_does_not_emit_card(self):
        async def fail_finalize(offer_id):
            self.events.append(("finalize", offer_id))
            return None

        self.service.mark_offer_delivered = fail_finalize
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies():
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(
                23, "show me a sexy photo"
            )

        self.assertIn(("finalize", "41"), self.events)
        self.assertIn(("cancel", "41"), self.events)
        self.assertIsNone(response.media_offer)
        self.assertIsNone(response.commerce_action)
        self.assertNotEqual(response.messages, ["i picked this one just for you"])
        replacement = next(event for event in self.events if event[0] == "replace")
        self.assertEqual(replacement[1], 731)
        self.assertEqual("\n".join(response.messages), replacement[2])
        self.assertTrue(
            validate_mia_reply("\n".join(response.messages), heat="rising").ok
        )


if __name__ == "__main__":
    unittest.main()
