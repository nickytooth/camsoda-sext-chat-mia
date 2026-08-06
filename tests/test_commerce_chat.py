import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from bot.chat_engine import ChatEngine, ChatResponse
from bot.heat import HeatState, HeatTurnResult
from bot.media_commerce import CommerceAction, CommerceDecision, MediaOffer
from bot.memory.stm import add_message, replace_assistant_message
from bot.output_guard import correction_prompt, validate_mia_reply
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
    def test_offer_guard_rejects_inventory_voice_but_allows_natural_mismatch_copy(self):
        for jargon in (
            "this is the closest available match",
            "i picked the selected alternative",
            "this is the available type alternative",
            "i do not have the exact ass variation",
            "this was the top result in my inventory",
            "this was the first search result",
        ):
            with self.subTest(jargon=jargon):
                result = validate_mia_reply(
                    jargon,
                    heat="high",
                    commerce_action="offer_fallback",
                )
                self.assertIn("media_offer_inventory_voice", result.reasons)

        natural = validate_mia_reply(
            "not quite the angle you asked for, but this one is still trouble",
            heat="high",
            commerce_action="offer_fallback",
        )
        self.assertNotIn("media_offer_inventory_voice", natural.reasons)

        retry = correction_prompt(("media_offer_inventory_voice",), "high")
        self.assertIn("casual texting voice", retry)
        self.assertIn("closest available match", retry)
        self.assertIn("angle", retry)

    def test_offer_guard_rejects_a_verbatim_catalog_label(self):
        label = "a private nude selfie from my bed"
        robotic = validate_mia_reply(
            f"god... i picked {label} for you",
            heat="high",
            commerce_action="offer_saved",
            commerce_media_type="photo",
            commerce_explicitness="nude",
            commerce_media_description=label,
            commerce_media_locations=("bedroom",),
        )
        self.assertIn("media_offer_catalog_voice", robotic.reasons)

        determiner_swap = validate_mia_reply(
            "not quite the angle, but i got this private nude selfie from my bed",
            heat="high",
            commerce_action="offer_fallback",
            commerce_media_type="photo",
            commerce_explicitness="nude",
            commerce_media_description=label,
            commerce_media_locations=("bedroom",),
        )
        self.assertIn("media_offer_catalog_voice", determiner_swap.reasons)

        origin_determiner_swap = validate_mia_reply(
            "not quite the angle, but i got this private nude selfie from the bed",
            heat="high",
            commerce_action="offer_fallback",
            commerce_media_type="photo",
            commerce_explicitness="nude",
            commerce_media_description=label,
            commerce_media_locations=("bedroom",),
        )
        self.assertIn(
            "media_offer_catalog_voice",
            origin_determiner_swap.reasons,
        )

        natural = validate_mia_reply(
            "god... open this before i change my mind",
            heat="high",
            commerce_action="offer_saved",
            commerce_media_type="photo",
            commerce_explicitness="nude",
            commerce_media_description=label,
            commerce_media_locations=("bedroom",),
        )
        self.assertNotIn("media_offer_catalog_voice", natural.reasons)

        retry = correction_prompt(("media_offer_catalog_voice",), "high")
        self.assertIn("Do not repeat", retry)
        self.assertIn("this pic", retry)

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

    async def test_direct_current_offer_preserves_one_natural_question_or_statement(self):
        natural_openers = (
            "ohhh... you really wanna cross that line? 😈",
            "wow... you really don't waste time 😈",
            "who taught you to ask like that?",
            "you don't play around, do you?",
            "jesus... you went right for it",
            "straight for my pussy, huh?",
            "you are making me reckless tonight",
            "hold on... who taught you to ask like that?",
            "you sure know how to ask for trouble",
        )
        for generated in natural_openers:
            with self.subTest(generated=generated):
                self.events.clear()
                with ExitStack() as stack:
                    for turn_patch in self._patch_turn_dependencies(
                        generated=generated
                    ):
                        stack.enter_context(turn_patch)
                    response = await self.engine._process_sexting(
                        23, "wanna see your ass"
                    )

                self.assertEqual(response.commerce_action, "offer_current")
                self.assertEqual(len(response.messages), 1)
                self.assertEqual(response.messages[0], generated)
                self.assertLessEqual(response.messages[0].count("?"), 1)

    async def test_direct_current_offer_replaces_confirmation_or_non_reactive_copy(self):
        invalid_openers = (
            "are you sure? do you want me to send this?",
            "want me to show it to you?",
            "ohhh... are you sure you want this?",
            "wow... do you really want it?",
            "ohhh... wanna see it?",
            "you sure?",
            "you want to see it?",
            "want a look?",
            "give me a sec and i will show it",
            "one sec... i will send it",
            "i will send it in a minute",
            "okay... if you're sure, i'll send it",
            "fuck yes babe look how ready i am for you 😈",
        )
        for generated in invalid_openers:
            with self.subTest(generated=generated):
                self.events.clear()
                with ExitStack() as stack:
                    for turn_patch in self._patch_turn_dependencies(
                        generated=generated
                    ):
                        stack.enter_context(turn_patch)
                    response = await self.engine._process_sexting(
                        23, "wanna see your ass"
                    )

                self.assertEqual(response.commerce_action, "offer_current")
                self.assertIsNotNone(response.media_offer)
                self.assertEqual(len(response.messages), 1)
                self.assertNotEqual(response.messages[0], generated)
                self.assertLessEqual(response.messages[0].count("?"), 1)
                self.assertNotRegex(
                    response.messages[0].lower(),
                    r"(?:are you sure|want me to|should i)",
                )

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
                    "ohhh, you really know what to ask for\n"
                    "open this before i change my mind"
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
        self.assertIn("open this", visible)
        self.assertNotIn("special moment", visible.lower())
        self.assertNotIn("Tyler", visible)
        self.assertNotIn("can't take", visible)

    async def test_direct_saved_offer_replaces_body_description_with_bold_reaction(self):
        saved_offer = offer_payload(43)
        saved_offer.update(
            content_id="mia_saved_bedroom_001",
            description="a private nude photo from her bed",
        )

        async def plan_saved(user_id, text, *, batch_number, heat, period):
            return {
                "action": "offer_saved",
                "brief": "offer the exact saved photo without a current excuse",
                "current_context": "",
                "offered_item_description": "a private nude photo from my bed",
                "offer": saved_offer,
            }

        async def finalize_saved(offer_id):
            return saved_offer

        self.service.plan_commerce_turn = plan_saved
        self.service.mark_offer_delivered = finalize_saved
        generated = "fuck yes babe look how ready i am for you 😈"
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(generated=generated):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(
                23, "show me your pussy babe"
            )

        self.assertEqual(response.commerce_action, "offer_saved")
        self.assertIsNotNone(response.media_offer)
        self.assertEqual(len(response.messages), 1)
        self.assertNotEqual(response.messages[0], generated)
        self.assertIn("no hesitation", response.messages[0].lower())
        self.assertLessEqual(response.messages[0].count("?"), 1)

    async def test_permission_reask_acceptance_never_asks_for_confirmation_again(self):
        accepted_offer = offer_payload(45)
        accepted_offer["trigger"] = "permission_reask"

        async def plan_accepted(user_id, text, *, batch_number, heat, period):
            return {
                "action": "offer_saved",
                "brief": "he already accepted the permission check",
                "offered_item_description": "a private photo from my bed",
                "offer": accepted_offer,
            }

        async def finalize_accepted(offer_id):
            return accepted_offer

        self.service.plan_commerce_turn = plan_accepted
        self.service.mark_offer_delivered = finalize_accepted
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(
                generated="ohhh... are you sure you want this?"
            ):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(23, "yes babe")

        self.assertEqual(response.commerce_action, "offer_saved")
        self.assertIsNotNone(response.media_offer)
        self.assertEqual(len(response.messages), 1)
        self.assertEqual(response.messages[0].count("?"), 0)
        self.assertNotRegex(response.messages[0].lower(), r"(?:are you sure|confirm)")

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
        self.assertNotIn("private photo from my bed", response.messages[0])
        self.assertNotIn("special moment", response.messages[0])
        self.assertTrue(response.messages[0].strip())

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
        self.assertIn("saved pic", visible)
        self.assertNotIn("private photo from my bathroom", visible)

    async def _run_natural_fallback_copy(
        self,
        *,
        current_context,
        generated,
        media_type="photo",
        description="a private photo from her bedroom",
        fallback_kind="semantic_near_match",
        requested_detail="",
        requested_media_type="",
        live_capture_blocker="",
        live_capture_blocker_kind="",
        current_locations=(),
        trigger="direct",
    ):
        fallback_offer = offer_payload(45)
        fallback_offer.update(
            content_id="mia_fallback_001",
            media_type=media_type,
            price_tokens=10 if media_type == "video" else 5,
            duration_seconds=15 if media_type == "video" else None,
            description=description,
            trigger=trigger,
        )

        async def plan_fallback(user_id, text, *, batch_number, heat, period):
            self.events.append(("plan", batch_number, period))
            return {
                "action": "offer_fallback",
                "brief": "offer the truthful fallback without inventory jargon",
                "current_context": current_context,
                "offered_item_description": description,
                "fallback_kind": fallback_kind,
                "requested_detail": requested_detail,
                "requested_media_type": requested_media_type,
                "live_capture_blocker": live_capture_blocker,
                "live_capture_blocker_kind": live_capture_blocker_kind,
                "current_locations": current_locations,
                "offer": fallback_offer,
            }

        async def finalize_fallback(offer_id):
            self.events.append(("finalize", offer_id))
            return fallback_offer

        self.service.plan_commerce_turn = plan_fallback
        self.service.mark_offer_delivered = finalize_fallback
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(generated=generated):
                stack.enter_context(turn_patch)
            return await self.engine._process_sexting(23, "show me a sexy photo")

    def assertNoInventoryJargon(self, text):
        lowered = text.lower()
        for phrase in (
            "variation",
            "closest available match",
            "selected alternative",
            "available type alternative",
        ):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, lowered)

    async def test_semantic_fallback_keeps_two_bubbles_without_inventory_jargon(self):
        response = await self._run_natural_fallback_copy(
            current_context=(
                "she does not have the exact ass variation; this is the closest "
                "available match"
            ),
            generated=(
                "straight to my ass? you're getting brave\n"
                "not quite the angle you asked for, but this one is not a downgrade"
            ),
            requested_detail="ass",
        )

        self.assertEqual(response.commerce_action, "offer_fallback")
        self.assertEqual(len(response.messages), 2)
        self.assertNoInventoryJargon(response.messages[1])
        self.assertTrue(
            any(
                phrase in response.messages[1].lower()
                for phrase in ("angle", "shot", "not quite", "asked for")
            ),
            response.messages[1],
        )

    async def test_direct_fallback_limits_questions_across_both_bubbles(self):
        response = await self._run_natural_fallback_copy(
            current_context="this is not quite the ass shot he asked for",
            generated=(
                "you really wanna go there?\n"
                "not quite the ass shot you asked for... want a look?"
            ),
            requested_detail="ass",
        )

        self.assertEqual(response.commerce_action, "offer_fallback")
        self.assertEqual(len(response.messages), 2)
        visible = "\n".join(response.messages)
        self.assertLessEqual(visible.count("?"), 1, visible)
        self.assertNotIn("want a look", visible.lower())

    async def test_direct_fallback_drops_confirmation_and_wait_language(self):
        generated_variants = (
            (
                "wow... straight to that huh\n"
                "not quite the ass shot you asked for... want a look?"
            ),
            (
                "wait for me to send it later\n"
                "not quite the ass shot you asked for... but this one is trouble"
            ),
            (
                "wow... straight to that huh\n"
                "not quite the ass shot you asked for... wait for me to send it later"
            ),
        )
        for generated in generated_variants:
            with self.subTest(generated=generated):
                response = await self._run_natural_fallback_copy(
                    current_context="this is not quite the ass shot he asked for",
                    generated=generated,
                    requested_detail="ass",
                )

                self.assertEqual(response.commerce_action, "offer_fallback")
                self.assertEqual(len(response.messages), 2)
                visible = "\n".join(response.messages).lower()
                self.assertNotRegex(
                    visible,
                    r"(?:you sure|want (?:to see it|a look)|wait for|hold on|"
                    r"send it later)",
                )

    async def test_permission_reask_fallback_contains_no_questions_or_new_gate(self):
        response = await self._run_natural_fallback_copy(
            current_context="this is not quite the ass shot he asked for",
            generated=(
                "you sure?\n"
                "not quite the ass shot you asked for... you want to see it?"
            ),
            requested_detail="ass",
            trigger="permission_reask",
        )

        self.assertEqual(response.commerce_action, "offer_fallback")
        self.assertEqual(len(response.messages), 2)
        visible = "\n".join(response.messages)
        self.assertEqual(visible.count("?"), 0, visible)
        self.assertNotRegex(
            visible.lower(),
            r"(?:you sure|want (?:to see it|a look)|are you sure|confirm)",
        )

    async def test_semantic_fallback_replaces_model_inventory_voice(self):
        response = await self._run_natural_fallback_copy(
            current_context="this is not quite the ass shot he asked for",
            generated=(
                "straight to my ass? you're getting brave\n"
                "i do not have the exact ass variation; this is the closest "
                "available match"
            ),
            requested_detail="ass",
        )

        self.assertEqual(len(response.messages), 2)
        self.assertNoInventoryJargon(response.messages[1])
        self.assertIn("ass", response.messages[1].lower())

    async def test_semantic_fallback_replaces_catalog_label_echo(self):
        generated_variants = (
            "this isn't quite the ass shot but i got this private nude clip "
            "from my bathroom",
            "this isn't quite the ass shot but i got this private nude clip "
            "from the bathroom",
            "this isn't quite the ass shot but this private nude clip is better",
        )
        for generated_second in generated_variants:
            with self.subTest(generated_second=generated_second):
                response = await self._run_natural_fallback_copy(
                    current_context="this is not quite the ass shot he asked for",
                    generated=(
                        "ohhh you really wanna see that huh?\n"
                        f"{generated_second}"
                    ),
                    media_type="video",
                    description="a private nude clip from my bathroom",
                    fallback_kind="semantic_near_match",
                    requested_detail="ass",
                )

                self.assertEqual(len(response.messages), 2)
                second = response.messages[1].lower()
                self.assertIn("ass", second)
                self.assertNotIn("private nude clip", second)
                self.assertNotIn("bathroom", second)

    async def test_semantic_fallback_preserves_natural_clip_reference(self):
        natural_variants = (
            "not quite the ass shot... but this clip is gonna make up for it",
            "not quite the ass shot... but this private clip is better",
            "not quite the ass shot... but this nude clip is better",
            "not quite the ass shot... but this clip from my bathroom is better",
        )
        for natural in natural_variants:
            with self.subTest(natural=natural):
                response = await self._run_natural_fallback_copy(
                    current_context="this is not quite the ass shot he asked for",
                    generated=f"ohhh you really wanna see that huh?\n{natural}",
                    media_type="video",
                    description="a private nude clip from my bathroom",
                    fallback_kind="semantic_near_match",
                    requested_detail="ass",
                )

                self.assertEqual(response.messages[1], natural)

    async def test_non_live_fallback_drops_an_invented_nearby_person(self):
        response = await self._run_natural_fallback_copy(
            current_context="this is not quite the ass shot he asked for",
            generated=(
                "straight to my ass? you're getting brave\n"
                "Tyler's beside me, so it's not quite the angle you asked for"
            ),
            requested_detail="ass",
        )

        self.assertEqual(len(response.messages), 2)
        self.assertNotIn("tyler", response.messages[1].lower())
        self.assertIn("ass", response.messages[1].lower())

    async def test_type_swap_names_both_types_without_inventory_jargon(self):
        response = await self._run_natural_fallback_copy(
            current_context=(
                "she does not have a matching photo; this video is the closest "
                "available type alternative"
            ),
            generated=(
                "you really want to see that?\n"
                "i don't have that as a photo, but the video is even better"
            ),
            media_type="video",
            description="a private video from her bathroom",
            fallback_kind="type_swap",
            requested_media_type="photo",
        )

        self.assertEqual(len(response.messages), 2)
        second = response.messages[1].lower()
        self.assertIn("photo", second)
        self.assertIn("video", second)
        self.assertNoInventoryJargon(second)

    async def test_live_fallback_preserves_the_grounded_blocker_naturally(self):
        response = await self._run_natural_fallback_copy(
            current_context=(
                "Tyler is on the couch in the next room, so she cannot capture "
                "the requested fresh version right now"
            ),
            generated=(
                "ohhh, you want a fresh one right now?\n"
                "Tyler's on the couch in the next room, so i'm not taking one rn... "
                "but i have something saved for you"
            ),
            fallback_kind="live_blocked",
            live_capture_blocker="Tyler is on the couch in the next room",
        )

        self.assertEqual(len(response.messages), 2)
        second = response.messages[1].lower()
        self.assertIn("tyler", second)
        self.assertTrue("couch" in second or "next room" in second, second)
        self.assertNoInventoryJargon(second)

    async def test_live_blocked_does_not_let_bar_term_ground_invented_tyler(self):
        response = await self._run_natural_fallback_copy(
            current_context=(
                "customers are around at the bar, so she cannot capture the "
                "requested fresh version right now"
            ),
            generated=(
                "ohhh, you want me to take one here at the bar?\n"
                "Tyler's at the bar, so i can't take a fresh one rn... but i have "
                "something saved"
            ),
            fallback_kind="live_blocked",
            live_capture_blocker="customers are around at the bar",
        )

        self.assertEqual(len(response.messages), 2)
        visible = "\n".join(response.messages).lower()
        self.assertNotIn("tyler", visible)
        self.assertTrue(
            "customers" in response.messages[1].lower()
            or "bar" in response.messages[1].lower(),
            response.messages[1],
        )

    async def test_live_blocked_rejects_roommate_when_tyler_is_the_blocker(self):
        response = await self._run_natural_fallback_copy(
            current_context=(
                "Tyler is on the couch in the next room, so she cannot capture "
                "the requested fresh version right now"
            ),
            generated=(
                "ohhh, you want a fresh one right now?\n"
                "my roommate's on the couch, so i can't take one rn... but i have "
                "something saved"
            ),
            fallback_kind="live_blocked",
            live_capture_blocker="Tyler is on the couch in the next room",
        )

        self.assertEqual(len(response.messages), 2)
        second = response.messages[1].lower()
        self.assertNotIn("roommate", second)
        self.assertIn("tyler", second)
        self.assertTrue("couch" in second or "next room" in second, second)

    async def test_live_blocked_rejects_a_wrong_location_with_the_right_crowd(self):
        response = await self._run_natural_fallback_copy(
            current_context="customers are around at the bar",
            generated=(
                "ohhh, you want a fresh one?\n"
                "the gym is full of people so i can't take a fresh one rn... but open this"
            ),
            fallback_kind="live_blocked",
            live_capture_blocker="customers are around at the bar",
            live_capture_blocker_kind="work_crowd",
            current_locations=("bar", "stockroom"),
        )

        visible = "\n".join(response.messages).lower()
        self.assertNotIn("gym", visible)
        self.assertIn("customers", response.messages[1].lower())

    async def test_live_blocked_rejects_tyler_at_an_ungrounded_location(self):
        response = await self._run_natural_fallback_copy(
            current_context="Tyler is asleep beside her",
            generated=(
                "ohhh, you want a fresh one?\n"
                "Tyler's at the bar so i can't take a fresh one rn... but open this"
            ),
            fallback_kind="live_blocked",
            live_capture_blocker="Tyler is asleep beside her",
            live_capture_blocker_kind="tyler",
            current_locations=("bedroom", "home"),
        )

        second = response.messages[1].lower()
        self.assertNotIn("bar", second)
        self.assertIn("tyler", second)

    async def test_live_blocked_replaces_unsafe_person_and_location_in_lead_bubble(self):
        response = await self._run_natural_fallback_copy(
            current_context=(
                "customers are around at the bar, so she cannot capture the "
                "requested fresh version right now"
            ),
            generated=(
                "i'm at the beach with my friend and you want a fresh one?\n"
                "customers are all around me at the bar, so i can't take one rn... "
                "but i have something saved"
            ),
            fallback_kind="live_blocked",
            live_capture_blocker="customers are around at the bar",
        )

        self.assertEqual(len(response.messages), 2)
        visible = "\n".join(response.messages).lower()
        self.assertNotIn("friend", visible)
        self.assertNotIn("beach", visible)
        self.assertTrue(
            "customers" in response.messages[1].lower()
            or "bar" in response.messages[1].lower(),
            response.messages[1],
        )

    async def test_live_unavailable_replaces_invented_person_and_current_scene(self):
        response = await self._run_natural_fallback_copy(
            current_context=(
                "she has no fresh version in the requested angle right now"
            ),
            generated=(
                "i'm at the club with my friend and you want one right now?\n"
                "my friend's beside me at the club, so i can't take a fresh one... "
                "but i have something saved"
            ),
            fallback_kind="live_unavailable",
        )

        self.assertEqual(len(response.messages), 2)
        visible = "\n".join(response.messages).lower()
        self.assertNotIn("friend", visible)
        self.assertNotIn("club", visible)
        self.assertTrue(
            any(term in response.messages[1].lower() for term in ("fresh", "right now", "rn")),
            response.messages[1],
        )

    async def test_live_unavailable_rejects_person_without_location_grammar(self):
        response = await self._run_natural_fallback_copy(
            current_context="there is no fresh version right now",
            generated=(
                "ohhh, you want a fresh one?\n"
                "my friend just walked in so i can't take a fresh one rn... but open this"
            ),
            fallback_kind="live_unavailable",
        )

        visible = "\n".join(response.messages).lower()
        self.assertNotIn("friend", visible)
        self.assertIn("fresh", response.messages[1].lower())

    async def test_non_live_fallback_rejects_context_claims_after_the_pivot(self):
        cases = (
            (
                "live_unavailable",
                "i don't have a fresh one rn but Tyler is here",
                {"current_context": "there is no fresh one right now"},
            ),
            (
                "semantic_near_match",
                "not quite the angle but Tyler's right here",
                {
                    "current_context": "this is not quite the ass shot",
                    "requested_detail": "ass",
                },
            ),
            (
                "type_swap",
                "i don't have that as a photo but Tyler's showing you the video instead",
                {
                    "current_context": "she has it as a video rather than a photo",
                    "media_type": "video",
                    "description": "a private video from her bathroom",
                    "requested_media_type": "photo",
                },
            ),
        )
        for fallback_kind, generated_second, kwargs in cases:
            for suffix in (
                "",
                " but i'm at the gym now",
                " but at the gym right now",
                " but at the gym now",
                " but at the gym today",
                " but at the gym atm",
            ):
                with self.subTest(
                    fallback_kind=fallback_kind,
                    generated_second=generated_second,
                    suffix=suffix,
                ):
                    response = await self._run_natural_fallback_copy(
                        generated=f"you really went there?\n{generated_second}{suffix}",
                        fallback_kind=fallback_kind,
                        **kwargs,
                    )
                    visible = "\n".join(response.messages).lower()
                    self.assertNotIn("tyler", visible)
                    self.assertNotIn("gym", visible)

    async def test_live_blocked_rejects_extra_context_after_the_pivot(self):
        for suffix in (
            " but Tyler is here too",
            " but i'm at the gym now",
            " but at the gym right now",
            " but my sister is here too",
            " but Taylor is beside me too",
            " but Sarah is waiting here",
        ):
            with self.subTest(suffix=suffix):
                response = await self._run_natural_fallback_copy(
                    current_context="customers are around at the bar",
                    generated=(
                        "you want a fresh one?\n"
                        "customers are everywhere at the bar so no fresh one"
                        f"{suffix}"
                    ),
                    fallback_kind="live_blocked",
                    live_capture_blocker="customers are around at the bar",
                    live_capture_blocker_kind="work_crowd",
                    current_locations=("bar", "stockroom"),
                )
                visible = "\n".join(response.messages).lower()
                self.assertNotIn("tyler", visible)
                self.assertNotIn("gym", visible)
                self.assertNotIn("sister", visible)
                self.assertNotIn("taylor", visible)
                self.assertNotIn("sarah", visible)

    async def test_live_blocked_rejects_location_first_current_scene(self):
        response = await self._run_natural_fallback_copy(
            current_context="customers are around at the bar",
            generated=(
                "you want a fresh one?\n"
                "customers are everywhere at the bar so no fresh one, "
                "but at the gym right now"
            ),
            fallback_kind="live_blocked",
            live_capture_blocker="customers are around at the bar",
            live_capture_blocker_kind="work_crowd",
            current_locations=("bar", "stockroom"),
        )

        self.assertNotIn("gym", "\n".join(response.messages).lower())

    async def test_type_swap_replaces_reversed_photo_to_video_explanation(self):
        response = await self._run_natural_fallback_copy(
            current_context="she has the requested detail as a video rather than a photo",
            generated=(
                "you really want to see that?\n"
                "i don't have that as a video, but the photo is even better"
            ),
            media_type="video",
            description="a private video from her bathroom",
            fallback_kind="type_swap",
            requested_media_type="photo",
        )

        self.assertEqual(len(response.messages), 2)
        second = response.messages[1].lower()
        self.assertRegex(
            second,
            r"(?:don't|do not)\s+have.{0,40}(?:photo|pic|picture).{0,80}(?:video|clip)",
        )
        self.assertNotRegex(
            second,
            r"(?:don't|do not)\s+have.{0,40}(?:video|clip).{0,80}(?:photo|pic|picture)",
        )

    async def test_type_swap_preserves_correct_photo_to_video_explanation(self):
        natural = "i don't have that as a photo, but the video is even better"
        response = await self._run_natural_fallback_copy(
            current_context="she has the requested detail as a video rather than a photo",
            generated=f"you really want to see that?\n{natural}",
            media_type="video",
            description="a private video from her bathroom",
            fallback_kind="type_swap",
            requested_media_type="photo",
        )

        self.assertEqual(len(response.messages), 2)
        self.assertEqual(response.messages[1], natural)

    async def test_type_swap_rejects_no_problem_as_false_negation(self):
        for unsafe in (
            "i have no problem giving you a photo, but the video is better",
            "it's not true that i don't have a photo, but the video is better",
            "not gonna lie the photo is hot, but the video is better",
            "i never said i don't have a photo, but the video is better",
            "i can't say i don't have a photo, but the video is better",
            "i can't believe the photo looks this hot but the video is better",
            "the photo isn't bad but the video is better",
            "i don't think the photo is bad but the video is better",
        ):
            with self.subTest(unsafe=unsafe):
                response = await self._run_natural_fallback_copy(
                    current_context="she has the requested detail as a video rather than a photo",
                    generated=f"you really want to see that?\n{unsafe}",
                    media_type="video",
                    description="a private video from her bathroom",
                    fallback_kind="type_swap",
                    requested_media_type="photo",
                )

                second = response.messages[1].lower()
                self.assertNotEqual(second, unsafe)
                self.assertRegex(
                    second,
                    r"(?:don't|do not)\s+have.{0,40}(?:photo|pic|picture)",
                )

    async def test_semantic_fallback_replaces_false_affirmative_detail_claim(self):
        response = await self._run_natural_fallback_copy(
            current_context="this is not quite the ass shot he asked for",
            generated=(
                "straight to my ass? you're getting brave\n"
                "this ass pic isn't quite the angle you asked for, but open it"
            ),
            fallback_kind="semantic_near_match",
            requested_detail="ass",
        )

        self.assertEqual(len(response.messages), 2)
        second = response.messages[1].lower()
        self.assertNotIn("this ass pic", second)
        self.assertRegex(second, r"not\s+quite.{0,30}ass\s+shot")

    async def test_semantic_fallback_preserves_negative_missing_detail_copy(self):
        natural = "not quite the ass shot you asked for... but open it"
        response = await self._run_natural_fallback_copy(
            current_context="this is not quite the ass shot he asked for",
            generated=f"straight to my ass? you're getting brave\n{natural}",
            fallback_kind="semantic_near_match",
            requested_detail="ass",
        )

        self.assertEqual(len(response.messages), 2)
        self.assertEqual(response.messages[1], natural)

    async def test_semantic_fallback_rejects_detail_in_a_later_positive_clause(self):
        for generated_second in (
            "not quite the angle, but this one's all ass",
            "not quite what you asked for but you can stare at my ass in this",
            "not quite the angle and this one's all ass",
            "not quite the angle — this one's all ass",
            "not quite the angle: this one's all ass",
            "not quite the angle because this one's all ass",
            "no doubt this is the ass shot, but it's not quite what you asked for",
            "not quite what you asked for, but this booty pic is dangerous",
            "not only is this an ass pic, it's not quite what you asked for",
            "not gonna lie this ass pic is better instead",
            "i can't deny this ass pic is hot instead",
        ):
            with self.subTest(generated_second=generated_second):
                response = await self._run_natural_fallback_copy(
                    current_context="this is not quite the ass shot he asked for",
                    generated=f"you really went there?\n{generated_second}",
                    fallback_kind="semantic_near_match",
                    requested_detail="ass",
                )

                second = response.messages[1].lower()
                self.assertNotIn("all ass", second)
                self.assertNotIn("stare at my ass", second)
                self.assertNotIn("booty pic", second)
                self.assertNotIn("no doubt", second)
                self.assertRegex(second, r"not\s+quite.{0,30}ass\s+shot")

    def test_saved_offer_copy_rejects_current_capture_excuses(self):
        for text in (
            "Tyler just walked in but open this",
            "there are customers all around me but open this",
            "i'm at the gym so i can't take a fresh one, but open this",
            "i'm behind the bar rn, open this",
            "at the bar rn, open this",
            "at the gym right now, open this",
            "the bar is packed right now, open this",
            "my sister dared me to send this",
            "Taylor dared me to send this",
            "Sarah's sitting next to me, so open this quietly",
            "i'm stuck in the office, but open this",
            "i'm in Paris right now, open this",
            "i'm backstage right now, open this",
            "i'm at Nikkk's place rn, open this",
            "i just snapped this five seconds ago... open it",
            "i took this for you just now",
            "i shot this especially for you a few seconds ago",
            "i recorded this just for you right now",
            "i made this for you rn",
            "i captured this for you just now... open it",
            "i took this a minute ago",
            "i snapped this a second ago",
            "this one's fresh, open it",
            "i snapped this moments ago",
            "this is brand new, open it",
            "just finished filming this for you",
            "at the bar now, open this",
            "at the gym today, open this",
            "at the club tonight, open this",
            "sitting at the bar, open this",
            "hiding in the stockroom, open this",
            "waiting at the gym, open this",
            "chilling at the club, open this",
            "getting ready in my bathroom but open this",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.engine._saved_offer_copy_is_safe(text))

        self.assertTrue(
            self.engine._saved_offer_copy_is_safe(
                "god... open this before i change my mind"
            )
        )
        for text in (
            "here's one from my bed",
            "here's something i took in my bathroom",
        ):
            with self.subTest(text=text):
                self.assertTrue(self.engine._saved_offer_copy_is_safe(text))

    def test_fallback_teaser_rejects_unknown_people_and_places(self):
        for text in (
            "i'm in the office with my sister... you really want one?",
            "my sister is here and you're asking me for that?",
            "i'm stuck in the office... bold request",
            "Taylor is beside me... bold request",
            "Sarah is waiting here... bold request",
            "Nikkk is right here... bold request",
            "i'm backstage right now... bold request",
            "sitting at the bar... bold request",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.engine._fallback_teaser_is_safe(text))

    async def test_fallback_copy_rejects_fresh_capture_claims(self):
        cases = (
            (
                "semantic_near_match",
                "not quite the ass shot but i just took this instead",
                {"requested_detail": "ass"},
            ),
            (
                "semantic_near_match",
                "not quite the ass shot but i took this a minute ago instead",
                {"requested_detail": "ass"},
            ),
            (
                "semantic_near_match",
                "not quite the ass shot but this one's fresh instead",
                {"requested_detail": "ass"},
            ),
            (
                "type_swap",
                "i don't have it as a photo but i just filmed this video instead",
                {
                    "media_type": "video",
                    "requested_media_type": "photo",
                },
            ),
            (
                "live_blocked",
                "customers are around at the bar so no fresh one but i just took this",
                {
                    "live_capture_blocker": "customers are around at the bar",
                    "live_capture_blocker_kind": "work_crowd",
                    "current_locations": ("bar",),
                },
            ),
        )
        for fallback_kind, unsafe, kwargs in cases:
            with self.subTest(fallback_kind=fallback_kind):
                response = await self._run_natural_fallback_copy(
                    current_context="this requested version is unavailable",
                    generated=f"you really went there?\n{unsafe}",
                    fallback_kind=fallback_kind,
                    **kwargs,
                )
                second = response.messages[1].lower()
                self.assertNotEqual(second, unsafe.lower())
                self.assertNotRegex(
                    second,
                    r"\b(?:just|a\s+(?:second|minute)\s+ago|brand[- ]new|"
                    r"this\s+one(?:'|\u2019)s\s+fresh)\b",
                )

    async def test_semantic_fallback_rejects_current_activity_excuse(self):
        response = await self._run_natural_fallback_copy(
            current_context="this is not quite the ass shot he asked for",
            generated=(
                "you really went there?\n"
                "not quite the ass shot but sitting at the bar so open this"
            ),
            fallback_kind="semantic_near_match",
            requested_detail="ass",
        )

        self.assertNotIn("bar", response.messages[1].lower())

    async def test_inventory_synonyms_are_replaced_in_fallback_copy(self):
        for generated_second in (
            "not quite the angle you asked for; this is the top result in my inventory",
            "not quite the angle you asked for; this was the first search result",
        ):
            with self.subTest(generated_second=generated_second):
                response = await self._run_natural_fallback_copy(
                    current_context="this is not quite the ass shot he asked for",
                    generated=f"you really went there?\n{generated_second}",
                    fallback_kind="semantic_near_match",
                    requested_detail="ass",
                )
                second = response.messages[1].lower()
                self.assertNotIn("inventory", second)
                self.assertNotIn("search result", second)
                self.assertNotIn("top result", second)

    async def test_legacy_type_swap_infers_video_when_selected_item_is_photo(self):
        response = await self._run_natural_fallback_copy(
            current_context="this photo is the available type alternative",
            generated=(
                "you really want to see that?\n"
                "i picked the selected alternative for you"
            ),
            media_type="photo",
            description="a private photo from her bedroom",
            fallback_kind="type_swap",
            requested_media_type="",
        )

        self.assertEqual(len(response.messages), 2)
        second = response.messages[1].lower()
        self.assertRegex(
            second,
            r"(?:don't|do not)\s+have.{0,40}(?:video|clip).{0,80}(?:photo|pic|picture)",
        )
        self.assertNotRegex(
            second,
            r"(?:don't|do not)\s+have.{0,40}(?:photo|pic|picture).{0,80}(?:photo|pic|picture)",
        )

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

    async def test_failed_generation_uses_trusted_copy_and_finalizes_card(self):
        with ExitStack() as stack:
            for turn_patch in self._patch_turn_dependencies(generated=""):
                stack.enter_context(turn_patch)
            response = await self.engine._process_sexting(
                23, "show me a sexy photo"
            )

        self.assertNotIn(("cancel", "41"), self.events)
        self.assertIn(("finalize", "41"), self.events)
        self.assertEqual(response.commerce_action, "offer_current")
        self.assertEqual(response.media_offer["offer_id"], 41)
        self.assertNotIn("full_key", response.media_offer)
        self.assertNotIn("access_url", response.media_offer)
        self.assertEqual(len(response.messages), 1)
        self.assertTrue(response.messages[0].strip())

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
