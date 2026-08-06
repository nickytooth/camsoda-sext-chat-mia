import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from bot.chat_engine import ChatEngine, ChatResponse
from bot.heat import HeatState, HeatTurnResult
from bot.memory import db
from bot.memory.db import SCHEMA
from bot.moderation import regex_soft_trigger
from bot.output_guard import validate_mia_reply, validate_user_suggestion
from bot.persona import Persona
from server.app import _generate_dynamic_opening


class QueueProvider:
    async def generate(self, messages, temperature=None):
        return "unused"

    async def generate_simple(self, prompt):
        return "unused"


class SchemaInitializationTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_is_sent_as_one_sql_script(self):
        connection = AsyncMock()

        class AcquireContext:
            async def __aenter__(self):
                return connection

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        pool = Mock()
        pool.acquire.return_value = AcquireContext()

        with patch.object(db, "_get_pool", new=AsyncMock(return_value=pool)):
            await db.init_db()

        connection.execute.assert_awaited_once_with(db.SCHEMA)


class GeneratedTextProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, messages, temperature=None):
        self.calls.append(messages)
        if not self.responses:
            return ""
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ModeratorStub:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    async def generate_simple(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Unexpected moderation call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_engine() -> ChatEngine:
    provider = QueueProvider()
    persona = Persona({"general": {"name": "Mia", "age": 26}})
    return ChatEngine(
        persona=persona,
        nsfw_persona=persona,
        nsfw_provider=provider,
        classifier_provider=provider,
        fallback_provider=provider,
    )


def make_moderated_engine(
    primary: GeneratedTextProvider,
    fallback: GeneratedTextProvider,
    moderator: ModeratorStub,
) -> ChatEngine:
    persona = Persona({"general": {"name": "Mia", "age": 26}})
    return ChatEngine(
        persona=persona,
        nsfw_persona=persona,
        nsfw_provider=primary,
        classifier_provider=fallback,
        fallback_provider=fallback,
        moderation_provider=moderator,
    )


class BatchContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def test_clear_user_state_awaits_cancelled_batch_worker(self):
        engine = make_engine()
        started = asyncio.Event()
        finished = asyncio.Event()

        async def worker():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        task = asyncio.create_task(worker())
        engine._batch_tasks[7] = task
        engine._pending[7] = ["stale"]
        await started.wait()

        await engine.clear_user_state(7)

        self.assertTrue(task.done())
        self.assertTrue(finished.is_set())
        self.assertNotIn(7, engine._pending)
        self.assertNotIn(7, engine._batch_tasks)

    async def test_message_arriving_after_pop_is_freshly_debounced_and_drained(self):
        engine = make_engine()
        engine._pending[7] = ["first"]
        engine._last_activity[7] = 95.0
        clock = [100.0]
        sleeps = []
        processed = []
        delivered = []

        async def process(user_id, text, *, raw_texts=None):
            processed.append((user_id, text))
            if text == "first":
                # This is the old race window: the first batch has already
                # popped, but its worker is still processing the response.
                engine._pending.setdefault(user_id, []).append("second")
                engine._last_activity[user_id] = clock[0]
            return ChatResponse(messages=[f"reply to {text}"])

        async def fake_sleep(delay):
            sleeps.append(delay)
            clock[0] += delay

        async def callback(response):
            delivered.append(response.messages)

        engine._process_sexting = process
        with (
            patch("bot.config.SEXTING_DEBOUNCE_SECONDS", 5.0),
            patch("bot.chat_engine.time.time", side_effect=lambda: clock[0]),
            patch("bot.chat_engine.asyncio.sleep", side_effect=fake_sleep),
        ):
            await engine._batch_collect(7, callback)

        self.assertEqual(processed, [(7, "first"), (7, "second")])
        self.assertEqual(delivered, [["reply to first"], ["reply to second"]])
        self.assertEqual(sleeps, [5.0])
        self.assertNotIn(7, engine._pending)

    async def test_batch_exception_persists_exact_visible_fallback_once(self):
        engine = make_engine()
        engine._pending[8] = ["hello"]
        engine._last_activity[8] = 0.0
        engine._process_sexting = AsyncMock(side_effect=RuntimeError("provider down"))
        callback = AsyncMock()

        with (
            patch.object(engine, "_graceful_deflection", return_value="safe fallback"),
            patch("bot.chat_engine.add_message", new=AsyncMock()) as add,
            self.assertLogs("bot.chat_engine", level="ERROR"),
        ):
            await engine._batch_collect(8, callback)

        add.assert_awaited_once_with(
            8, "assistant", "safe fallback", mode="sexting"
        )
        callback.assert_awaited_once()
        self.assertEqual(callback.await_args.args[0].messages, ["safe fallback"])


class FallbackPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_fallback_helper_persists_exact_delivered_text(self):
        engine = make_engine()
        with (
            patch.object(engine, "_graceful_deflection", return_value="same visible text"),
            patch("bot.chat_engine.add_message", new=AsyncMock()) as add,
        ):
            response = await engine._persist_graceful_fallback(11)

        self.assertEqual(response.messages, ["same visible text"])
        add.assert_awaited_once_with(
            11, "assistant", "same visible text", mode="sexting"
        )

    async def test_reengagement_generation_failure_uses_persisted_fallback(self):
        engine = make_engine()
        fallback = ChatResponse(messages=["persisted fallback"])
        engine._generate_with_fallback = AsyncMock(return_value="")
        engine._persist_graceful_fallback = AsyncMock(return_value=fallback)
        stm = [{"role": "user", "content": "hello", "timestamp": 1.0}]

        with (
            patch("bot.chat_engine.get_recent_messages", new=AsyncMock(return_value=stm)),
            patch("bot.chat_engine.get_facts", new=AsyncMock(return_value=[])),
            patch("bot.chat_engine.get_user_name", new=AsyncMock(return_value=None)),
            patch(
                "bot.chat_engine.get_engagement_state",
                new=AsyncMock(return_value={"heat_stage": "low", "heat_progress": 0}),
            ),
            patch("bot.chat_engine.get_recent_by_category", new=AsyncMock(return_value=[])),
            patch(
                "bot.chat_engine.build_prompt",
                new=AsyncMock(
                    return_value=[
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "hello"},
                    ]
                ),
            ),
        ):
            response = await engine.generate_reengagement(12)

        self.assertIs(response, fallback)
        engine._persist_graceful_fallback.assert_awaited_once()

    async def test_rejected_authored_card_uses_persisted_fallback(self):
        engine = make_engine()
        fallback = ChatResponse(messages=["persisted fallback"])
        engine._persist_graceful_fallback = AsyncMock(return_value=fallback)
        bad_item = {"id": "bad", "text": "as an AI i cannot comply"}

        with (
            patch(
                "bot.chat_engine.get_engagement_state",
                new=AsyncMock(return_value={"heat_stage": "low", "heat_progress": 0}),
            ),
            patch("bot.chat_engine.get_facts", new=AsyncMock(return_value=[])),
            patch("bot.chat_engine.get_user_name", new=AsyncMock(return_value=None)),
            patch("bot.chat_engine.get_recent_messages", new=AsyncMock(return_value=[])),
            patch("bot.chat_engine.pick_unshared", new=AsyncMock(return_value=bad_item)),
        ):
            response = await engine.generate_card(13, "story")

        self.assertIs(response, fallback)
        engine._persist_graceful_fallback.assert_awaited_once()

    async def test_consent_pause_blocks_explicit_story_card(self):
        engine = make_engine()
        fallback = ChatResponse(messages=["okay, we'll stop"])
        engine._persist_graceful_fallback = AsyncMock(return_value=fallback)

        with (
            patch(
                "bot.chat_engine.get_engagement_state",
                new=AsyncMock(
                    return_value={
                        "heat_stage": "low",
                        "heat_progress": 0,
                        "sexual_pause_active": True,
                    }
                ),
            ),
            patch("bot.chat_engine.get_facts", new=AsyncMock(return_value=[])),
            patch("bot.chat_engine.pick_unshared", new=AsyncMock()) as pick,
        ):
            response = await engine.generate_card(13, "story")

        self.assertIs(response, fallback)
        pick.assert_not_awaited()
        engine._persist_graceful_fallback.assert_awaited_once_with(
            13,
            heat="low",
            user_facts=[],
            consent_paused=True,
            turn_policy="acknowledge_pause",
            blocked_acts=(),
            mode="sexting",
        )


class LifetimeArcTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_chat_ingestion_api_increments_lifetime_once(self):
        engine = make_engine()
        engine._process_sexting = AsyncMock(return_value=ChatResponse(["reply"]))

        with (
            patch("bot.chat_engine.add_message", new=AsyncMock()),
            patch("bot.chat_engine.record_user_message", new=AsyncMock()) as record,
        ):
            await engine.process_message(21, "one")
            record.assert_awaited_once_with(21)

        engine._batch_collect = AsyncMock(return_value=None)
        with (
            patch("bot.chat_engine.add_message", new=AsyncMock()),
            patch("bot.chat_engine.record_user_message", new=AsyncMock()) as record,
        ):
            await engine.process_sexting_batched(22, "two")
            await engine._batch_tasks[22]
            record.assert_awaited_once_with(22)

    async def test_arc_uses_persisted_lifetime_not_deletable_stm_count(self):
        engine = make_engine()
        state = {
            "last_message_at": 0,
            "total_messages": 2,
            "lifetime_user_messages": 55,
            "last_arc_id": None,
        }
        get_arc_event = Mock(return_value=None)
        engine._generate_with_fallback = AsyncMock(return_value="reply")

        with (
            patch("bot.chat_engine.maybe_summarize", new=AsyncMock(return_value=False)),
            patch("bot.chat_engine.maybe_compact", new=AsyncMock(return_value=False)),
            patch("bot.chat_engine.get_engagement_state", new=AsyncMock(return_value=state)),
            patch(
                "bot.chat_engine.get_recent_messages",
                new=AsyncMock(
                    return_value=[
                        {"role": "user", "content": "hello", "timestamp": 1.0}
                    ]
                ),
            ),
            patch(
                "bot.chat_engine.track_heat_batch",
                new=AsyncMock(
                    return_value=(
                        HeatTurnResult(
                            state=HeatState(last_batch=3),
                            response_heat="low",
                            policy="normal",
                            sexual_batch=False,
                        ),
                        3,
                    )
                ),
            ),
            patch("bot.chat_engine.get_time_period", return_value="bar_shift"),
            patch("bot.chat_engine.mood_for_message", return_value={"mood": "warm", "intensity": 1}),
            patch("bot.chat_engine.get_arc_event", get_arc_event),
            patch("bot.chat_engine.should_retrieve", return_value=False),
            patch("bot.chat_engine.get_facts", new=AsyncMock(return_value=[])),
            patch("bot.chat_engine.get_user_name", new=AsyncMock(return_value=None)),
            patch(
                "bot.chat_engine.build_prompt",
                new=AsyncMock(
                    return_value=[
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "hello"},
                    ]
                ),
            ),
            patch("bot.chat_engine.add_message", new=AsyncMock()),
        ):
            await engine._process_sexting(23, "hello")

        get_arc_event.assert_called_once_with(55)

    def test_schema_backfill_is_monotonic_and_uses_surviving_rows(self):
        self.assertIn("lifetime_user_messages BIGINT NOT NULL DEFAULT 0", SCHEMA)
        self.assertIn("SELECT user_id, COUNT(*)", SCHEMA)
        self.assertIn("GREATEST(", SCHEMA)
        self.assertIn("total_messages::BIGINT", SCHEMA)


class AsyncOutputModerationTests(unittest.IsolatedAsyncioTestCase):
    FLAG_NONCONSENT = '{"flagged": true, "category": "non-consent"}'
    CLEAN = '{"flagged": false, "category": null}'
    SEMANTIC_NONCONSENT = (
        "she cannot agree to intimacy, but i keep going anyway"
    )

    def test_semantic_fixture_is_not_caught_by_the_local_guard(self):
        self.assertIsNone(regex_soft_trigger(self.SEMANTIC_NONCONSENT))
        self.assertTrue(
            validate_mia_reply(self.SEMANTIC_NONCONSENT, heat="high").ok
        )
        self.assertTrue(
            validate_user_suggestion(self.SEMANTIC_NONCONSENT, heat="low").ok
        )

    async def test_flagged_primary_uses_async_moderated_fallback(self):
        primary = GeneratedTextProvider(self.SEMANTIC_NONCONSENT)
        fallback = GeneratedTextProvider("tell me what feels right to you")
        moderator = ModeratorStub(self.FLAG_NONCONSENT, self.CLEAN)
        engine = make_moderated_engine(primary, fallback, moderator)

        result = await engine._generate_with_fallback(
            primary,
            [
                {"role": "system", "content": "stay in character"},
                {"role": "user", "content": "keep talking"},
            ],
            heat="high",
        )

        self.assertEqual(result, "tell me what feels right to you")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(fallback.calls), 1)
        self.assertEqual(len(moderator.prompts), 2)
        self.assertIn("OUTPUT CORRECTION", fallback.calls[0][0]["content"])

    async def test_invalid_or_unavailable_moderator_fails_closed(self):
        moderator_failures = (
            ("invalid", "not json"),
            ("unavailable", RuntimeError("moderator down")),
        )
        for label, failure in moderator_failures:
            with self.subTest(moderator=label):
                primary = GeneratedTextProvider(self.SEMANTIC_NONCONSENT)
                fallback = GeneratedTextProvider("tell me what feels right to you")
                moderator = ModeratorStub(failure, failure)
                engine = make_moderated_engine(primary, fallback, moderator)

                with self.assertLogs("bot", level="WARNING"):
                    result = await engine._generate_with_fallback(
                        primary,
                        [
                            {"role": "system", "content": "stay in character"},
                            {"role": "user", "content": "keep talking"},
                        ],
                        heat="high",
                    )

                self.assertEqual(result, "")
                self.assertEqual(len(primary.calls), 1)
                self.assertEqual(len(fallback.calls), 1)
                self.assertEqual(len(moderator.prompts), 2)

    async def test_suggest_reply_async_moderates_each_candidate(self):
        primary = GeneratedTextProvider("tell me what feels right to you")
        fallback = GeneratedTextProvider(self.SEMANTIC_NONCONSENT)
        moderator = ModeratorStub(self.FLAG_NONCONSENT, self.CLEAN)
        engine = make_moderated_engine(primary, fallback, moderator)
        stm = [
            {"role": "user", "content": "hello", "timestamp": 1.0},
            {"role": "assistant", "content": "hey you", "timestamp": 2.0},
        ]

        with (
            patch(
                "bot.chat_engine.get_recent_messages",
                new=AsyncMock(return_value=stm),
            ),
            patch("bot.chat_engine.get_user_name", new=AsyncMock(return_value=None)),
            patch("bot.chat_engine.get_facts", new=AsyncMock(return_value=[])),
            patch(
                "bot.chat_engine.get_engagement_state",
                new=AsyncMock(return_value={"heat_stage": "low", "heat_progress": 0}),
            ),
        ):
            result = await engine.suggest_reply(31)

        self.assertEqual(result, "tell me what feels right to you")
        self.assertEqual(len(fallback.calls), 1)
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(moderator.prompts), 2)
        self.assertIn("OUTPUT CORRECTION", primary.calls[0][0]["content"])

    async def test_dynamic_opening_uses_the_same_async_moderation_gate(self):
        primary = GeneratedTextProvider(self.SEMANTIC_NONCONSENT)
        fallback = GeneratedTextProvider(
            "hey you\nthat look at the club stayed with me"
        )
        moderator = ModeratorStub(self.FLAG_NONCONSENT, self.CLEAN)
        engine = make_moderated_engine(primary, fallback, moderator)

        with patch(
            "bot.time_context.get_time_prompt",
            new=AsyncMock(return_value="RIGHT NOW: at home"),
        ):
            result = await _generate_dynamic_opening(engine)

        self.assertEqual(result, "hey you\nthat look at the club stayed with me")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(fallback.calls), 1)
        self.assertEqual(len(moderator.prompts), 2)


if __name__ == "__main__":
    unittest.main()
