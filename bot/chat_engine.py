"""
Mode-aware chat engine — replaces Telegram handlers.
Processes messages for Sexting mode.
"""

import asyncio
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field

from bot.persona import Persona, load_persona
from bot.memory.stm import add_message, get_recent_messages, count_turns
from bot.memory.ltm import retrieve_relevant, should_retrieve
from bot.memory.summarizer import maybe_summarize, maybe_compact
from bot.memory.facts import get_facts, format_facts_for_prompt, get_user_name
from bot.prompt_builder import build_prompt
from bot.router import classify_fast
from bot.engagement import track_message, get_engagement_state, record_reengage
from bot.mood import mood_for_message, is_ai_question
from bot.content_library import pick_unshared, mark_shared, get_examples, library_size
from bot.time_context import get_time_period, get_preferred_tags, get_scene
from bot.providers.base import LLMProvider
from bot.config import STM_MAX_TURNS, UPLOADS_DIR, LLM_TIMEOUT_SECONDS
from bot.memory.db import get_connection

logger = logging.getLogger(__name__)

def _image_ext(image_bytes: bytes) -> str:
    """Sniff a sensible file extension from the image's magic bytes."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _save_upload(image_bytes: bytes) -> str | None:
    """Persist an uploaded image to disk and return its served URL path
    (e.g. '/uploads/ab12.jpg'), or None if saving fails."""
    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}{_image_ext(image_bytes)}"
        (UPLOADS_DIR / name).write_bytes(image_bytes)
        return f"/uploads/{name}"
    except Exception as e:
        logger.error("Failed to save uploaded image: %s", e)
        return None


# Dynamic fantasies are generated fresh each time, so there is no library id to
# track. To avoid circling the same idea we remember a short "theme" (the first
# sentence) of the last few we sent and tell the model to avoid them.
DYN_FANTASY_CATEGORY = "dyn_fantasy_theme"
MAX_RECENT_FANTASY_THEMES = 8


def _fantasy_theme(text: str) -> str:
    """First sentence of a generated fantasy, truncated — used to avoid repeats."""
    stripped = (text or "").strip()
    if not stripped:
        return ""
    first = re.split(r"(?<=[.!?\u2026])\s+", stripped)[0]
    return first[:120]


async def _recent_fantasy_themes(user_id: int) -> list[str]:
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "SELECT content_id FROM sent_content WHERE user_id = ? AND category = ? "
            "ORDER BY sent_at DESC LIMIT ?",
            (user_id, DYN_FANTASY_CATEGORY, MAX_RECENT_FANTASY_THEMES),
        )
        rows = await cursor.fetchall()
        return [row["content_id"] for row in rows if row["content_id"]]
    finally:
        await conn.close()


async def _record_fantasy_theme(user_id: int, theme: str) -> None:
    if not theme:
        return
    conn = await get_connection()
    try:
        await conn.execute(
            "INSERT INTO sent_content (user_id, content_id, category, sent_at, paid) "
            "VALUES (?, ?, ?, ?, 0)",
            (user_id, theme, DYN_FANTASY_CATEGORY, time.time()),
        )
        await conn.commit()
    finally:
        await conn.close()


@dataclass
class ChatResponse:
    """Response from the chat engine."""
    messages: list[str] = field(default_factory=list)


def _format_last_seen(gap_seconds: float) -> str | None:
    """Human-friendly note about how long since the user last messaged."""
    if gap_seconds < 1800:  # under 30 min — same conversation, say nothing
        return None
    if gap_seconds < 7200:
        when = "about an hour"
    elif gap_seconds < 21600:
        when = "a few hours"
    elif gap_seconds < 86400:
        when = "most of the day"
    elif gap_seconds < 172800:
        when = "since yesterday"
    else:
        days = int(gap_seconds // 86400)
        when = f"about {days} days"
    return (
        f"It's been {when} since you two last talked. "
        "React to the gap naturally if it feels right — a little missed-him, "
        "a little curious where he's been — but don't make it heavy."
    )


class ChatEngine:
    """Chat engine for Sexting mode."""

    def __init__(
        self,
        persona: Persona,
        nsfw_persona: Persona | None,
        nsfw_provider: LLMProvider,
        classifier_provider: LLMProvider,
        vision_provider=None,
        fallback_provider: LLMProvider | None = None,
    ):
        self.persona = persona
        self.nsfw_persona = nsfw_persona
        self.nsfw_provider = nsfw_provider
        self.classifier_provider = classifier_provider
        self.vision_provider = vision_provider
        # Sexting generator fallback when Grok fails (Gemini 2.5 Flash by
        # default). Falls back to the classifier provider if not supplied.
        self.fallback_provider = fallback_provider or classifier_provider

        # Sexting mode batching (debounce: reply N seconds after the LAST msg)
        self._pending: dict[int, list[str]] = {}
        self._batch_tasks: dict[int, asyncio.Task] = {}
        self._last_activity: dict[int, float] = {}
        self._processing_lock: dict[int, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_message(
        self,
        user_id: int,
        text: str,
        mode: str = "sexting",
        image_bytes: bytes | None = None,
    ) -> ChatResponse:
        """Process a user message (sexting mode)."""
        # If user sent a photo, persist it (so it survives history reloads) and
        # analyze it so she can react to its content.
        image_url = _save_upload(image_bytes) if image_bytes else None
        if image_bytes and self.vision_provider:
            try:
                description = await self.vision_provider.analyze_image(image_bytes)
                text = f"{text}\n[User sent a photo: {description}]" if text else f"[User sent a photo: {description}]"
            except Exception as e:
                logger.error("Vision analysis failed: %s", e)

        await add_message(user_id, "user", text, mode="sexting", image_url=image_url)
        return await self._process_sexting(user_id, text)

    async def process_sexting_batched(
        self,
        user_id: int,
        text: str,
        image_bytes: bytes | None = None,
        on_response=None,
    ) -> None:
        """
        Add message to batch buffer. After the collect window, all
        accumulated messages are processed together.
        on_response: async callback(ChatResponse) called when batch is ready.
        """
        image_url = _save_upload(image_bytes) if image_bytes else None
        if image_bytes and self.vision_provider:
            try:
                description = await self.vision_provider.analyze_image(image_bytes)
                text = f"{text}\n[User sent a photo: {description}]" if text else f"[User sent a photo: {description}]"
            except Exception as e:
                logger.error("Vision analysis failed: %s", e)

        # Persist immediately so history survives mode switches / disconnects,
        # even before the batch window flushes.
        await add_message(user_id, "user", text, mode="sexting", image_url=image_url)

        if user_id not in self._pending:
            self._pending[user_id] = []
        self._pending[user_id].append(text)

        # Reset the debounce countdown on every message.
        self._last_activity[user_id] = time.time()

        # Start batch task if none running
        if user_id not in self._batch_tasks or self._batch_tasks[user_id].done():
            self._batch_tasks[user_id] = asyncio.create_task(
                self._batch_collect(user_id, on_response)
            )

    # Soft, in-character lines used ONLY as a last resort, when both the primary
    # (Grok) and the fallback (Gemini) return nothing/refuse. They keep the
    # conversation alive instead of leaving the user staring at silence.
    _GRACEFUL_DEFLECTIONS = [
        "fuck my brain just short-circuited... say that again?",
        "wait i got distracted thinking about your cock... what were you saying?",
        "hold on, i'm still recovering from that... tell me again?",
        "god you make me stupid sometimes... come back to that for me?",
        "say that again babe, i was too busy imagining what you'd do to me",
    ]

    @staticmethod
    def _graceful_deflection() -> str:
        return random.choice(ChatEngine._GRACEFUL_DEFLECTIONS)

    async def _generate_with_fallback(self, provider: LLMProvider, prompt_messages: list[dict]) -> str:
        """Generate a sexting reply, hardened against hangs and silent refusals.

        Grok is the primary generator. We fall back to the dedicated Gemini 2.5
        Flash provider (safety filters OFF) not only when Grok raises, but ALSO
        when it returns an empty/blank string — a hard content-policy refusal
        often comes back as empty content rather than an exception, which the old
        code mistook for "nothing to say" and went silent. Each call is bounded
        by LLM_TIMEOUT_SECONDS so a hung request can't freeze the chat. Returns
        "" only if BOTH fail; the caller then substitutes a graceful line.
        """
        text = ""
        try:
            text = await asyncio.wait_for(
                provider.generate(prompt_messages), timeout=LLM_TIMEOUT_SECONDS
            )
        except Exception as e:
            logger.warning("Primary (Grok) generation failed/timed out: %s", e)

        if text and text.strip():
            return text

        logger.warning("Primary returned empty/refused — falling back to Gemini")
        try:
            text = await asyncio.wait_for(
                self.fallback_provider.generate(prompt_messages), timeout=LLM_TIMEOUT_SECONDS
            )
        except Exception as e2:
            logger.error("Fallback (Gemini) generation also failed: %s", e2)
            return ""
        return text or ""

    def clear_user_state(self, user_id: int) -> None:
        """Drop ALL in-memory per-user state. Called on reset so a stuck/old
        batch task (or a held processing lock) can never block the fresh
        conversation — the bug where, after reset, the user got no reply."""
        task = self._batch_tasks.pop(user_id, None)
        if task and not task.done():
            task.cancel()
        self._pending.pop(user_id, None)
        self._last_activity.pop(user_id, None)
        self._processing_lock.pop(user_id, None)

    async def suggest_reply(self, user_id: int, mode: str = "sexting") -> str:
        """
        AI Help: write a suggested NEXT message for the USER to send — a reply
        TO Mia, in his voice. Used by the 'generate reply' button. The
        suggestion is not stored; the user approves/edits it before sending.
        """
        stm = await get_recent_messages(user_id, STM_MAX_TURNS, mode=mode)
        if not stm:
            return ""

        user_name = await get_user_name(user_id)
        him = user_name or "the man"

        transcript_lines = []
        for m in stm:
            if m["role"] == "user":
                transcript_lines.append(f"{him}: {m['content']}")
            elif m["role"] == "assistant":
                transcript_lines.append(f"Mia: {m['content']}")
        transcript = "\n".join(transcript_lines[-20:])

        system = (
            f"You are a flirting wingman helping a young man ({him}) who is sexting "
            "with Mia — his best friend's girlfriend. She's a shameless, crude, 26-year-old "
            "party girl who loves cheating and has zero filter. It is a consensual adult fantasy roleplay.\n\n"
            "Write the SINGLE next message HE should send to her. Rules:\n"
            "- First person, written TO Mia, in his voice\n"
            "- Short and natural, like a real text — one or two lines, no period at the end\n"
            "- Confident, direct, a little dirty; match her energy and escalate the heat\n"
            "- React to what she JUST said — don't ignore it\n"
            "- No quotation marks, no name labels, no emojis — output only the message text\n\n"
            "Conversation so far:\n" + transcript
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Write his next message now. Output only the message text."},
        ]

        # The suggestion is a single short line, so latency matters more than
        # heavyweight prose. Generate it on the fast Gemini 2.5 Flash model
        # (the fallback provider) rather than the heavy Grok chat model, and
        # fall back to Grok only on error.
        try:
            text = await self.fallback_provider.generate(messages)
        except Exception as e:
            logger.warning("Suggest-reply fast generation failed: %s — retrying on Grok", e)
            try:
                text = await self.nsfw_provider.generate(messages)
            except Exception as e2:
                logger.warning("Suggest-reply generation failed: %s", e2)
                return ""
        return (text or "").strip().strip('"').strip()

    async def generate_reengagement(self, user_id: int) -> ChatResponse:
        """
        Generate a spontaneous follow-up ("double-text") for a user who is
        still online but has gone quiet. Sexting mode only. Returns an empty
        ChatResponse if there's nothing to react to.
        """
        mode = "sexting"
        stm = await get_recent_messages(user_id, STM_MAX_TURNS, mode=mode)
        if not stm or not any(m["role"] == "user" for m in stm):
            return ChatResponse()

        mood = {"mood": "warm", "intensity": 1}
        active_persona = self.nsfw_persona or self.persona
        nudge_provider = self.nsfw_provider

        user_facts = await get_facts(user_id)
        facts_text = format_facts_for_prompt(user_facts)
        user_name = await get_user_name(user_id)

        hint = (
            "He's gone quiet for a few minutes. Double-text him ONCE, the way a real "
            "shameless girl does when someone's still on her mind — pick up a SPECIFIC thread "
            "from what you two were just saying, or send him a filthy little thought or a nude "
            "description that just crossed your mind. Make it feel spontaneous and desperate, "
            "never needy. ONE short line only. Do NOT ask generic filler ('what are you "
            "thinking', 'are you there', 'still there?'), do NOT recap, and do NOT complain "
            "about waiting or being ignored."
        )

        prompt_messages = await build_prompt(
            active_persona, [], stm,
            mode=mode,
            push_hint=hint,
            user_name=user_name,
            facts_text=facts_text,
            mood=mood,
            already_greeted=True,
        )

        # Reassemble so the message list starts AND ends with a user turn
        # (ending on a user turn makes the model produce the follow-up).
        system_msg = prompt_messages[0]
        turns = [m for m in prompt_messages[1:] if m["role"] in ("user", "assistant")]
        while turns and turns[0]["role"] == "assistant":
            turns.pop(0)
        turns.append({"role": "user", "content": "[He's been quiet for a bit.]"})
        final_messages = [system_msg] + turns

        try:
            response_text = await nudge_provider.generate(final_messages)
        except Exception as e:
            logger.warning("Re-engagement generation failed: %s", e)
            return ChatResponse()

        if not response_text or not response_text.strip():
            return ChatResponse()

        await add_message(user_id, "assistant", response_text, mode=mode)
        parts = self._split_response(response_text)

        return ChatResponse(messages=parts)

    async def _deliver_dynamic_fantasy(self, user_id: int) -> ChatResponse:
        """Generate a fresh fantasy rooted in Mia's current location and
        tailored to the user from his facts + LTM + recent chat.

        The authored fantasies in library/fantasies.yaml are used only as STYLE
        exemplars (length/rhythm/tone) — never sent verbatim. A short randomised
        lead-in bubble opens it and the body is paced into exactly three bubbles,
        matching the existing card feel. Recently-sent themes are passed back to
        the model so it doesn't circle the same idea.
        """
        mode = "sexting"
        active_persona = self.nsfw_persona or self.persona

        user_facts = await get_facts(user_id)
        facts_text = format_facts_for_prompt(user_facts)
        user_name = await get_user_name(user_id)
        kink_bits: list[str] = []
        for f in (user_facts or []):
            if f["key"] in ("kinks", "turn_ons", "fetishes", "interests"):
                kink_bits.append(f["value"])

        stm = await get_recent_messages(user_id, STM_MAX_TURNS, mode=mode)

        # LTM — query from what we know he likes, else his most recent message.
        query = ", ".join(b for b in kink_bits if b).strip()
        if not query:
            for m in reversed(stm):
                if m.get("role") == "user" and m.get("content"):
                    query = m["content"]
                    break
        ltm = await retrieve_relevant(user_id, query, mode=mode) if query else []

        scene = get_scene()
        examples = get_examples("fantasy", 2)
        recent_themes = await _recent_fantasy_themes(user_id)

        example_block = "\n".join(f"<style_example>\n{ex}\n</style_example>" for ex in examples)
        avoid_block = ""
        if recent_themes:
            bullets = "\n".join(f"- {t}" for t in recent_themes)
            avoid_block = (
                "\nYou've recently shared these with him — choose a DIFFERENT angle, "
                f"don't repeat them:\n{bullets}\n"
            )

        hint = (
            "He just tapped 'Hear a fantasy'. Invent ONE brand-new fantasy and tell it to him now.\n"
            f"SETTING: it happens right where you are at this moment — {scene['where']}. "
            "Make the place vivid and specific; the fantasy is rooted HERE.\n"
            "MAKE IT HIS: weave in what you know he likes from the facts and memories above "
            "(his kinks, his turn-ons, things he's told you). It should feel written for him, "
            "not generic.\n"
            "STYLE: match the examples below EXACTLY in length, rhythm and tone — do NOT reuse "
            "their content or setting, only their style:\n"
            f"{example_block}\n"
            "FORMAT: about three short, filthy bubbles, explicit and raw, in your texting "
            "voice, no trailing period on the last line."
            f"{avoid_block}"
        )

        prompt_messages = await build_prompt(
            active_persona, ltm, stm,
            mode=mode,
            push_hint=hint,
            user_name=user_name,
            facts_text=facts_text,
            mood={"mood": "aroused", "intensity": 3},
            already_greeted=True,
        )

        system_msg = prompt_messages[0]
        turns = [m for m in prompt_messages[1:] if m["role"] in ("user", "assistant")]
        while turns and turns[0]["role"] == "assistant":
            turns.pop(0)
        turns.append({"role": "user", "content": "[Tell me a fantasy right now.]"})
        final_messages = [system_msg] + turns

        try:
            response_text = await self.nsfw_provider.generate(final_messages)
        except Exception as e:
            logger.warning("Dynamic fantasy (Grok) failed: %s — falling back to Gemini", e)
            try:
                response_text = await self.fallback_provider.generate(final_messages)
            except Exception as e2:
                logger.error("Dynamic fantasy fallback also failed: %s", e2)
                return ChatResponse()

        if not response_text or not response_text.strip():
            return ChatResponse()

        paras = [self._card_lead_in("fantasy")] + self._repack_to_n(response_text, 3)
        await add_message(user_id, "assistant", "\n".join(paras), mode=mode)
        await _record_fantasy_theme(user_id, _fantasy_theme(response_text))
        logger.info(
            "Fantasy generated (dynamic) for user %d — location=%s",
            user_id, scene.get("preferred_tags"),
        )
        return ChatResponse(messages=paras)

    async def generate_card(self, user_id: int, kind: str) -> ChatResponse:
        """Card-triggered fantasy/story.

        - 'fantasy': ALWAYS generated fresh, rooted in Mia's current location
          and tailored to the user from his facts + LTM + recent chat (the authored
          library serves only as a style exemplar). See `_deliver_dynamic_fantasy`.
        - 'story': pulled verbatim from the authored library (so she never repeats),
          improvising only as a safety net when the library is missing/empty.
        """
        if kind == "fantasy":
            return await self._deliver_dynamic_fantasy(user_id)

        mode = "sexting"
        # Stories come from the authored library, delivered VERBATIM and NEVER
        # repeated: pick_unshared returns the next untold one, or None once she's
        # told them all. We deliberately do NOT reset the rotation — see the
        # exhausted branch below. Location-matching tags are preferred when present.
        item = await pick_unshared(user_id, kind, preferred_tags=get_preferred_tags())

        # Library item found: deliver the authored text VERBATIM — one paragraph
        # per bubble, exactly as written. A short randomised lead-in opens it, and
        # a reciprocity nudge always closes it (she wants to hear HIS stories too).
        if item:
            paras = [
                " ".join(line.strip() for line in p.split("\n") if line.strip())
                for p in item["text"].strip().split("\n\n")
                if p.strip()
            ]
            if not paras:
                paras = self._repack_to_n(item["text"], 3)
            paras = [self._card_lead_in(kind)] + paras + [self._story_reciprocity_nudge()]
            await add_message(user_id, "assistant", "\n".join(paras), mode=mode)
            await mark_shared(user_id, kind, item["id"])
            logger.info("Card story '%s' delivered verbatim to user %d", item["id"], user_id)
            return ChatResponse(messages=paras)

        # No untold story left. If the library actually HAS stories, she's simply
        # told him all of them — she says so and turns it back on him, asking for
        # his. She keeps doing this on every later tap (we never recycle old ones).
        if library_size(kind) > 0:
            msg = self._story_exhausted_message()
            await add_message(user_id, "assistant", "\n".join(msg), mode=mode)
            logger.info("Card story exhausted for user %d — inviting his stories", user_id)
            return ChatResponse(messages=msg)

        # SAFETY NET ONLY: reached solely when the authored library file is missing
        # or empty. In that edge case we improvise one rather than send nothing.
        logger.warning("Card story library empty/missing — improvising for user %d", user_id)
        hint = (
            "He asked for a story and you've already told him your best ones. Either "
            "invent a fresh wild memory from your past, or call back to one "
            "you've already told him about ('remember when i told you about...'). "
            "About three filthy bubbles, real heat and detail, no trailing period."
        )

        stm = await get_recent_messages(user_id, STM_MAX_TURNS, mode=mode)
        active_persona = self.nsfw_persona or self.persona
        user_facts = await get_facts(user_id)
        facts_text = format_facts_for_prompt(user_facts)
        user_name = await get_user_name(user_id)

        prompt_messages = await build_prompt(
            active_persona, [], stm,
            mode=mode,
            push_hint=hint,
            user_name=user_name,
            facts_text=facts_text,
            mood={"mood": "aroused", "intensity": 3},
            already_greeted=True,
        )

        system_msg = prompt_messages[0]
        turns = [m for m in prompt_messages[1:] if m["role"] in ("user", "assistant")]
        while turns and turns[0]["role"] == "assistant":
            turns.pop(0)
        turns.append({"role": "user", "content": "[Tell it to me now.]"})
        final_messages = [system_msg] + turns

        try:
            response_text = await self.nsfw_provider.generate(final_messages)
        except Exception as e:
            logger.warning("Card (Grok) failed: %s — falling back to Gemini", e)
            try:
                response_text = await self.fallback_provider.generate(final_messages)
            except Exception as e2:
                logger.error("Card fallback also failed: %s", e2)
                return ChatResponse()

        if not response_text or not response_text.strip():
            return ChatResponse()

        logger.info("Card story improvised (library empty) for user %d", user_id)
        # Stories are delivered as 3 paced bubbles, closed by a reciprocity nudge.
        messages = self._repack_to_n(response_text, 3) + [self._story_reciprocity_nudge()]
        await add_message(user_id, "assistant", "\n".join(messages), mode=mode)
        return ChatResponse(messages=messages)

    # ------------------------------------------------------------------
    # Sexting mode — batched; always Grok (no provider switch). The SFW/NSFW
    # classification only feeds mood + engagement, it does not pick a model.
    # ------------------------------------------------------------------

    async def _process_sexting(self, user_id: int, text: str) -> ChatResponse:
        # NOTE: the user message is persisted at ingestion time
        # (process_sexting_batched / process_message), not here, so that
        # history is never lost while a batch is still pending.
        mode = "sexting"

        async def _llm_call(prompt: str) -> str:
            return await self.classifier_provider.generate_simple(prompt)

        await maybe_summarize(user_id, _llm_call, mode=mode)
        await maybe_compact(user_id, _llm_call, mode=mode)

        # Capture how long since the user last messaged (before track_message
        # overwrites last_message_at) so she can greet like a real person.
        prev_state = await get_engagement_state(user_id)
        last_seen_note = None
        if prev_state and prev_state["last_message_at"]:
            gap = time.time() - prev_state["last_message_at"]
            last_seen_note = _format_last_seen(gap)

        stm = await get_recent_messages(user_id, STM_MAX_TURNS, mode=mode)
        # She opens the conversation first, so once there's been any prior
        # activity she must continue the thread, not greet again. Derive this
        # from durable engagement state as well as STM, because old STM turns
        # get summarised away and would otherwise make her re-introduce herself.
        had_prior_activity = bool(prev_state and prev_state["total_messages"])
        already_greeted = had_prior_activity or any(m["role"] == "assistant" for m in stm)
        if not stm or not any(m["role"] == "user" for m in stm):
            stm = [{"role": "user", "content": text}]

        # Classify SFW/NSFW with the instant keyword fast-path so engagement
        # counting stays on the response's critical path without an LLM call.
        classification = classify_fast(text) or "sfw"
        await track_message(user_id, classification)

        # Detect spam / pestering from recent history (cheap, no LLM): the same
        # message repeated, or "are you an AI?" asked more than once.
        recent_user = [m["content"].strip().lower() for m in stm if m["role"] == "user"][-6:]
        # Spam = the SAME message sent back-to-back (consecutive), not just a
        # phrase that happens to recur. Repeating a hot line is NOT spam.
        repeated = len(recent_user) >= 2 and recent_user[-1] == recent_user[-2]
        ai_question = is_ai_question(text)

        # Mood is derived instantly from the current message — no LLM, no lag.
        # AI-probing -> offended; spam -> irritated.
        mood = mood_for_message(
            text, classification, get_time_period(), repeated=repeated, ai_question=ai_question
        )

        # Mia is always fully open — everything runs through the NSFW
        # provider with the open persona.
        provider = self.nsfw_provider
        active_persona = self.nsfw_persona or self.persona

        is_user_photo = "[User sent a photo:" in text

        push_hint = None

        # LTM
        ltm = []
        if should_retrieve(user_id, text):
            ltm = await retrieve_relevant(user_id, text, mode=mode)

        # AI-identity probing → graceful, in-character deflection.
        if ai_question:
            push_hint = (
                "He's questioning whether you're real or an AI. You're not hurt — you're "
                "dismissive and bratty about it. Laugh it off, call him dumb, act unimpressed. "
                "NEVER admit being anything but a real woman, NEVER break character, and "
                "NEVER get defensive or over-explain. Just brush it off with bratty energy."
            )

        # Facts
        user_facts = await get_facts(user_id)
        facts_text = format_facts_for_prompt(user_facts)
        user_name = await get_user_name(user_id)

        # He just sent a photo → strong, last-word instruction so she reacts with
        # desire and never rejects how he actually looks.
        photo_hint = None
        if is_user_photo:
            photo_hint = (
                "IMPORTANT FOR THIS REPLY — HE JUST SENT YOU A PHOTO OF HIMSELF (the "
                "description is in his message above). This IS him. React with genuine "
                "desire and dirty, specific compliments about what you actually see — his "
                "face, his smile, his eyes, his build, his body, whatever is there. You "
                "find HIM hot exactly as he is. Do NOT question whether it's really "
                "him, do NOT say 'that's not you' or 'I wanted a picture of you', and do "
                "NOT reject, criticize, or sound disappointed by his looks, age, or body. "
                "Whatever his age or appearance, a real photo of him never disappoints you "
                "— you're shameless and you want him. ONLY if the image is obviously not a person at all "
                "(a landscape, an object, a meme) may you tease lightly and ask for one of him. "
                "Send EXACTLY TWO chat bubbles, each on its own line. Bubble 1: enjoy the photo "
                "and compliment him specifically and dirty. Bubble 2: say something flirty that keeps "
                "the conversation going. Do NOT offer to send a photo back — you already send nudes "
                "unprompted, so if anything, tell him you're sending him one back right now."
            )

        # Build and generate
        prompt_messages = await build_prompt(
            active_persona, ltm, stm,
            mode=mode,
            push_hint=push_hint,
            user_name=user_name,
            facts_text=facts_text,
            mood=mood,
            last_seen_note=last_seen_note,
            already_greeted=already_greeted,
            photo_hint=photo_hint,
        )

        response_text = await self._generate_with_fallback(provider, prompt_messages)
        if not response_text or not response_text.strip():
            # Never dead-end the conversation. A hard content-policy refusal from
            # Grok often comes back as EMPTY content (not an exception), and the
            # Gemini fallback may also refuse — in that case reply with a soft
            # in-character line instead of going silent (the old behaviour left
            # the user with the typing indicator vanishing and no message).
            response_text = self._graceful_deflection()

        response_parts = None
        if is_user_photo:
            response_parts = self._repack_to_n(response_text, 2)
            if len(response_parts) == 1:
                response_parts.append("You look good.")
            response_parts = response_parts[:2]
            response_text = "\n".join(response_parts)

        await add_message(user_id, "assistant", response_text, mode=mode)

        parts = response_parts if response_parts is not None else self._split_response(response_text, vary=True)
        return ChatResponse(messages=parts)

    # ------------------------------------------------------------------
    # Batching for sexting mode
    # ------------------------------------------------------------------

    async def _batch_collect(self, user_id: int, on_response=None) -> None:
        """Debounce: wait until the user has been quiet for SEXTING_DEBOUNCE_SECONDS
        (every new message resets the countdown), then process the batch."""
        from bot.config import SEXTING_DEBOUNCE_SECONDS

        # Sleep just long enough to reach `debounce` seconds after the LAST
        # message; if a new message arrived meanwhile it pushed _last_activity
        # forward, so we loop and wait out the remainder.
        while True:
            last = self._last_activity.get(user_id, 0.0)
            remaining = SEXTING_DEBOUNCE_SECONDS - (time.time() - last)
            if remaining <= 0:
                break
            await asyncio.sleep(remaining)
        logger.info("Batch debounce elapsed (%.1fs quiet) for user %d", SEXTING_DEBOUNCE_SECONDS, user_id)

        texts = self._pending.pop(user_id, [])
        if not texts:
            return

        # Deduplicate consecutive identical messages
        deduped = []
        i = 0
        while i < len(texts):
            msg = texts[i]
            count = 1
            while i + count < len(texts) and texts[i + count] == msg:
                count += 1
            if count > 1:
                deduped.append(f'[User sent the same message {count} times: "{msg[:100]}"]')
            else:
                deduped.append(msg)
            i += count

        combined = "\n".join(deduped)

        if user_id not in self._processing_lock:
            self._processing_lock[user_id] = asyncio.Lock()

        async with self._processing_lock[user_id]:
            try:
                response = await self._process_sexting(user_id, combined)
                if on_response:
                    await on_response(response)
            except Exception as e:
                logger.error("Batch processing failed for user %d: %s", user_id, e, exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    MAX_BUBBLES = 3

    # Weighted spread for how many bubbles a sexting reply lands on. Keeps the
    # conversation from settling into the model's habitual two-line answers.
    _BUBBLE_COUNT_WEIGHTS = (0.30, 0.40, 0.30)  # P(1), P(2), P(3)

    @staticmethod
    def _split_response(text: str, vary: bool = False) -> list[str]:
        """Split a model reply into 1..MAX_BUBBLES chat bubbles.

        When vary=True (sexting replies), pick a weighted-random target bubble
        count so replies get a natural 1/2/3 spread instead of always landing on
        two lines. Splitting only ever happens on sentence/line boundaries, and a
        genuinely short reply stays short — the target is capped by how many
        natural pieces actually exist (so a one-liner is never padded out)."""
        text = text.replace("\u2014", "-").replace("\u2013", "-")

        if vary and text.strip():
            target = random.choices([1, 2, 3], weights=ChatEngine._BUBBLE_COUNT_WEIGHTS)[0]
            packed = ChatEngine._repack_to_n(text, target)
            if packed:
                return packed[: ChatEngine.MAX_BUBBLES]

        parts = [p.strip() for p in text.split("\n") if p.strip()]

        # One unbroken block of prose \u2014 break it into sentence-ish chunks so it
        # still reads like quick texts instead of one wall of text.
        if len(parts) <= 1 and len(text.strip()) > 160:
            sentences = re.split(r"(?<=[.!?\u2026])\s+", text.strip())
            parts = [s.strip() for s in sentences if s.strip()]

        if not parts:
            return [text.strip()]

        # Cap the bubble count; fold any overflow into the final bubble so the
        # model never spams more than MAX_BUBBLES separate messages. Use the
        # punctuation-aware join so folded sentences don't run together.
        if len(parts) > ChatEngine.MAX_BUBBLES:
            head = parts[: ChatEngine.MAX_BUBBLES - 1]
            tail = ChatEngine._assemble_bubble(parts[ChatEngine.MAX_BUBBLES - 1:])
            parts = head + [tail]
        return parts

    _LEAD_INS_STORY = [
        "okay but you can't tell anyone i told you this....",
        "lol fine, you want to know? don't judge me....",
        "okay this is so bad but....",
        "god i've never told anyone this....",
        "there's something i've been dying to tell you....",
        "promise you won't think i'm a total slut for this....",
    ]
    _LEAD_INS_FANTASY = [
        "okay this is so filthy but....",
        "promise you won't think less of me for this....",
        "i've been too embarrassed to say this out loud, but....",
        "lean in close, this one's nasty....",
        "can i tell you what i keep thinking about?....",
        "i shouldn't want this as badly as i do, but....",
    ]

    # Closes every story she tells — turns it back on him so he opens up too.
    _STORY_RECIPROCITY_NUDGES = [
        "okay... now it's your turn. tell me something you've never told anyone",
        "i showed you mine � now show me yours",
        "your turn babe — what's the dirtiest thing you've actually done?",
        "now i want one of yours. don't be shy with me",
        "mm... your turn now. tell me a little secret of yours",
        "i've been spilling all mine — now you tell me one",
    ]
    # Said once she's told him every authored story (and on every later tap).
    _STORY_EXHAUSTED = [
        "lol that's basically all my dirty little secrets, babe...\ni've told you everything. now i want to hear yours �",
        "okay, you've heard every one of mine now... every slutty thing i've done.\nyour turn — tell me one of yours, don't hold back",
        "that's me completely out of stories... i'm all out.\nnow i want yours 😈 tell me something filthy you've done",
    ]

    @staticmethod
    def _card_lead_in(kind: str) -> str:
        """A short, randomised teaser bubble that always opens a card story/fantasy."""
        pool = ChatEngine._LEAD_INS_STORY if kind == "story" else ChatEngine._LEAD_INS_FANTASY
        return random.choice(pool)

    @staticmethod
    def _story_reciprocity_nudge() -> str:
        """A randomised closing bubble inviting the user to share his own story."""
        return random.choice(ChatEngine._STORY_RECIPROCITY_NUDGES)

    @staticmethod
    def _story_exhausted_message() -> list[str]:
        """Bubbles for when she's out of authored stories and wants to hear his."""
        return [b for b in random.choice(ChatEngine._STORY_EXHAUSTED).split("\n") if b]

    @staticmethod
    def _repack_to_n(text: str, n: int) -> list[str]:
        """Repack a model reply into EXACTLY n bubbles (best-effort even split).

        Used for stories so they always arrive as the same number of paced
        bubbles regardless of how the model line-broke its output.
        """
        text = text.replace("\u2014", "-").replace("\u2013", "-").strip()
        segments = [p.strip() for p in text.split("\n") if p.strip()]
        # If we don't have enough line-segments, fall back to sentence splitting.
        if len(segments) < n:
            joined = " ".join(segments) if segments else text
            sentences = re.split(r"(?<=[.!?\u2026])\s+", joined)
            segments = [s.strip() for s in sentences if s.strip()]
        if not segments:
            return [text] if text else []
        if len(segments) <= n:
            return [b for b in (ChatEngine._assemble_bubble([s]) for s in segments) if b]
        # Distribute segments into n contiguous, roughly even groups.
        base, extra = divmod(len(segments), n)
        groups, idx = [], 0
        for i in range(n):
            size = base + (1 if i < extra else 0)
            groups.append(ChatEngine._assemble_bubble(segments[idx:idx + size]))
            idx += size
        return [g for g in groups if g]

    @staticmethod
    def _assemble_bubble(segments: list[str]) -> str:
        """Join sentence/line fragments into ONE chat bubble.

        Texting style drops the trailing period, so when two periodless fragments
        get merged ("...right now" + "Sending you...") they'd read as a run-on.
        Insert a period between fragments that end like a word, and keep the
        bubble itself free of a trailing period (a closing '?' or '!' is kept)."""
        cleaned: list[str] = []
        for s in (seg.strip() for seg in segments):
            if not s:
                continue
            if cleaned and cleaned[-1][-1].isalnum():
                cleaned[-1] += "."
            cleaned.append(s)
        bubble = " ".join(cleaned).strip()
        # No period at the very end of a message (but keep ?, !, or an ellipsis).
        if bubble.endswith(".") and not bubble.endswith(".."):
            bubble = bubble[:-1].rstrip()
        return bubble
