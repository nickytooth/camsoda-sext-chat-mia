import json
import logging
from bot.memory.embeddings import embed_texts
from bot.memory.ltm import store_memory, count_memories, get_all_memories, delete_memories_by_ids, find_similar_memory, update_memory
from bot.memory.stm import get_oldest_messages, delete_messages_by_ids, count_turns
from bot.config import STM_MAX_TURNS, STM_SUMMARIZE_BATCH, LTM_COMPACTION_THRESHOLD

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Analyze the following conversation and extract TWO things:

1. "memories": structured memory entries, each with:
   - category: one of "fact", "preference", "relationship", "event", "thread"
   - content: a concise statement (1-2 sentences max)
   - importance: integer 1-10 (10 = critical identity info like name, 1 = trivial)

2. "facts": key-value pairs for specific known facts about the user. Use ONLY these keys when applicable:
   name, location, age, job, gender, boundaries, agreed_prices, relationship_status
   Also include any other notable preferences or details as custom keys (e.g. "favorite_color", "pet_name", "kinks").

IMPORTANT — FICTION vs REALITY:
Lines tagged [STORY] or [FANTASY] are fiction: role-play scenes, invented
fantasies, or erotic stories the bot told. The events in them did NOT happen
between the user and the bot. Do NOT extract their events, settings, or
personas as real "facts" or identity memories. Only extract real-world details
the user genuinely revealed about themselves (typically from untagged lines).
You MAY record a low-importance "thread" memory about an ongoing story or a
fantasy already shared (for continuity / avoiding repeats), but never as a
real event or fact.

Return ONLY a JSON object with both keys. No other text.

Example:
{
  "memories": [
    {"category": "fact", "content": "User's name is Alex and he lives in London", "importance": 10},
    {"category": "preference", "content": "User likes being called 'good boy'", "importance": 7}
  ],
  "facts": [
    {"key": "name", "value": "Alex"},
    {"key": "location", "value": "London"},
    {"key": "kinks", "value": "likes being called good boy"}
  ]
}

Conversation:
"""

COMPACTION_PROMPT = """You have the following memory entries about a user. Some may be duplicates or overlapping.
Merge them into a clean, deduplicated list. Keep the most important and up-to-date information.
Return ONLY a JSON array with the same format.

Entries:
"""


def _extract_json(raw: str):
    """Parse a JSON object/array from an LLM response.

    Gemini (and others) often wrap JSON in markdown ```json ... ``` fences or
    add stray prose around it, which makes a direct json.loads() fail. This
    strips fences and, as a fallback, slices from the first opening bracket to
    the last closing bracket before parsing.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the closing fence.
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    if not (text.startswith("{") or text.startswith("[")):
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        end = max(text.rfind("}"), text.rfind("]"))
        if starts and end > min(starts):
            text = text[min(starts):end + 1]
    return json.loads(text)


def _format_messages_for_summary(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        prefix = "User" if msg["role"] == "user" else "Bot"
        if msg.get("mode") == "story":
            tag = "[STORY] "
        elif msg.get("tag") in ("fantasy_card", "story_card"):
            # Card-delivered fiction (fantasies/stories she told) — must never be
            # summarized as real events between her and the user.
            tag = "[FANTASY] "
        else:
            tag = ""
        lines.append(f"{tag}{prefix}: {msg['content']}")
    return "\n".join(lines)


async def maybe_summarize(user_id: int, llm_call, mode: str = "sexting") -> bool:
    # Per-mode memory: story and sexting each summarize their own transcript into
    # their own long-term pool, so nothing bleeds between the two.
    turns = await count_turns(user_id, mode=mode)
    if turns < STM_MAX_TURNS:
        return False

    oldest = await get_oldest_messages(user_id, STM_SUMMARIZE_BATCH, mode=mode)
    if not oldest:
        return False

    conversation_text = _format_messages_for_summary(oldest)
    prompt = SUMMARIZE_PROMPT + conversation_text

    try:
        raw_response = await llm_call(prompt)
        parsed = _extract_json(raw_response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Summarization failed: %s", e)
        return False
    except Exception as e:
        logger.error("Summarization LLM call failed: %s", e)
        return False

    # Handle both old format (plain array) and new format (object with memories+facts)
    if isinstance(parsed, list):
        entries = parsed
        facts = []
    else:
        entries = parsed.get("memories", [])
        facts = parsed.get("facts", [])

    # Store memories as embeddings in LTM
    if entries:
        valid_entries = [e for e in entries if e.get("content")]
        texts = [entry["content"] for entry in valid_entries]
        if not texts:
            logger.warning("Summarization: no valid entries with content, skipping")
            message_ids = [msg["id"] for msg in oldest]
            await delete_messages_by_ids(message_ids)
            return True
        embeddings = await embed_texts(texts)

        new_count = 0
        updated_count = 0
        for entry, embedding in zip(valid_entries, embeddings):
            content = entry["content"]
            existing = await find_similar_memory(user_id, embedding, threshold=0.85, mode=mode)
            if existing:
                new_imp = max(existing["importance"], entry.get("importance", 5))
                await update_memory(existing["id"], content, new_imp, embedding)
                updated_count += 1
            else:
                await store_memory(
                    user_id=user_id,
                    category=entry.get("category", "fact"),
                    content=content,
                    importance=entry.get("importance", 5),
                    embedding=embedding,
                    mode=mode,
                )
                new_count += 1

        logger.info("Summarized %d messages for user %d: %d new, %d updated memories", len(oldest), user_id, new_count, updated_count)

    # Store structured facts
    if facts:
        from bot.memory.facts import upsert_fact
        fact_count = 0
        for fact in facts:
            key = fact.get("key", "").strip().lower()
            value = fact.get("value", "").strip()
            if key and value:
                await upsert_fact(user_id, key, value)
                fact_count += 1
        if fact_count:
            logger.info("Extracted %d facts for user %d", fact_count, user_id)

    message_ids = [msg["id"] for msg in oldest]
    await delete_messages_by_ids(message_ids)

    return True


async def maybe_compact(user_id: int, llm_call, mode: str = "sexting") -> bool:
    mem_count = await count_memories(user_id, mode=mode)
    if mem_count < LTM_COMPACTION_THRESHOLD:
        return False

    memories = await get_all_memories(user_id, mode=mode)
    entries_text = json.dumps(
        [{"category": m["category"], "content": m["content"], "importance": m["importance"]} for m in memories],
        indent=2,
    )
    prompt = COMPACTION_PROMPT + entries_text

    try:
        raw_response = await llm_call(prompt)
        new_entries = _extract_json(raw_response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Compaction failed: %s", e)
        return False
    except Exception as e:
        logger.error("Compaction LLM call failed: %s", e)
        return False

    if not isinstance(new_entries, list) or not new_entries:
        logger.warning("Compaction: no valid entries returned, keeping old memories")
        return False

    valid_entries = [e for e in new_entries if e.get("content")]
    if not valid_entries:
        logger.warning("Compaction: no entries with content, keeping old memories")
        return False

    texts = [entry["content"] for entry in valid_entries]
    embeddings = await embed_texts(texts)

    # Store new memories BEFORE deleting old ones to prevent data loss on failure
    for entry, embedding in zip(valid_entries, embeddings):
        await store_memory(
            user_id=user_id,
            category=entry.get("category", "fact"),
            content=entry["content"],
            importance=entry.get("importance", 5),
            embedding=embedding,
            mode=mode,
        )

    old_ids = [m["id"] for m in memories]
    await delete_memories_by_ids(old_ids)

    logger.info("Compacted %d memories into %d for user %d", len(old_ids), len(valid_entries), user_id)
    return True
