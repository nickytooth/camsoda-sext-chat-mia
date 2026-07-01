import base64
import logging
from openai import AsyncOpenAI
from bot.providers.base import LLMProvider
from bot.config import XAI_API_KEY, XAI_MODEL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_client = None

IMAGE_ANALYSIS_PROMPT = (
    "Describe what you see in this image in 1-2 sentences. "
    "Be specific about the content, setting, and any people visible. "
    "If it contains nudity or sexual content, describe it plainly without censoring. "
    "Return ONLY the description, nothing else."
)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    return _client


class GrokProvider(LLMProvider):
    async def generate(self, messages: list[dict]) -> str:
        client = _get_client()
        response = await client.chat.completions.create(
            model=XAI_MODEL,
            messages=messages,
            max_tokens=1024,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        return response.choices[0].message.content

    async def generate_simple(self, prompt: str) -> str:
        client = _get_client()
        response = await client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        return response.choices[0].message.content

    async def analyze_image(self, image_bytes: bytes) -> str:
        """Analyze an image using Grok's vision capability."""
        client = _get_client()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = await client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=256,
        )
        description = response.choices[0].message.content
        logger.info("Image analysis result: %s", description[:100])
        return description
