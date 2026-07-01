from google import genai
from bot.providers.base import LLMProvider
from bot.config import GOOGLE_API_KEY, GOOGLE_MODEL

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


class GeminiProvider(LLMProvider):
    def __init__(self, model: str | None = None, thinking_budget: int | None = None):
        self._model = model or GOOGLE_MODEL
        # thinking_budget=0 disables Gemini "thinking" — much lower latency for
        # short generations. Leave None to keep the model's default behaviour.
        self._thinking_budget = thinking_budget

    def _build_config(self, system_msg: str | None = None) -> "genai.types.GenerateContentConfig":
        # Safety filters are disabled across the board: this provider also runs
        # summarization/classification over explicit sexting transcripts, which
        # the default filters would otherwise drop (empty response).
        return genai.types.GenerateContentConfig(
            system_instruction=system_msg if system_msg else None,
            # Cap, not a forced length: short classifications stay short, but
            # long summarization/compaction JSON won't get truncated mid-object.
            max_output_tokens=2048,
            thinking_config=(
                genai.types.ThinkingConfig(thinking_budget=self._thinking_budget)
                if self._thinking_budget is not None else None
            ),
            safety_settings=[
                genai.types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                genai.types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                genai.types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                genai.types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            ],
        )

    async def generate(self, messages: list[dict]) -> str:
        client = _get_client()

        system_msg = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        response = await client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=self._build_config(system_msg),
        )
        if response.text is None:
            raise RuntimeError("Gemini returned empty response (likely safety-filtered)")
        return response.text

    async def generate_simple(self, prompt: str) -> str:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._build_config(),
        )
        if response.text is None:
            raise RuntimeError("Gemini returned empty response (likely safety-filtered)")
        return response.text
