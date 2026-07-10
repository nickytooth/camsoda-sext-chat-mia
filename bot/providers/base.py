from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict], temperature: float | None = None) -> str:
        """Generate a chat completion. `temperature=None` keeps the model's
        default; callers may pass a value to match the moment (e.g. hotter
        when aroused, tighter when she's firing back)."""
        ...

    @abstractmethod
    async def generate_simple(self, prompt: str) -> str:
        """Single-prompt call for summarization/classification (no message history)."""
        ...
