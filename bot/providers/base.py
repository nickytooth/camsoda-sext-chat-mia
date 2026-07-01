from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict]) -> str:
        ...

    @abstractmethod
    async def generate_simple(self, prompt: str) -> str:
        """Single-prompt call for summarization/classification (no message history)."""
        ...
