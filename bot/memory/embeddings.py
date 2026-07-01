import numpy as np
from openai import AsyncOpenAI
from bot.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


async def embed_text(text: str) -> np.ndarray:
    client = _get_client()
    response = await client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


async def embed_texts(texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []
    client = _get_client()
    response = await client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)
