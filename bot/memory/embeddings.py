"""Bounded embedding helpers used by long-term memory."""

from __future__ import annotations

import unicodedata

import numpy as np
from openai import AsyncOpenAI

from bot.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


MAX_EMBEDDING_INPUT_CHARS = 4_000
EMBEDDING_BATCH_SIZE = 64

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


def _prepare_embedding_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("embedding input must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf", "Cs"} else char
        for char in text
    )
    text = " ".join(text.split()).strip()
    if not text:
        raise ValueError("embedding input must not be empty")
    # Queries are user-controlled and previously had no input bound.  A prefix
    # is deterministic and sufficient for retrieval without sending an
    # arbitrarily large message to the embedding provider.
    return text[:MAX_EMBEDDING_INPUT_CHARS]


def _response_embeddings(response, expected: int) -> list[np.ndarray]:
    data = list(getattr(response, "data", []) or [])
    if len(data) != expected:
        raise RuntimeError(
            f"embedding provider returned {len(data)} vectors for {expected} inputs"
        )
    vectors = [np.asarray(item.embedding, dtype=np.float32) for item in data]
    for vector in vectors:
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise RuntimeError("embedding provider returned an invalid vector")
    return vectors


async def embed_text(text: str) -> np.ndarray:
    prepared = _prepare_embedding_text(text)
    client = _get_client()
    response = await client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=prepared,
    )
    return _response_embeddings(response, 1)[0]


async def embed_texts(texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []
    if not isinstance(texts, list):
        raise TypeError("embedding batch must be a list of strings")

    prepared = [_prepare_embedding_text(text) for text in texts]
    client = _get_client()
    vectors: list[np.ndarray] = []
    for offset in range(0, len(prepared), EMBEDDING_BATCH_SIZE):
        batch = prepared[offset:offset + EMBEDDING_BATCH_SIZE]
        response = await client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        vectors.extend(_response_embeddings(response, len(batch)))
    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    try:
        left = np.asarray(a, dtype=np.float32)
        right = np.asarray(b, dtype=np.float32)
    except (TypeError, ValueError):
        return 0.0
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.shape != right.shape
        or left.size == 0
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        return 0.0
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not np.isfinite(norm) or norm == 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / norm)
    return similarity if np.isfinite(similarity) else 0.0
