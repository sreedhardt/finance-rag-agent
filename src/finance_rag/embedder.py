"""Embeddings via the Gemini embeddings API (gemini-embedding-001).

The indexer takes any object with embed_documents/embed_query, so tests inject
a deterministic fake and never hit the network."""

from __future__ import annotations

import math

from google import genai
from google.genai import types

from . import config

_BATCH_SIZE = 50


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class GeminiEmbedder:
    def __init__(self, client: genai.Client | None = None,
                 model: str = config.EMBED_MODEL, dim: int = config.EMBED_DIM):
        self._client = client or genai.Client()
        self._model = model
        self._dim = dim

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start:start + _BATCH_SIZE]
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dim,
                ),
            )
            vectors.extend(_normalize(e.values) for e in response.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
