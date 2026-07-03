import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


class FakeEmbedder:
    """Deterministic, offline embedder: hash-derived unit vectors so identical
    text always maps to the identical vector. Good enough to test index
    round-trips without the network."""

    DIM = 16

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b - 128) / 128.0 for b in digest[: self.DIM]]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def sample_docs_dir():
    return Path(__file__).parents[1] / "data" / "raw"
