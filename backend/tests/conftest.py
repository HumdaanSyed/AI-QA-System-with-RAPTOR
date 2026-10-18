import zlib
from collections.abc import Iterator

import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from raptor_rag.api import create_app
from raptor_rag.config import Settings


class FakeEmbedder:
    """Deterministic bag-of-words vectors, so tests need no model download."""

    dim = 16

    def embed(self, texts: list[str]) -> list[tuple[list[float], models.SparseVector]]:
        out = []
        for text in texts:
            hashes = [zlib.crc32(w.encode()) for w in text.lower().split()]
            dense = [0.0] * self.dim
            for h in hashes:
                dense[h % self.dim] += 1.0
            dense[0] += 1e-3  # never all-zero (cosine)
            indices = sorted({h % 100_000 for h in hashes})
            out.append((dense, models.SparseVector(indices=indices, values=[1.0] * len(indices))))
        return out


class FakeReranker:
    """Scores passages by how many question words they contain."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        words = set(query.lower().split())
        return [float(len(words & set(p.lower().split()))) for p in passages]


class FakeLLM:
    """Streams a scripted answer and records the prompts it was given."""

    model = "fake"

    def __init__(self, script: list[str] | None = None):
        self.script = script or ["The mean is the average [1]. ", "See also [2, 9]."]
        self.prompts: list[str] = []

    def stream(self, instructions: str, prompt: str) -> Iterator[str]:
        self.prompts.append(prompt)
        yield from self.script

    def complete(self, instructions: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Summary: " + " ".join(prompt.split()[:20])


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


@pytest.fixture
def app(tmp_path) -> FastAPI:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        data_dir=tmp_path,
        qdrant_location=":memory:",
        max_upload_mb=1,
    )
    return create_app(
        settings,
        embedder=FakeEmbedder(),
        qdrant=QdrantClient(":memory:"),
        reranker=FakeReranker(),
        llm=FakeLLM(),
        summarizer=FakeLLM(),
    )


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
