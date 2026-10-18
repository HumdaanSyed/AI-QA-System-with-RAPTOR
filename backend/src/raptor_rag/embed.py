from typing import Protocol

from qdrant_client import models


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[tuple[list[float], models.SparseVector]]:
        """Return a (dense, sparse) vector pair for each text."""
        ...


class BGEM3Embedder:
    """BGE-M3 produces dense and sparse (lexical) vectors in one pass."""

    dim = 1024

    def __init__(self, model_name: str):
        from FlagEmbedding import BGEM3FlagModel  # pyright: ignore[reportMissingImports]

        self.model = BGEM3FlagModel(model_name, use_fp16=False)

    def embed(self, texts: list[str]) -> list[tuple[list[float], models.SparseVector]]:
        out = self.model.encode(texts, return_dense=True, return_sparse=True, batch_size=16)
        result = []
        for dense, weights in zip(out["dense_vecs"], out["lexical_weights"], strict=True):
            sparse = models.SparseVector(
                indices=[int(k) for k in weights], values=[float(v) for v in weights.values()]
            )
            result.append((dense.tolist(), sparse))
        return result
