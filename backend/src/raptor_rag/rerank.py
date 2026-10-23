from typing import Any, Protocol

from raptor_rag.embed import MODEL_LOCK


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance score per passage (higher is more relevant)."""
        ...


class BGEReranker:
    """Cross-encoder: reads the query and passage together, so it ranks better than vectors."""

    def __init__(self, model_name: str):
        from FlagEmbedding import FlagReranker  # pyright: ignore[reportMissingImports]

        self.model = FlagReranker(model_name, use_fp16=True)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        with MODEL_LOCK:
            scores: Any = self.model.compute_score([(query, p) for p in passages], normalize=True)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]
