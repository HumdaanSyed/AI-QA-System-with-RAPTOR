import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

from qdrant_client import QdrantClient, models

from raptor_rag import store
from raptor_rag.chunking import chunk_blocks
from raptor_rag.config import Settings
from raptor_rag.db import DB
from raptor_rag.embed import Embedder
from raptor_rag.llm import LLM
from raptor_rag.pdf import extract_blocks
from raptor_rag.raptor import Node, build_tree, summarizer
from raptor_rag.rerank import Reranker

log = logging.getLogger(__name__)

EMBED_BATCH = 32


@dataclass
class Deps:
    settings: Settings
    db: DB
    qdrant: QdrantClient
    embedder: Embedder
    reranker: Reranker
    llm: LLM  # answers
    summarizer: LLM  # RAPTOR summaries


def to_point(node: Node, doc: dict) -> models.PointStruct:
    payload = {
        "doc_id": doc["id"],
        "doc_name": doc["filename"],
        "level": node.level,
        "text": node.text,
        "page_start": node.page_start,
        "page_end": node.page_end,
    }
    if node.child_ids:
        payload["child_ids"] = node.child_ids
    return models.PointStruct(
        id=node.id, vector={"dense": node.dense, "sparse": node.sparse}, payload=payload
    )


def doc_path(settings: Settings, kb_id: str, doc_id: str) -> Path:
    return settings.data_dir / "files" / kb_id / f"{doc_id}.pdf"


def ingest_document(deps: Deps, kb_id: str, doc_id: str) -> None:
    """Parse, chunk, embed and index one document (plus its RAPTOR tree), recording progress."""
    db = deps.db
    doc = db.get_document(doc_id)
    if doc is None:  # deleted while queued
        return
    try:
        db.set_status(doc_id, "parsing")
        chunks = chunk_blocks(
            extract_blocks(doc_path(deps.settings, kb_id, doc_id)), deps.settings.chunk_chars
        )
        if not chunks:
            raise ValueError("No extractable text found (is it a scanned PDF?).")

        store.delete_document(deps.qdrant, kb_id, doc_id)  # makes re-indexing idempotent
        db.set_status(doc_id, "embedding")
        leaves: list[Node] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors = deps.embedder.embed([c.text for c in batch])
            nodes = [
                Node(str(uuid.uuid4()), c.text, dense, sparse, c.page_start, c.page_end)
                for c, (dense, sparse) in zip(batch, vectors, strict=True)
            ]
            store.upsert(deps.qdrant, kb_id, [to_point(n, doc) for n in nodes])
            leaves += nodes
            db.set_status(doc_id, "embedding", progress=len(leaves) * 100 // len(chunks))

        kb = db.get_kb(kb_id)
        if kb and kb["raptor_enabled"]:
            db.set_status(doc_id, "building_tree")
            summarize = summarizer(deps.summarizer, db, doc["filename"])
            tree = build_tree(leaves, summarize, deps.embedder.embed)
            store.upsert(deps.qdrant, kb_id, [to_point(n, doc) for n in tree])

        if db.get_document(doc_id) is None:  # deleted mid-ingest: remove what we just wrote
            store.delete_document(deps.qdrant, kb_id, doc_id)
            return
        db.set_status(doc_id, "ready", progress=100)
    except Exception as e:
        log.exception("Ingestion failed for document %s", doc_id)
        # Only our own messages are user-facing; library errors can contain server paths.
        message = str(e) if isinstance(e, ValueError) else "Could not process this PDF."
        db.set_status(doc_id, "failed", error=message)


class Worker:
    """Runs ingestion jobs one at a time on a background thread.

    shortcut: jobs live in memory and are lost on restart (marked failed at startup);
    use a persistent queue (e.g. Redis + RQ) before running multiple server instances.
    """

    def __init__(self, handle: Callable[[str, str], None]):
        self.handle = handle
        self.queue: Queue[tuple[str, str]] = Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, kb_id: str, doc_id: str) -> None:
        self.queue.put((kb_id, doc_id))

    def wait(self) -> None:
        """Block until every submitted job has finished (used by tests)."""
        self.queue.join()

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            try:
                self.handle(*job)
            except Exception:  # keep the thread alive for later jobs
                log.exception("Job %s failed", job)
            finally:
                self.queue.task_done()
