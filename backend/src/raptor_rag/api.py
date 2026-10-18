import shutil
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Response, UploadFile
from pydantic import BaseModel, StringConstraints
from qdrant_client import QdrantClient

from raptor_rag import store
from raptor_rag.config import Settings
from raptor_rag.db import DB
from raptor_rag.embed import BGEM3Embedder, Embedder
from raptor_rag.ingest import Deps, Worker, doc_path, ingest_document
from raptor_rag.pdf import validate_pdf

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class KBIn(BaseModel):
    name: Name


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    qdrant: QdrantClient | None = None,
) -> FastAPI:
    """Build the app. Tests pass in-memory Qdrant and a fake embedder.

    Run with: uvicorn raptor_rag.api:create_app --factory
    """
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = DB(settings.data_dir / "app.db")
    db.fail_in_flight()
    deps = Deps(
        settings=settings,
        db=db,
        qdrant=qdrant or QdrantClient(location=settings.qdrant_location),
        embedder=embedder or BGEM3Embedder(settings.embed_model),
    )
    worker = Worker(lambda kb_id, doc_id: ingest_document(deps, kb_id, doc_id))

    app = FastAPI(title="RAPTOR RAG")
    app.state.deps = deps
    app.state.worker = worker

    def get_kb(kb_id: str) -> dict:
        kb = db.get_kb(kb_id)
        if kb is None:
            raise HTTPException(404, "Knowledge base not found.")
        return kb

    def get_doc(kb_id: str, doc_id: str) -> dict:
        doc = db.get_document(doc_id)
        if doc is None or doc["kb_id"] != kb_id:
            raise HTTPException(404, "Document not found.")
        return doc

    @app.post("/kb", status_code=201)
    def create_kb(body: KBIn) -> dict:
        kb = db.create_kb(body.name)
        try:
            store.create_collection(deps.qdrant, kb["id"], deps.embedder.dim)
        except Exception:
            db.delete_kb(kb["id"])  # don't leave a KB without a collection
            raise
        return kb

    @app.get("/kb")
    def list_kbs() -> list[dict]:
        return db.list_kbs()

    @app.patch("/kb/{kb_id}")
    def rename_kb(kb_id: str, body: KBIn) -> dict:
        get_kb(kb_id)
        db.rename_kb(kb_id, body.name)
        return get_kb(kb_id)

    @app.delete("/kb/{kb_id}", status_code=204)
    def delete_kb(kb_id: str) -> Response:
        get_kb(kb_id)
        store.drop_collection(deps.qdrant, kb_id)
        shutil.rmtree(settings.data_dir / "files" / kb_id, ignore_errors=True)
        db.delete_kb(kb_id)  # cascades to documents
        return Response(status_code=204)

    @app.post("/kb/{kb_id}/documents", status_code=202)
    def upload_document(kb_id: str, file: UploadFile) -> dict:
        get_kb(kb_id)
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(max_bytes + 1)
        try:
            validate_pdf(data, max_bytes)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        filename = Path(file.filename or "document.pdf").name[:200]
        doc = db.add_document(kb_id, filename)
        path = doc_path(settings, kb_id, doc["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        worker.submit(kb_id, doc["id"])
        return doc

    @app.get("/kb/{kb_id}/status")
    def kb_status(kb_id: str) -> dict:
        return {"kb": get_kb(kb_id), "documents": db.list_documents(kb_id)}

    @app.delete("/kb/{kb_id}/documents/{doc_id}", status_code=204)
    def delete_document(kb_id: str, doc_id: str) -> Response:
        get_doc(kb_id, doc_id)
        store.delete_document(deps.qdrant, kb_id, doc_id)
        doc_path(settings, kb_id, doc_id).unlink(missing_ok=True)
        db.delete_document(doc_id)
        return Response(status_code=204)

    @app.post("/kb/{kb_id}/documents/{doc_id}/reindex", status_code=202)
    def reindex_document(kb_id: str, doc_id: str) -> dict:
        get_doc(kb_id, doc_id)
        db.set_status(doc_id, "queued")
        worker.submit(kb_id, doc_id)
        return get_doc(kb_id, doc_id)

    return app
