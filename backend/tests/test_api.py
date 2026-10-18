from dataclasses import replace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from raptor_rag import store
from raptor_rag.ingest import doc_path, ingest_document

from .conftest import FakeEmbedder, make_pdf


def create_kb(client: TestClient, name: str = "Books") -> str:
    r = client.post("/kb", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def upload(client: TestClient, kb_id: str, pages: list[str], name: str = "a.pdf") -> str:
    r = client.post(
        f"/kb/{kb_id}/documents", files={"file": (name, make_pdf(pages), "application/pdf")}
    )
    assert r.status_code == 202, r.text
    return r.json()["id"]


def docs_by_id(client: TestClient, kb_id: str) -> dict[str, dict]:
    return {d["id"]: d for d in client.get(f"/kb/{kb_id}/status").json()["documents"]}


def test_kb_crud(client: TestClient, app: FastAPI):
    kb_id = create_kb(client, "  Books  ")
    assert client.get("/kb").json()[0]["name"] == "Books"

    r = client.patch(f"/kb/{kb_id}", json={"name": "Library"})
    assert r.json()["name"] == "Library"
    assert client.patch(f"/kb/{kb_id}", json={"name": "   "}).status_code == 422

    qdrant = app.state.deps.qdrant
    assert qdrant.collection_exists(store.collection_name(kb_id))
    assert client.delete(f"/kb/{kb_id}").status_code == 204
    assert client.get("/kb").json() == []
    assert not qdrant.collection_exists(store.collection_name(kb_id))
    assert client.get(f"/kb/{kb_id}/status").status_code == 404


def test_upload_rejects_non_pdf_and_oversize(client: TestClient):
    kb_id = create_kb(client)
    r = client.post(
        f"/kb/{kb_id}/documents", files={"file": ("x.pdf", b"hello", "application/pdf")}
    )
    assert r.status_code == 422
    big = b"%PDF-" + b"0" * (1024 * 1024)  # test app allows 1 MB
    r = client.post(f"/kb/{kb_id}/documents", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 422
    assert docs_by_id(client, kb_id) == {}


def test_ingest_indexes_chunks_with_pages(client: TestClient, app: FastAPI):
    kb_id = create_kb(client)
    doc_id = upload(client, kb_id, ["Neural networks", "Backpropagation", "Gradient descent"])
    app.state.worker.wait()

    doc = docs_by_id(client, kb_id)[doc_id]
    assert (doc["status"], doc["progress"]) == ("ready", 100)
    points, _ = app.state.deps.qdrant.scroll(store.collection_name(kb_id), with_payload=True)
    payload = points[0].payload
    assert payload["doc_id"] == doc_id and payload["doc_name"] == "a.pdf"
    assert (payload["page_start"], payload["page_end"]) == (1, 3)


def test_corrupt_pdf_marks_document_failed(client: TestClient, app: FastAPI):
    kb_id = create_kb(client)
    r = client.post(
        f"/kb/{kb_id}/documents",
        files={"file": ("bad.pdf", b"%PDF-1.7 garbage", "application/pdf")},
    )
    app.state.worker.wait()
    doc = docs_by_id(client, kb_id)[r.json()["id"]]
    assert doc["status"] == "failed"
    assert doc["error"] == "Could not process this PDF."  # no server paths leaked


def test_document_deleted_mid_ingest_leaves_no_points(client: TestClient, app: FastAPI):
    deps = app.state.deps
    kb_id = create_kb(client)
    doc_id = upload(client, kb_id, ["Decision trees", "Random forests"])
    app.state.worker.wait()

    class DeletingEmbedder(FakeEmbedder):
        def embed(self, texts):
            deps.db.delete_document(doc_id)  # user deletes while chunks are being written
            return super().embed(texts)

    ingest_document(replace(deps, embedder=DeletingEmbedder()), kb_id, doc_id)
    assert store.count_document(deps.qdrant, kb_id, doc_id) == 0


def test_delete_document_removes_only_its_data(client: TestClient, app: FastAPI):
    qdrant, settings = app.state.deps.qdrant, app.state.deps.settings
    kb_id = create_kb(client)
    keep = upload(client, kb_id, ["Statistics basics", "Mean and median"], "keep.pdf")
    drop = upload(client, kb_id, ["Hashing trick", "Sparse vectors"], "drop.pdf")
    app.state.worker.wait()
    keep_count = store.count_document(qdrant, kb_id, keep)
    assert keep_count > 0 and store.count_document(qdrant, kb_id, drop) > 0

    assert client.delete(f"/kb/{kb_id}/documents/{drop}").status_code == 204
    assert store.count_document(qdrant, kb_id, drop) == 0
    assert store.count_document(qdrant, kb_id, keep) == keep_count
    assert not doc_path(settings, kb_id, drop).exists()
    assert list(docs_by_id(client, kb_id)) == [keep]
    assert client.delete(f"/kb/{kb_id}/documents/{drop}").status_code == 404


def test_reindex_does_not_duplicate_points(client: TestClient, app: FastAPI):
    qdrant = app.state.deps.qdrant
    kb_id = create_kb(client)
    doc_id = upload(client, kb_id, ["Clustering", "Gaussian mixtures"])
    app.state.worker.wait()
    before = store.count_document(qdrant, kb_id, doc_id)

    assert client.post(f"/kb/{kb_id}/documents/{doc_id}/reindex").status_code == 202
    app.state.worker.wait()
    assert store.count_document(qdrant, kb_id, doc_id) == before
    assert docs_by_id(client, kb_id)[doc_id]["status"] == "ready"


def test_document_routes_check_kb_ownership(client: TestClient, app: FastAPI):
    kb_a, kb_b = create_kb(client, "A"), create_kb(client, "B")
    doc_id = upload(client, kb_a, ["Owned by A"])
    app.state.worker.wait()
    assert client.delete(f"/kb/{kb_b}/documents/{doc_id}").status_code == 404
    assert client.post(f"/kb/{kb_b}/documents/{doc_id}/reindex").status_code == 404


def test_cors_allows_only_the_frontend_origin(client: TestClient):
    def preflight(origin: str):
        return client.options(
            "/kb", headers={"Origin": origin, "Access-Control-Request-Method": "POST"}
        )

    allowed = preflight("http://localhost:3000")
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in preflight("https://evil.example").headers
