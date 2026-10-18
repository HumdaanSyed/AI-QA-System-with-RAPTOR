import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import models
from sklearn.mixture import GaussianMixture

from raptor_rag import store
from raptor_rag.raptor import Node, build_tree, cluster

from .test_api import create_kb, upload


def blobs(per_blob: int = 20, n_blobs: int = 3, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(n_blobs, dim)) * 10
    return np.vstack([c + rng.normal(size=(per_blob, dim)) for c in centers])


def test_cluster_covers_every_node_and_is_repeatable():
    embeddings = blobs()
    groups = cluster(embeddings)
    assert len(groups) >= 2
    assert set().union(*groups) == set(range(len(embeddings)))
    assert cluster(embeddings) == groups  # seeded, so re-indexing hits the summary cache


def test_build_tree_links_levels_and_page_ranges():
    embeddings = blobs()
    leaves = [
        Node(
            str(i),
            f"text {i}",
            list(map(float, e)),
            models.SparseVector(indices=[], values=[]),
            i,
            i,
        )
        for i, e in enumerate(embeddings, start=1)
    ]
    rng = np.random.default_rng(1)

    def embed(texts):
        return [
            (list(rng.normal(size=16)), models.SparseVector(indices=[], values=[])) for _ in texts
        ]

    tree = build_tree(leaves, lambda groups: [f"summary of {len(g)}" for g in groups], embed)
    by_id = {n.id: n for n in leaves + tree}
    assert {n.level for n in tree} >= {1, 2}
    assert len([n for n in tree if n.level == max(t.level for t in tree)]) == 1  # one root
    level1 = [n for n in tree if n.level == 1]
    assert {c for n in level1 for c in n.child_ids} == {n.id for n in leaves}
    for node in tree:
        children = [by_id[c] for c in node.child_ids]
        assert all(c.level == node.level - 1 for c in children)
        assert node.page_start == min(c.page_start for c in children)
        assert node.page_end == max(c.page_end for c in children)


def summary_points(app: FastAPI, kb_id: str) -> list[dict]:
    points, _ = app.state.deps.qdrant.scroll(store.collection_name(kb_id), limit=1000)
    return [p.payload for p in points if p.payload["level"] > 0]


def test_ingest_builds_tree_and_delete_removes_it(client: TestClient, app: FastAPI):
    app.state.deps.settings.chunk_chars = 20  # several chunks from a small PDF
    kb_id = create_kb(client)
    doc_id = upload(client, kb_id, ["Mean median and mode", "Variance and deviation", "Outliers"])
    app.state.worker.wait()

    summaries = summary_points(app, kb_id)
    assert summaries and all(s["doc_id"] == doc_id and s["child_ids"] for s in summaries)
    summarizer = app.state.deps.summarizer
    calls = len(summarizer.prompts)

    client.post(f"/kb/{kb_id}/documents/{doc_id}/reindex")
    app.state.worker.wait()
    assert len(summarizer.prompts) == calls  # summaries came from the cache
    assert len(summary_points(app, kb_id)) == len(summaries)  # no duplicates

    client.delete(f"/kb/{kb_id}/documents/{doc_id}")
    assert store.count_document(app.state.deps.qdrant, kb_id, doc_id) == 0


def test_raptor_can_be_disabled_per_knowledge_base(client: TestClient, app: FastAPI):
    app.state.deps.settings.chunk_chars = 20
    r = client.post("/kb", json={"name": "Flat", "raptor_enabled": False})
    assert r.json()["raptor_enabled"] is False
    kb_id = r.json()["id"]
    upload(client, kb_id, ["Mean median and mode", "Variance and deviation"])
    app.state.worker.wait()
    assert summary_points(app, kb_id) == []
    assert app.state.deps.summarizer.prompts == []


def test_cluster_skips_mixture_fits_that_fail(monkeypatch):
    # Collapsed components raise ValueError in scikit-learn (seen on Linux CI).
    real_fit = GaussianMixture.fit

    def flaky_fit(self, X, y=None):
        if self.n_components % 2:
            raise ValueError("ill-defined empirical covariance")
        return real_fit(self, X)

    monkeypatch.setattr(GaussianMixture, "fit", flaky_fit)
    embeddings = blobs()
    groups = cluster(embeddings)
    assert set().union(*groups) == set(range(len(embeddings)))

    monkeypatch.setattr(
        GaussianMixture, "fit", lambda self, X, y=None: (_ for _ in ()).throw(ValueError())
    )
    assert cluster(embeddings) == [list(range(len(embeddings)))]  # every fit failed: one group
