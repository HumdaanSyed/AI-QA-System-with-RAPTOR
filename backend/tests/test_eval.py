from fastapi import FastAPI
from fastapi.testclient import TestClient

from evals.run import output_text, retrieval_metrics
from raptor_rag import rag

from .test_api import create_kb, upload


def test_retrieval_metrics_count_only_passages_from_the_right_pages():
    q = {"doc_id": "a", "page_start": 10, "page_end": 12}
    sources = [
        {"level": 1, "doc_id": "a", "page_start": 1, "page_end": 90},  # summary: never a hit
        {"level": 0, "doc_id": "b", "page_start": 11, "page_end": 11},  # wrong document
        {"level": 0, "doc_id": "a", "page_start": 12, "page_end": 13},  # overlaps: hit at rank 3
    ]
    assert retrieval_metrics(sources, q) == {"recall@5": 1.0, "mrr": 1 / 3}
    assert retrieval_metrics(sources[:2], q) == {"recall@5": 0.0, "mrr": 0.0}


def test_output_text_joins_message_parts():
    body = {
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "message", "content": [{"type": "output_text", "text": "a"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "b"}]},
        ]
    }
    assert output_text(body) == "ab"


def test_retrieve_switches(client: TestClient, app: FastAPI):
    app.state.deps.settings.chunk_chars = 20
    kb_id = create_kb(client)
    upload(client, kb_id, ["Mean median and mode", "Variance and deviation", "Outliers"])
    app.state.worker.wait()
    deps = app.state.deps

    with_tree = rag.retrieve(deps, kb_id, "summary mean variance")
    assert any(s["level"] > 0 for s in with_tree)
    for switches in [
        {"summaries": False},
        {"hybrid": False, "rerank": False, "summaries": False},
    ]:
        found = rag.retrieve(deps, kb_id, "mean variance", **switches)
        assert found and all(s["level"] == 0 for s in found)
