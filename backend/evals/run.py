"""Evaluate retrieval and answer quality across configurations.

Run from backend/ (needs OPENAI_API_KEY in .env, Qdrant running and the `ml` extra):

    uv run python -m evals.run setup data/samples              # index PDFs into an "eval" KB
    uv run python -m evals.run questions --specific 30 --broad 20
    uv run python -m evals.run score --limit 10 --judge-model gpt-6-luna   # cheap pilot
    uv run python -m evals.run score --batch                   # full run, Batch API (50% off)

Questions, answers and judgements are written to evals/data/ (git-ignored: they quote
the source documents). The summary table is written to evals/results.md.
"""

import argparse
import json
import random
import shutil
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from qdrant_client import models

from raptor_rag import rag, store
from raptor_rag.api import create_app
from raptor_rag.ingest import Deps, doc_path, ingest_document

HERE = Path(__file__).parent
DATA = HERE / "data"
QUESTIONS = DATA / "questions.jsonl"
STATE = DATA / "state.json"

CONFIGS: dict[str, dict] = {
    "Dense only (baseline)": {"hybrid": False, "rerank": False, "summaries": False},
    "Hybrid": {"hybrid": True, "rerank": False, "summaries": False},
    "Hybrid + rerank": {"hybrid": True, "rerank": True, "summaries": False},
    "Hybrid + rerank + RAPTOR": {"hybrid": True, "rerank": True, "summaries": True},
}

QUESTION_INSTRUCTIONS = """You write evaluation questions for a document question-answering system.

Given passages from a document, write ONE question a reader might ask and its reference answer.
- The answer must be fully supported by the passages; do not use outside knowledge.
- Ask it the way a real user would, as if they had never seen these passages: never mention
  "the passage", "the text", "the document", "this section" or page numbers.
- {kind}
Return JSON only."""

SPECIFIC = "Ask about one specific fact, definition, number or step stated in the passage."
BROAD = (
    "Ask a broad question whose answer needs information spread across several of the "
    "passages (an overview, comparison or summary of the section), not a single fact."
)

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTIONS = """You grade answers from a document question-answering system.

You get the question, a reference answer, the numbered sources the system retrieved, and the
system's answer (which cites sources as [n]). Count carefully:
- reference_points: the distinct facts in the reference answer.
- reference_points_in_sources: how many of those facts are stated in the sources.
- claims: the distinct factual claims in the system's answer.
- supported_claims: how many of those claims are supported by the sources (not by your own
  knowledge).
- citations: the number of [n] citation markers in the answer.
- correct_citations: how many of those point to a source that supports the sentence they follow.
- relevance: 1-5, how completely and correctly the answer addresses the question compared with
  the reference (1 = wrong or refuses although the sources contain the answer, 5 = complete).
Return JSON only."""

JUDGE_FIELDS = [
    "reference_points",
    "reference_points_in_sources",
    "claims",
    "supported_claims",
    "citations",
    "correct_citations",
    "relevance",
]
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "integer"} for f in JUDGE_FIELDS},
    "required": JUDGE_FIELDS,
    "additionalProperties": False,
}


def deps() -> Deps:
    return create_app().state.deps


def json_format(name: str, schema: dict) -> dict:
    return {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}


def output_text(body: dict) -> str:
    return "".join(
        part["text"]
        for item in body["output"]
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )


def run_requests(client: OpenAI, requests: dict[str, dict], batch: bool) -> dict[str, str]:
    """Send Responses API requests ({id: body}) and return {id: output text}."""
    if not batch:

        def one(body: dict) -> str:
            return output_text(client.responses.create(**body).model_dump())

        with ThreadPoolExecutor(8) as pool:
            return dict(zip(requests, pool.map(one, requests.values()), strict=True))

    lines = "\n".join(
        json.dumps({"custom_id": k, "method": "POST", "url": "/v1/responses", "body": v})
        for k, v in requests.items()
    )
    file = client.files.create(file=("batch.jsonl", lines.encode()), purpose="batch")
    job = client.batches.create(
        input_file_id=file.id, endpoint="/v1/responses", completion_window="24h"
    )
    while job.status not in ("completed", "failed", "expired", "cancelled"):
        print(f"  batch {job.id}: {job.status} {job.request_counts}", flush=True)
        time.sleep(30)
        job = client.batches.retrieve(job.id)
    if job.status != "completed" or not job.output_file_id:
        raise RuntimeError(f"Batch {job.id} ended as {job.status}")
    rows = [json.loads(r) for r in client.files.content(job.output_file_id).text.splitlines()]
    failed = [r["custom_id"] for r in rows if r.get("error") or r["response"]["status_code"] != 200]
    if failed:
        raise RuntimeError(f"{len(failed)} batch requests failed, e.g. {failed[:3]}")
    return {r["custom_id"]: output_text(r["response"]["body"]) for r in rows}


def setup(pdf_dir: Path) -> None:
    """Create an "eval" knowledge base (RAPTOR on) and index every PDF in pdf_dir."""
    d = deps()
    kb = d.db.create_kb("eval", raptor_enabled=True)
    store.create_collection(d.qdrant, kb["id"], d.embedder.dim)
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc = d.db.add_document(kb["id"], pdf.name)
        path = doc_path(d.settings, kb["id"], doc["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, path)
        started = time.time()
        ingest_document(d, kb["id"], doc["id"])
        status = d.db.get_document(doc["id"])
        print(f"{pdf.name}: {status and status['status']} in {time.time() - started:.0f}s")
    DATA.mkdir(exist_ok=True)
    STATE.write_text(json.dumps({"kb_id": kb["id"]}))
    print(f"Knowledge base {kb['id']} saved to {STATE}")


def leaf_chunks(d: Deps, kb_id: str) -> list[dict]:
    points, offset = [], None
    while True:
        page, offset = d.qdrant.scroll(
            store.collection_name(kb_id),
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="level", match=models.MatchValue(value=0))]
            ),
            limit=500,
            offset=offset,
        )
        points += [p.payload for p in page if p.payload]
        if offset is None:
            break
    return sorted(points, key=lambda p: (p["doc_id"], p["page_start"]))


def questions(n_specific: int, n_broad: int, model: str, seed: int = 7) -> None:
    """Draft questions from random passages (specific) and passage windows (broad)."""
    d = deps()
    kb_id = json.loads(STATE.read_text())["kb_id"]
    chunks = leaf_chunks(d, kb_id)
    rng = random.Random(seed)
    picks: list[tuple[str, list[dict]]] = [
        ("specific", [c]) for c in rng.sample(chunks, n_specific)
    ]
    for _ in range(n_broad):  # 6 consecutive chunks from the same document
        start = rng.randrange(len(chunks) - 6)
        window = [c for c in chunks[start : start + 6] if c["doc_id"] == chunks[start]["doc_id"]]
        picks.append(("broad", window))

    requests = {
        str(i): {
            "model": model,
            "instructions": QUESTION_INSTRUCTIONS.format(
                kind=SPECIFIC if kind == "specific" else BROAD
            ),
            "input": "\n\n".join(f"<passage>\n{c['text']}\n</passage>" for c in window),
            "text": json_format("question", QUESTION_SCHEMA),
        }
        for i, (kind, window) in enumerate(picks)
    }
    out = run_requests(OpenAI(api_key=d.settings.openai_api_key), requests, batch=False)
    DATA.mkdir(exist_ok=True)
    with QUESTIONS.open("w") as f:
        for i, (kind, window) in enumerate(picks):
            qa = json.loads(out[str(i)])
            row = {
                "id": i,
                "type": kind,
                **qa,
                "doc_id": window[0]["doc_id"],
                "doc_name": window[0]["doc_name"],
                "page_start": min(c["page_start"] for c in window),
                "page_end": max(c["page_end"] for c in window),
            }
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(picks)} questions to {QUESTIONS} — review them before scoring.")


def is_hit(source: dict, q: dict) -> bool:
    """A retrieved passage is relevant if it comes from the question's pages."""
    return (
        source["level"] == 0
        and source["doc_id"] == q["doc_id"]
        and source["page_start"] <= q["page_end"]
        and source["page_end"] >= q["page_start"]
    )


def retrieval_metrics(sources: list[dict], q: dict, k: int = 5) -> dict:
    ranks = [i for i, s in enumerate(sources, start=1) if is_hit(s, q)]
    return {"recall@5": float(bool(ranks) and ranks[0] <= k), "mrr": 1 / ranks[0] if ranks else 0.0}


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def score(limit: int | None, answer_model: str, judge_model: str, batch: bool) -> None:
    d = deps()
    kb_id = json.loads(STATE.read_text())["kb_id"]
    qs = [json.loads(line) for line in QUESTIONS.read_text().splitlines()][:limit]
    client = OpenAI(api_key=d.settings.openai_api_key)

    # 1. Retrieve for every config (local models, no API cost).
    runs: dict[tuple[str, int], dict] = {}
    for name, switches in CONFIGS.items():
        for q in qs:
            sources = rag.retrieve(d, kb_id, q["question"], **switches)
            runs[(name, q["id"])] = {"sources": sources, **retrieval_metrics(sources, q)}
        print(f"retrieved: {name}")

    # 2. Answer with the app's own prompt.
    def key(name: str, qid: int) -> str:
        return f"{list(CONFIGS).index(name)}-{qid}"

    by_key = {key(n, qid): (n, qid) for n, qid in runs}
    q_by_id = {q["id"]: q for q in qs}
    answers = run_requests(
        client,
        {
            k: {
                "model": answer_model,
                "instructions": rag.INSTRUCTIONS,
                "input": rag.build_prompt(q_by_id[qid]["question"], runs[(n, qid)]["sources"]),
                "reasoning": {"effort": "low"},
            }
            for k, (n, qid) in by_key.items()
            if runs[(n, qid)]["sources"]
        },
        batch,
    )
    for k, (n, qid) in by_key.items():
        runs[(n, qid)]["answer"] = answers.get(k, rag.NOT_FOUND)
    print("answered")

    # 3. Judge.
    judgements = run_requests(
        client,
        {
            k: {
                "model": judge_model,
                "instructions": JUDGE_INSTRUCTIONS,
                "input": (
                    f"<question>\n{q_by_id[qid]['question']}\n</question>\n\n"
                    f"<reference_answer>\n{q_by_id[qid]['answer']}\n</reference_answer>\n\n"
                    f"{rag.format_sources(runs[(n, qid)]['sources'])}\n\n"
                    f"<answer>\n{runs[(n, qid)]['answer']}\n</answer>"
                ),
                "text": json_format("judgement", JUDGE_SCHEMA),
            }
            for k, (n, qid) in by_key.items()
        },
        batch,
    )
    for k, (n, qid) in by_key.items():
        j = json.loads(judgements[k])
        runs[(n, qid)].update(
            judge=j,
            context_recall=ratio(j["reference_points_in_sources"], j["reference_points"]),
            faithfulness=ratio(j["supported_claims"], j["claims"]),
            citation_accuracy=ratio(j["correct_citations"], j["citations"]),
            relevance=j["relevance"],
        )

    DATA.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (DATA / f"run-{stamp}.json").write_text(
        json.dumps(
            [{"config": n, "question_id": qid, **r} for (n, qid), r in runs.items()], indent=1
        )
    )
    table = results_table(runs, qs, answer_model, judge_model)
    (HERE / "results.md").write_text(table)
    print(table)


METRICS = {
    "recall@5": "Recall@5",
    "mrr": "MRR",
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "citation_accuracy": "Citation accuracy",
    "relevance": "Answer relevance (1-5)",
}


def results_table(runs: dict, qs: list[dict], answer_model: str, judge_model: str) -> str:
    def mean(name: str, metric: str, keep: Callable[[dict], bool]) -> str:
        values = [
            runs[(name, q["id"])][metric]
            for q in qs
            if keep(q) and runs[(name, q["id"])][metric] is not None
        ]
        if not values:
            return "–"
        avg = statistics.mean(values)
        return f"{avg:.2f}" if metric == "relevance" else f"{avg:.1%}"

    lines = [
        "# Evaluation results",
        "",
        f"{len(qs)} questions ({sum(q['type'] == 'specific' for q in qs)} specific, "
        f"{sum(q['type'] == 'broad' for q in qs)} broad). Answers: `{answer_model}`, "
        f"judge: `{judge_model}`.",
        "Recall@5 and MRR count retrieved passages from the question's source pages "
        "(RAPTOR summaries never count as hits). Context recall, faithfulness, citation accuracy "
        "and relevance are LLM-judged.",
    ]
    for title, keep in [
        ("All questions", lambda q: True),
        ("Specific questions", lambda q: q["type"] == "specific"),
        ("Broad questions", lambda q: q["type"] == "broad"),
    ]:
        lines += ["", f"## {title}", "", "| Config | " + " | ".join(METRICS.values()) + " |"]
        lines.append("|---" * (len(METRICS) + 1) + "|")
        for name in CONFIGS:
            row = [mean(name, m, keep) for m in METRICS]
            lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("pdf_dir", type=Path)
    p = sub.add_parser("questions")
    p.add_argument("--specific", type=int, default=30)
    p.add_argument("--broad", type=int, default=20)
    p.add_argument("--model", default="gpt-6-luna")
    p = sub.add_parser("score")
    p.add_argument("--limit", type=int)
    p.add_argument("--answer-model", default="gpt-6.1-sol")
    p.add_argument("--judge-model", default="gpt-6.1-sol")
    p.add_argument("--batch", action="store_true", help="use the Batch API (50%% cheaper)")
    args = parser.parse_args()

    if args.cmd == "setup":
        setup(args.pdf_dir)
    elif args.cmd == "questions":
        questions(args.specific, args.broad, args.model)
    else:
        score(args.limit, args.answer_model, args.judge_model, args.batch)


if __name__ == "__main__":
    main()
