import pymupdf

from raptor_rag.chunking import chunk_blocks
from raptor_rag.db import DB
from raptor_rag.pdf import extract_blocks


def test_chunks_pack_blocks_and_track_pages():
    blocks = [(1, "a" * 6), (1, "b" * 6), (2, "c" * 6), (3, "d" * 25)]
    chunks = chunk_blocks(blocks, max_chars=14)
    # "c" can't share a chunk with the first 14-char piece of "d".
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3), (3, 3)]
    assert chunks[0].text == "aaaaaa\nbbbbbb"
    assert all(len(c.text) <= 14 for c in chunks)
    joined = "".join(c.text.replace("\n", "") for c in chunks)
    assert joined == "".join(text for _, text in blocks)  # nothing lost or duplicated


def test_extract_blocks_drops_running_headers_and_footers(tmp_path):
    doc = pymupdf.open()
    for n in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 40), "Acme, Inc. | Handbook")
        page.insert_text((72, 300), f"Body text {n}")
        page.insert_text((72, 770), f"Page {n} of 3")
    doc[0].insert_text((300, 40), "Draft")  # margin text on one page only is kept
    path = tmp_path / "doc.pdf"
    doc.save(path)
    assert extract_blocks(path) == [
        (1, "Draft"),
        (1, "Body text 1"),
        (2, "Body text 2"),
        (3, "Body text 3"),
    ]

    single = pymupdf.open()
    single.new_page().insert_text((72, 40), "Acme, Inc. | Handbook")
    single.save(tmp_path / "one.pdf")
    assert extract_blocks(tmp_path / "one.pdf") == [(1, "Acme, Inc. | Handbook")]


def test_fail_in_flight_marks_only_running_jobs(tmp_path):
    db = DB(tmp_path / "t.db")
    kb = db.create_kb("kb")
    running = db.add_document(kb["id"], "a.pdf")["id"]
    done = db.add_document(kb["id"], "b.pdf")["id"]
    db.set_status(running, "embedding", 40)
    db.set_status(done, "ready", 100)

    db.fail_in_flight()
    assert db.get_document(running)["status"] == "failed"  # type: ignore[index]
    assert db.get_document(done)["status"] == "ready"  # type: ignore[index]
