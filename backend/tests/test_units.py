from raptor_rag.chunking import chunk_blocks
from raptor_rag.db import DB


def test_chunks_pack_blocks_and_track_pages():
    blocks = [(1, "a" * 6), (1, "b" * 6), (2, "c" * 6), (3, "d" * 25)]
    chunks = chunk_blocks(blocks, max_chars=14)
    # "c" can't share a chunk with the first 14-char piece of "d".
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3), (3, 3)]
    assert chunks[0].text == "aaaaaa\nbbbbbb"
    assert all(len(c.text) <= 14 for c in chunks)
    joined = "".join(c.text.replace("\n", "") for c in chunks)
    assert joined == "".join(text for _, text in blocks)  # nothing lost or duplicated


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
