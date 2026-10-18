import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    raptor_enabled INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summary_cache (
    key TEXT PRIMARY KEY,
    summary TEXT NOT NULL
);
"""

# Statuses a job can be in while it is still running.
IN_FLIGHT = ("queued", "parsing", "embedding", "building_tree")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _kb(row: sqlite3.Row) -> dict:
    return {**dict(row), "raptor_enabled": bool(row["raptor_enabled"])}


class DB:
    """SQLite metadata store. Opens a connection per call, so it is safe across threads."""

    def __init__(self, path: Path):
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_kb(self, name: str, raptor_enabled: bool = True) -> dict:
        row = {
            "id": uuid.uuid4().hex,
            "name": name,
            "created_at": _now(),
            "raptor_enabled": raptor_enabled,
        }
        with self._conn() as c:
            c.execute(
                "INSERT INTO knowledge_bases VALUES (:id, :name, :created_at, :raptor_enabled)",
                row,
            )
        return row

    def list_kbs(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM knowledge_bases ORDER BY created_at").fetchall()
        return [_kb(r) for r in rows]

    def get_kb(self, kb_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
        return _kb(row) if row else None

    def rename_kb(self, kb_id: str, name: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE knowledge_bases SET name = ? WHERE id = ?", (name, kb_id))

    def delete_kb(self, kb_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))

    def add_document(self, kb_id: str, filename: str) -> dict:
        row = {
            "id": uuid.uuid4().hex,
            "kb_id": kb_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "error": None,
            "created_at": _now(),
        }
        with self._conn() as c:
            c.execute(
                "INSERT INTO documents VALUES "
                "(:id, :kb_id, :filename, :status, :progress, :error, :created_at)",
                row,
            )
        return row

    def get_document(self, doc_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_documents(self, kb_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at", (kb_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def set_status(
        self, doc_id: str, status: str, progress: int = 0, error: str | None = None
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET status = ?, progress = ?, error = ? WHERE id = ?",
                (status, progress, error, doc_id),
            )

    def delete_document(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def fail_in_flight(self) -> None:
        """Jobs live in memory, so any job still running at startup was lost."""
        marks = ",".join("?" * len(IN_FLIGHT))
        with self._conn() as c:
            c.execute(
                f"UPDATE documents SET status = 'failed', "
                f"error = 'Interrupted by a server restart; re-index to retry.' "
                f"WHERE status IN ({marks})",
                IN_FLIGHT,
            )

    def get_summary(self, key: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT summary FROM summary_cache WHERE key = ?", (key,)).fetchone()
        return row["summary"] if row else None

    def put_summary(self, key: str, summary: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO summary_cache VALUES (?, ?)", (key, summary))
