from pathlib import Path

import pymupdf


def validate_pdf(data: bytes, max_bytes: int) -> None:
    """Reject uploads that are too big or not PDFs (checked by content, not filename)."""
    if len(data) > max_bytes:
        raise ValueError(f"File is larger than {max_bytes // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("File is not a PDF.")


def extract_blocks(path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text_block) pairs in reading order; pages are 1-based."""
    blocks: list[tuple[int, str]] = []
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc.pages(), start=1):
            for block in page.get_text("blocks", sort=True):
                text = block[4].strip()
                if text:
                    blocks.append((page_number, text))
    return blocks
