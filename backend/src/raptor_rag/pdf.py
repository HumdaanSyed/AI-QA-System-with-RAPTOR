import re
from collections import Counter
from pathlib import Path

import pymupdf


def validate_pdf(data: bytes, max_bytes: int) -> None:
    """Reject uploads that are too big or not PDFs (checked by content, not filename)."""
    if len(data) > max_bytes:
        raise ValueError(f"File is larger than {max_bytes // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("File is not a PDF.")


def extract_blocks(path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text_block) pairs in reading order; pages are 1-based.

    Running headers/footers are dropped: blocks in the top/bottom 10% of the page whose
    text (digits ignored, so "Page 5 of 7" matches "Page 6 of 7") repeats on most pages.
    """
    blocks: list[tuple[int, str, str | None]] = []  # (page, text, margin key or None)
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc.pages(), start=1):
            height = page.rect.height
            for block in page.get_text("blocks", sort=True):
                text = block[4].strip()
                if text:
                    in_margin = block[3] < 0.1 * height or block[1] > 0.9 * height
                    key = re.sub(r"\d+", "#", text) if in_margin else None
                    blocks.append((page_number, text, key))
        pages = doc.page_count
    counts = Counter(key for _, _, key in set(blocks) if key)  # one count per page
    return [
        (page, text)
        for page, text, key in blocks
        if not (pages > 1 and key and counts[key] > pages / 2)
    ]
