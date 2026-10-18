from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int


def chunk_blocks(blocks: list[tuple[int, str]], max_chars: int) -> list[Chunk]:
    """Pack text blocks (paragraphs) into chunks of up to max_chars, keeping page ranges.

    Blocks longer than max_chars are cut into max_chars pieces.
    """
    chunks: list[Chunk] = []
    parts: list[str] = []
    pages: list[int] = []

    def flush() -> None:
        if parts:
            chunks.append(Chunk("\n".join(parts), pages[0], pages[-1]))
            parts.clear()
            pages.clear()

    for page, text in blocks:
        for i in range(0, len(text), max_chars):
            piece = text[i : i + max_chars]
            if sum(len(p) + 1 for p in parts) + len(piece) > max_chars:
                flush()
            parts.append(piece)
            pages.append(page)
    flush()
    return chunks
