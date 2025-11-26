"""
HTML → Chunk Pipeline (Phase 1)
-------------------------------
Turns cleaned HTML blocks into chunked text batches
that MemoryManager can ingest.
"""

from memory.chunker import Chunker


class HTMLChunker:
    """
    Breaks long ChatGPT export text blocks into smaller chunks.
    """

    def __init__(self):
        self.chunker = Chunker()

    def chunk_blocks(self, blocks: list) -> list:
        """
        Input: list of cleaned text blocks.
        Output: list of chunked strings (ready for embeddings).
        """

        final_chunks = []

        for block in blocks:
            if not block or not block.strip():
                continue

            chunks = self.chunker.chunk_text(block)

            for c in chunks:
                if c and c.strip():
                    final_chunks.append(c)

        return final_chunks
