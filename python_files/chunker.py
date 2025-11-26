"""
Chunker (Phase 1)
-----------------

Responsibilities:
- Convert raw text into safe, evenly sized chunks
- Ensure compatibility with EmbeddingEngine + MemoryManager
- Preserve sentence boundaries where possible
- No circular imports
"""

import re


class Chunker:
    """
    Phase 1 Chunker:
      - Splits text into ~600-character chunks
      - Avoids splitting mid-sentence when possible
      - Removes excessive whitespace
      - Returns a list of clean text chunks
    """

    MAX_CHUNK_SIZE = 600

    def __init__(self):
        pass

    # ---------------------------------------------------------
    #                 PUBLIC ENTRY POINT
    # ---------------------------------------------------------

    def chunk_text(self, text: str):
        """
        Break text into ~600-char chunks with sentence boundaries.
        Returns: list[str]
        """
        if not text or not text.strip():
            return []

        cleaned = self._normalize(text)
        sentences = self._split_sentences(cleaned)

        return self._group(sentences)

    # ---------------------------------------------------------
    #                     NORMALIZATION
    # ---------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """Clean spacing, strip control characters."""
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ---------------------------------------------------------
    #                     SENTENCE SPLITTING
    # ---------------------------------------------------------

    def _split_sentences(self, text: str):
        """
        Simple sentence splitter (not NLP-heavy).
        Splits on: '.', '?', '!'
        Keeps punctuation.
        """
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)

        # Clean & remove empties
        return [s.strip() for s in sentences if s.strip()]

    # ---------------------------------------------------------
    #                     GROUPING INTO CHUNKS
    # ---------------------------------------------------------

    def _group(self, sentences):
        """
        Groups sentences into ~600-char chunks.
        """
        chunks = []
        buffer = ""

        for sentence in sentences:
            # If adding this sentence would exceed max chunk size
            if len(buffer) + len(sentence) + 1 > self.MAX_CHUNK_SIZE:
                if buffer:
                    chunks.append(buffer.strip())
                buffer = sentence
            else:
                if buffer:
                    buffer += " " + sentence
                else:
                    buffer = sentence

        # Add last buffer
        if buffer.strip():
            chunks.append(buffer.strip())

        return chunks
