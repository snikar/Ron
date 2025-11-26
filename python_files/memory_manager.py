import json
from datetime import datetime
from pathlib import Path

from memory.embeddings import EmbeddingEngine
from memory.chunker import Chunker
from core.config import MEMORY_JSON, MEMORY_BACKUP_JSON, ALLOW_MEMORY_WRITE


class MemoryManager:
    """
    Phase 1 Memory Manager for Jeff
    --------------------------------
    Responsibilities:
      - Store long-term memory entries in JSON
      - Timestamped writes (Option B: append with metadata)
      - Semantic search via EmbeddingEngine
      - Keyword search fallback
      - One instance ONLY — all brains receive it by DI
    """

    def __init__(self, allow_write: bool = True):
        self.allow_write = allow_write
        self.memory_path = MEMORY_JSON
        self.backup_path = MEMORY_BACKUP_JSON

        self.embeddings = EmbeddingEngine()
        self.chunker = Chunker()

        self.memory = self._load_memory()

    # ---------------------------------------------------------
    #                   LOADING / SAVING
    # ---------------------------------------------------------

    def _load_memory(self):
        """Load memory.json or return empty list."""
        if self.memory_path.exists():
            try:
                data = json.loads(self.memory_path.read_text())
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return []

    def _save_memory(self):
        """Persist memory.json and a timestamped backup."""
        self.memory_path.write_text(
            json.dumps(self.memory, indent=2),
            encoding="utf-8"
        )
        self.backup_path.write_text(
            json.dumps(self.memory, indent=2),
            encoding="utf-8"
        )

    # ---------------------------------------------------------
    #                        ADD MEMORY
    # ---------------------------------------------------------

    def add(self, text: str, source: str = "chat", metadata: dict = None):
        """
        Add a memory entry (if allowed).
        Memory entries look like:

        {
          "timestamp": "...",
          "text": "...",
          "source": "chat",
          "metadata": {...},
          "chunks": [chunk metadata],
        }
        """

        if not self.allow_write or not ALLOW_MEMORY_WRITE:
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = metadata or {}

        # 1. Chunk the text
        chunks = self.chunker.chunk_text(text)

        # 2. Embed and add chunks to FAISS
        chunk_records = []
        for c in chunks:
            meta = {
                "timestamp": timestamp,
                "source": source,
                **metadata
            }
            self.embeddings.add_chunk(c, meta)
            chunk_records.append({
                "text": c,
                "metadata": meta
            })

        # 3. Store raw memory entry
        entry = {
            "timestamp": timestamp,
            "text": text,
            "source": source,
            "metadata": metadata,
            "chunks": chunk_records
        }

        self.memory.append(entry)
        self._save_memory()
        return True

    # ---------------------------------------------------------
    #                        RECALL
    # ---------------------------------------------------------

    def search(self, query: str, k: int = 5):
        """
        Semantic search via FAISS + keyword fallback.
        Returns:
        [
          {"text": ..., "metadata": ...},
          ...
        ]
        """

        # Semantic search first
        results = self.embeddings.search(query, k)

        # If semantic search returns nothing, fallback to dumb keyword
        if not results:
            keyword_hits = []
            for entry in self.memory:
                if query.lower() in entry["text"].lower():
                    keyword_hits.append(entry)
            return keyword_hits[:k]

        return results

    # ---------------------------------------------------------
    #                HELPER: TOGGLE MEMORY WRITE
    # ---------------------------------------------------------

    def set_write_mode(self, allow: bool):
        """Turn persistent memory ON/OFF from UI."""
        self.allow_write = allow

    # ---------------------------------------------------------
    #                   HELPER: GET LATEST
    # ---------------------------------------------------------

    def latest(self, n=5):
        """Return latest N memory entries."""
        return self.memory[-n:]
