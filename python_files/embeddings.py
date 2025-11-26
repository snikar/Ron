import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from core.config import (
    TEXT_CHUNKS_DIR,
    FAISS_INDEX,
    DEFAULT_EMBED_MODEL,
    OPENAI_API_KEY,
)
from core.spend_guard import log_embedding_cost


class EmbeddingEngine:
    """
    Handles:
      - Generating text embeddings (OpenAI new SDK)
      - Storing chunks to text_chunks/
      - Maintaining FAISS index
      - Mapping vector IDs to chunk metadata
    """

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Put your key in 'openai_key.txt' in the jeff folder "
                "or set the OPENAI_API_KEY environment variable."
            )

        # Explicitly pass api_key so we don't depend on env magic
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.index, self.id_map = self._load_index()

    # -------------------------------------------------------
    #                 INDEX LOADING / SAVING
    # -------------------------------------------------------

    def _load_index(self):
        """
        Load FAISS index + id_map from disk.
        If not present, create fresh ones.
        """
        TEXT_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

        # 1536 dims for text-embedding-3-small
        index = faiss.IndexFlatL2(1536)

        map_path = FAISS_INDEX.with_suffix(".map.json")
        if FAISS_INDEX.exists() and map_path.exists():
            try:
                index = faiss.read_index(str(FAISS_INDEX))
                id_map = json.loads(map_path.read_text(encoding="utf-8"))
                return index, id_map
            except Exception:
                # If anything is corrupted, start from a clean slate
                pass

        return index, {}

    def _save_index(self) -> None:
        """Persist FAISS index + ID map to disk."""
        faiss.write_index(self.index, str(FAISS_INDEX))
        map_path = FAISS_INDEX.with_suffix(".map.json")
        map_path.write_text(json.dumps(self.id_map, indent=2), encoding="utf-8")

    # -------------------------------------------------------
    #                 EMBEDDING GENERATION
    # -------------------------------------------------------

    def embed_text(self, text: str, model: str | None = None) -> list[float]:
        """
        Generate an embedding for a given string.
        """
        model = model or DEFAULT_EMBED_MODEL

        try:
            response = self.client.embeddings.create(
                model=model,
                input=text,
            )
            embedding = response.data[0].embedding

            # New SDK includes usage info
            usage = getattr(response, "usage", None) or {}
            tokens_in = usage.get("prompt_tokens", 0)
            log_embedding_cost(tokens_in, model)

            return embedding
        except Exception as e:
            raise RuntimeError(f"[Embedding Error] {e}") from e

    # -------------------------------------------------------
    #                 CHUNK STORAGE
    # -------------------------------------------------------

    def add_chunk(self, text: str, metadata: dict) -> None:
        """
        Store text chunk on disk, embed it, add to FAISS index.
        metadata = {"source": ..., "timestamp": ..., etc}
        """
        chunk_id = f"chunk_{len(self.id_map) + 1}"
        chunk_path = TEXT_CHUNKS_DIR / f"{chunk_id}.txt"
        chunk_path.write_text(text, encoding="utf-8")

        embedding = self.embed_text(text)
        vector = self._to_vector(embedding)

        self.index.add(vector)

        self.id_map[len(self.id_map)] = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
        }

        self._save_index()

    # -------------------------------------------------------
    #                 SEARCH
    # -------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Semantic search over stored chunks using FAISS.
        """
        if not self.id_map:
            return []

        embedding = self.embed_text(query)
        vector = self._to_vector(embedding)

        distances, indices = self.index.search(vector, k)

        results: list[dict] = []
        for idx in indices[0]:
            if idx in self.id_map:
                results.append(self.id_map[idx])
        return results

    # -------------------------------------------------------
    #                 UTILS
    # -------------------------------------------------------

    @staticmethod
    def _to_vector(embedding: list[float]) -> np.ndarray:
        """
        Convert Python list → FAISS-friendly shape (1, dim).
        """
        return np.array(embedding, dtype="float32").reshape(1, -1)
