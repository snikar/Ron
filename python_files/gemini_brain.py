"""Gemini brain for Jeff (Phase 1).

Mirrors the OpenAIBrain API:

- `generate_reply(text)` is the main entry point.
- `chat(text)` is a simple alias.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

import google.generativeai as genai

from core.config import DEFAULT_GEMINI_MODEL, GEMINI_API_KEY
from core.prompts import JEFF_SYSTEM_PROMPT

try:
    from core.spend_guard import log_chat_cost
except Exception:  # pragma: no cover - defensive
    def log_chat_cost(*args, **kwargs):
        return None


class GeminiBrain:
    def __init__(
        self,
        memory,
        write_memory: bool,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "Gemini API key missing. Put it in gemini_key.txt (next to main.py) "
                "or set GEMINI_API_KEY in your environment."
            )

        genai.configure(api_key=GEMINI_API_KEY)

        self.memory = memory
        self.write_memory = write_memory
        self.model_name = model or DEFAULT_GEMINI_MODEL
        self.system_prompt = system_prompt or JEFF_SYSTEM_PROMPT
        self.model = genai.GenerativeModel(self.model_name)

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    def _build_memory_context(self, user_text: str) -> str:
        if not self.memory:
            return ""

        try:
            hits = self.memory.search(user_text, k=3)
        except Exception:
            return ""

        if not hits:
            return ""

        lines = []
        for item in hits:
            snippet = (item.get("text") or "").strip()
            meta = item.get("metadata", {})
            source = meta.get("source", "memory")
            if snippet:
                lines.append(f"- {snippet}  [source: {source}]")
        return "\n".join(lines)

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def generate_reply(self, text: str) -> str:
        memory_context = self._build_memory_context(text)

        parts: List[str] = [self.system_prompt]
        if memory_context:
            parts.append(
                "Relevant long-term memory for this user:\n" + memory_context
            )
        parts.append("User message:\n" + text)

        prompt = "\n\n".join(parts)

        response = self.model.generate_content(prompt)
        message = response.text or ""

        # Gemini's Python client uses usage_metadata for token counts.
        try:
            meta = getattr(response, "usage_metadata", None)
            if meta:
                tokens_in = getattr(meta, "prompt_token_count", 0)
                tokens_out = getattr(meta, "candidates_token_count", 0)
                log_chat_cost(tokens_in, tokens_out, self.model_name)
        except Exception:
            pass

        if self.write_memory and self.memory:
            try:
                self.memory.add_memory_entry(
                    text,
                    source=f"chat:{self.model_name}",
                    write=True,
                )
            except Exception:
                pass

        return message

    def chat(self, text: str) -> str:
        return self.generate_reply(text)
