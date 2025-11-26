"""OpenAI brain for Jeff (Phase 1).

This class is intentionally simple:

- Takes a MemoryManager so it can run semantic search for context.
- Uses the default Jeff system prompt unless you override it.
- Exposes `generate_reply(text)` as the primary call.
- Also exposes `chat(text)` as a backwards-compatible alias.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from openai import OpenAI

from core.config import DEFAULT_CHAT_MODEL, OPENAI_API_KEY
from core.prompts import JEFF_SYSTEM_PROMPT

# Cost logging is optional
try:
    from core.spend_guard import log_chat_cost
except Exception:  # pragma: no cover - defensive
    def log_chat_cost(*args, **kwargs):
        return None


class OpenAIBrain:
    def __init__(
        self,
        memory,
        write_memory: bool,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OpenAI API key missing. Put it in openai_key.txt (next to main.py) "
                "or set OPENAI_API_KEY in your environment."
            )

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.memory = memory
        self.write_memory = write_memory
        self.model = model or DEFAULT_CHAT_MODEL
        self.system_prompt = system_prompt or JEFF_SYSTEM_PROMPT

    # -------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------

    def _build_memory_context(self, user_text: str) -> str:
        """Pull a few relevant memory chunks for extra context."""

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
        """Generate a reply from the OpenAI chat model."""

        memory_context = self._build_memory_context(text)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]

        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant long-term memory for this user:\n"
                        f"{memory_context}"
                    ),
                }
            )

        messages.append({"role": "user", "content": text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        message = response.choices[0].message.content or ""

        # Cost logging (best effort)
        try:
            usage = response.usage
            tokens_in = getattr(usage, "input_tokens", 0)
            tokens_out = getattr(usage, "output_tokens", 0)
            log_chat_cost(tokens_in, tokens_out, self.model)
        except Exception:
            pass

        # Optional memory write
        if self.write_memory and self.memory:
            try:
                self.memory.add_memory_entry(
                    text,
                    source=f"chat:{self.model}",
                    write=True,
                )
            except Exception:
                # Don't let memory failures break the chat.
                pass

        return message

    def chat(self, text: str) -> str:
        """Backwards-compatible alias used in some earlier routing code."""
        return self.generate_reply(text)
