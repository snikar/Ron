"""
Routing Engine (Phase 1)
------------------------
Selects which model "brain" Jeff should use.

Supports:
 - Manual override
 - Auto-route with fallback
 - Clean dependency injection
 - Zero circular imports

Brains:
 - OpenAIBrain
 - GeminiBrain
 - LocalBrain
"""

from core.config import (
    DEFAULT_CHAT_MODEL,
    OPENAI_MODELS,
    GEMINI_MODELS,
    LOCAL_MODELS,
)
from models.openai_brain import OpenAIBrain
from models.gemini_brain import GeminiBrain
from models.local_brain import LocalBrain


class Router:
    """
    Phase-1 Model Router

    Returns an instantiated brain object based on:
      - user-selected model (manual)
      - or auto-routing rules
    """

    def __init__(self, memory_manager, write_memory: bool):
        self.memory = memory_manager
        self.write_memory = write_memory

    # -------------------------------------------------------
    #                PUBLIC ENTRY POINT
    # -------------------------------------------------------

    def get_brain(self, model_name: str = None):
        """
        model_name:
            - If provided → manual override
            - If None → auto-route using DEFAULT_CHAT_MODEL
        """

        if model_name:
            return self._manual_route(model_name)

        # Auto-routing path
        return self._auto_route()

    # -------------------------------------------------------
    #                     MANUAL ROUTE
    # -------------------------------------------------------

    def _manual_route(self, model: str):
        model_lower = model.lower()

        if model_lower in OPENAI_MODELS:
            return OpenAIBrain(self.memory, self.write_memory)

        if model_lower in GEMINI_MODELS:
            return GeminiBrain(self.memory, self.write_memory)

        if model_lower in LOCAL_MODELS:
            return LocalBrain(self.memory, self.write_memory)

        # Default fallback
        return LocalBrain(self.memory, self.write_memory)

    # -------------------------------------------------------
    #                     AUTO ROUTE
    # -------------------------------------------------------

    def _auto_route(self):
        """
        Auto-route rules:
        1. Try DEFAULT_CHAT_MODEL (OpenAI)
        2. If fails, try Gemini
        3. If all fail, fallback to LocalBrain
        """

        model = DEFAULT_CHAT_MODEL.lower()

        # Default → usually OpenAI
        if model in OPENAI_MODELS:
            try:
                return OpenAIBrain(self.memory, self.write_memory)
            except Exception:
                pass

        # Try gemini fallback
        if GEMINI_MODELS:
            try:
                return GeminiBrain(self.memory, self.write_memory)
            except Exception:
                pass

        # Final fallback
        return LocalBrain(self.memory, self.write_memory)
