from core.prompts import system_prompt
from core.spend_guard import log_chat_cost

class LocalBrain:
    """
    Local model stub.
    Phase 1 requirement:
        - memory injection
        - stable interface
        - SpinGuard-compatible (fake token counts)
    """

    MODEL_NAME = "local-llm"

    def __init__(self, memory, write_memory: bool):
        self.memory = memory
        self.write_memory = write_memory

    def chat(self, text: str) -> str:
        # Fake "local model" reply
        reply = f"(Local model simulated reply) {text}"

        # Fake token accounting to keep SpinGuard consistent
        fake_in = len(text.split()) * 2
        fake_out = len(reply.split()) * 2
        log_chat_cost(fake_in, fake_out, "local")

        if self.write_memory:
            self.memory.write(reply)

        return reply
