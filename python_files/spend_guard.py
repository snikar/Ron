"""
SpendGuard — Token/Cost Safety Layer for Jeff
---------------------------------------------

This module tracks:
- Daily API spend (OpenAI + Gemini)
- Embedding costs
- Chat completion costs
- Hard daily cap to prevent runaway token usage

Integrated into Phase 1 model wrappers.
"""

import json
from datetime import datetime
from pathlib import Path
from core.config import DATA_DIR

# -----------------------
# SETTINGS
# -----------------------

SPEND_LOG_PATH = DATA_DIR / "spend_log.json"

# Hard daily cap – adjust if needed
DAILY_SPEND_LIMIT = 2.00  # USD per day

# Token price estimates (per 1 token)
# These are approximate. We only need a throttle guard here, not billing precision.
PRICES = {
    # OpenAI embeddings
    "text-embedding-3-small": 0.02 / 1_000_000,
    "text-embedding-3-large": 0.13 / 1_000_000,

    # OpenAI models
    "gpt-4o-mini_in": 0.15 / 1_000_000,
    "gpt-4o-mini_out": 0.60 / 1_000_000,

    # Gemini 1.5
    "gemini-1.5-pro_in": 0.50 / 1_000_000,
    "gemini-1.5-pro_out": 1.50 / 1_000_000,
}

# -----------------------
# Helpers
# -----------------------

def _load_log():
    """Load spend_log.json or create an empty log."""
    if not SPEND_LOG_PATH.exists():
        return {}
    try:
        return json.loads(SPEND_LOG_PATH.read_text())
    except Exception:
        return {}


def _write_log(data: dict):
    """Write the updated spend log back to disk."""
    SPEND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEND_LOG_PATH.write_text(json.dumps(data, indent=2))


def _today_key():
    """Return today's date key for JSON dict."""
    return datetime.now().strftime("%Y-%m-%d")


# -----------------------
# Public API
# -----------------------

def log_embedding_cost(tokens: int, model: str):
    """
    Log cost of an embedding batch (token count).
    """
    if tokens <= 0:
        return 0.0

    cost_per_token = PRICES.get(model, 0.0)
    cost = tokens * cost_per_token

    _apply_cost(cost)
    return cost


def log_chat_cost(tokens_in: int, tokens_out: int, model: str):
    """
    Log cost of a chat call (input + output tokens).
    Model naming convention expected:
        '{model_name}_in'
        '{model_name}_out'
    """
    if tokens_in < 0 or tokens_out < 0:
        return 0.0

    cost_in = tokens_in * PRICES.get(f"{model}_in", 0.0)
    cost_out = tokens_out * PRICES.get(f"{model}_out", 0.0)
    total_cost = cost_in + cost_out

    _apply_cost(total_cost)
    return total_cost


# -----------------------
# Core Logic
# -----------------------

def _apply_cost(cost: float):
    """Add cost to today’s total, enforce the daily cap."""
    data = _load_log()
    today = _today_key()

    previous = data.get(today, 0.0)
    new_total = round(previous + cost, 6)
    data[today] = new_total

    # Write the updated log
    _write_log(data)

    # Cap enforcement
    if new_total > DAILY_SPEND_LIMIT:
        raise RuntimeError(
            f"⚠️ Jeff exceeded the daily API cap: ${new_total:.2f} "
            f"(limit ${DAILY_SPEND_LIMIT:.2f})"
        )

    return new_total
