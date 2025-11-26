from pathlib import Path
import os

# ---------------------------------------------------------
# ROOT & APP
# ---------------------------------------------------------

# Project root: /.../Jeff_Phase1/jeff
ROOT = Path(__file__).resolve().parent.parent

APP_TITLE = "Jeff-AI (Phase 1)"

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

GEMINI_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

LOCAL_MODELS = [
    "mistral",
    "phi-3-mini",
    "llama",
]

# Default chat brain (you picked 4.1)
DEFAULT_CHAT_MODEL = "gpt-4.1"

# Default Gemini model
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"

# Default embedding model
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------
# API KEYS (env first, then key files)
# ---------------------------------------------------------

def _read_key(path: Path) -> str:
    """Read a key file if it exists, otherwise return empty string."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

OPENAI_KEY_PATH = ROOT / "openai_key.txt"
GEMINI_KEY_PATH = ROOT / "gemini_key.txt"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or _read_key(OPENAI_KEY_PATH)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or _read_key(GEMINI_KEY_PATH)

# ---------------------------------------------------------
# DATA / MEMORY / INDEX
# ---------------------------------------------------------

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEXT_CHUNKS_DIR = DATA_DIR / "text_chunks"
TEXT_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

MEMORY_JSON = DATA_DIR / "memory.json"
MEMORY_BACKUP_JSON = DATA_DIR / "memory_backup.json"

FAISS_INDEX = DATA_DIR / "vector_index.faiss"

# Global default for memory writes
ALLOW_MEMORY_WRITE = True  # Streamlit toggle can override per-session

# ---------------------------------------------------------
# LOGS
# ---------------------------------------------------------

LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS_LOG = LOGS_DIR / "actions.log"
ERRORS_LOG = LOGS_DIR / "errors.log"
UPDATES_LOG = LOGS_DIR / "updates.log"

# Spend guard log (for API cost tracking)
SPEND_LOG_JSON = DATA_DIR / "spend_log.json"

