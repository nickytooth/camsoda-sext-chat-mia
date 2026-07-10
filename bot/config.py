import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# PostgreSQL connection string. Railway injects DATABASE_URL automatically when a
# Postgres service is attached; locally it defaults to a dev container.
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mia"
)
# Mia is a single, always-open persona for all sexting.
PERSONA_FILE_SEXTING = BASE_DIR / os.getenv("SINGLE_PERSONA_FILE", "personas/mia.yaml")

# Authored libraries the "Hear a fantasy" / "Hear a story" cards draw from
# (same for everyone, tracked-as-shared per user in the DB). In library/ so they
# ship in git (unlike the gitignored content/).
FANTASIES_FILE = BASE_DIR / os.getenv("FANTASIES_FILE", "library/fantasies.yaml")
STORIES_FILE = BASE_DIR / os.getenv("STORIES_FILE", "library/stories.yaml")
# Slow background storyline with Tyler — advances by real days since the user's
# first message (see library/tyler_arc.yaml).
TYLER_ARC_FILE = BASE_DIR / os.getenv("TYLER_ARC_FILE", "library/tyler_arc.yaml")

# Where photos the USER uploads are stored on disk and served from (so they
# survive a page reload / history refresh). Lives under data/ (gitignored).
UPLOADS_DIR = BASE_DIR / os.getenv("UPLOADS_DIR", "data/uploads")

# xAI / Grok (NSFW)
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.3")

# Google / Gemini (classification + summarization)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview")
# Used only as the sexting generator fallback when Grok fails — Gemini has its
# safety filters disabled, so it can carry the explicit prompt.
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")

# OpenAI (embeddings only)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Server
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Humanize timing (seconds)
# Sexting batching is a debounce: she replies this many seconds after the
# user's LAST message; every new message resets the countdown.
SEXTING_DEBOUNCE_SECONDS = float(os.getenv("SEXTING_DEBOUNCE_SECONDS", "5"))

# Max seconds to wait for a single LLM generation before treating it as a
# failure and falling back. Prevents a hung provider request from freezing the
# chat (the "typing…" indicator stuck forever).
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))

# Memory settings
# STM_MAX_TURNS counts USER turns (one user message = one turn). Once a user has
# this many turns, the oldest messages are summarised into LTM. Note that
# get_recent_messages fetches up to STM_MAX_TURNS * 2 rows (user + assistant).
STM_MAX_TURNS = int(os.getenv("STM_MAX_TURNS", "18"))
STM_SUMMARIZE_BATCH = int(os.getenv("STM_SUMMARIZE_BATCH", "10"))
LTM_TOP_K = int(os.getenv("LTM_TOP_K", "5"))
LTM_COMPACTION_THRESHOLD = int(os.getenv("LTM_COMPACTION_THRESHOLD", "500"))

# Scoring weights for LTM retrieval
LTM_SIMILARITY_WEIGHT = float(os.getenv("LTM_SIMILARITY_WEIGHT", "0.5"))
LTM_IMPORTANCE_WEIGHT = float(os.getenv("LTM_IMPORTANCE_WEIGHT", "0.3"))
LTM_RECENCY_WEIGHT = float(os.getenv("LTM_RECENCY_WEIGHT", "0.2"))

# Default user ID for single-user demo
DEFAULT_USER_ID = int(os.getenv("DEFAULT_USER_ID", "1"))
