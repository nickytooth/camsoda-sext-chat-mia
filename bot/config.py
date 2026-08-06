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
# Slow background storyline with Tyler — advances by persisted user-message
# count (see library/tyler_arc.yaml).
TYLER_ARC_FILE = BASE_DIR / os.getenv("TYLER_ARC_FILE", "library/tyler_arc.yaml")

# Visual content commerce. The catalog contains metadata and private object
# keys only; object delivery is handled separately and must entitlement-check
# before issuing a short-lived URL for the full asset.
_configured_media_catalog = os.getenv("MEDIA_CATALOG_FILE", "library/media_catalog.yaml")
MEDIA_CATALOG_FILE = BASE_DIR / _configured_media_catalog
MEDIA_PHOTO_PRICE_TOKENS = int(os.getenv("MEDIA_PHOTO_PRICE_TOKENS", "5"))
MEDIA_VIDEO_PRICE_TOKENS = int(os.getenv("MEDIA_VIDEO_PRICE_TOKENS", "10"))
DEMO_WALLET_INITIAL_TOKENS = int(os.getenv("DEMO_WALLET_INITIAL_TOKENS", "1000"))
DEMO_WALLET_REFILL_TOKENS = int(os.getenv("DEMO_WALLET_REFILL_TOKENS", "1000"))

# Sales pacing is counted in processed debounce batches (engagement_state's
# total_messages), never in the raw ingestion counter.
MEDIA_PROACTIVE_MIN_BATCHES = int(os.getenv("MEDIA_PROACTIVE_MIN_BATCHES", "8"))
MEDIA_PROACTIVE_COOLDOWN_BATCHES = int(
    os.getenv("MEDIA_PROACTIVE_COOLDOWN_BATCHES", "8")
)
MEDIA_REPEAT_COOLDOWN_BATCHES = int(os.getenv("MEDIA_REPEAT_COOLDOWN_BATCHES", "8"))
MEDIA_SOFT_DECLINE_MIN_BATCHES = int(
    os.getenv("MEDIA_SOFT_DECLINE_MIN_BATCHES", "30")
)
MEDIA_SOFT_DECLINE_MAX_BATCHES = int(
    os.getenv("MEDIA_SOFT_DECLINE_MAX_BATCHES", "40")
)
MEDIA_HARD_DECLINE_SNOOZE_BATCHES = int(
    os.getenv("MEDIA_HARD_DECLINE_SNOOZE_BATCHES", "100")
)

# Cloudflare R2 credentials. Full objects belong in a private bucket. These
# values are intentionally empty by default so local development can use a
# delivery adapter without accidentally exposing a bucket.
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "")
R2_SIGNED_PHOTO_TTL_SECONDS = int(
    os.getenv("R2_SIGNED_PHOTO_TTL_SECONDS", "600")
)
R2_SIGNED_VIDEO_TTL_SECONDS = int(
    os.getenv("R2_SIGNED_VIDEO_TTL_SECONDS", "3600")
)
R2_SIGNED_PREVIEW_TTL_SECONDS = int(
    os.getenv("R2_SIGNED_PREVIEW_TTL_SECONDS", "3600")
)
COMMERCE_DEV_RESET_ENABLED = os.getenv(
    "COMMERCE_DEV_RESET_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

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

# A rising/high sexual session cools back to the normal flirty register when
# the user has not sent another sexual processed batch for this long.  Normal
# chat batches do not decrement the persistent rising state on their own.
HEAT_SESSION_TIMEOUT_SECONDS = int(os.getenv("HEAT_SESSION_TIMEOUT_SECONDS", "3600"))

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
