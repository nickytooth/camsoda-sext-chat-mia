# Mia — AI Girlfriend Sexting Chat

A web-based AI girlfriend **sexting** chat app. Features real-time WebSocket chat, a single always-open persona, two-tier memory, time-of-day awareness, authored fantasy/story cards, and an entitlement-checked visual paywall for private photos and videos. User media uploads and vision analysis remain intentionally unsupported.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16 + React + Tailwind CSS |
| Backend | Python FastAPI + uvicorn |
| Real-time | WebSocket (native) |
| Chat LLM | xAI Grok (generates every reply) |
| Generation fallback | Google Gemini Flash (when Grok fails / refuses) |
| Classifier / Summarizer | Google Gemini Flash |
| Input moderation | xAI Grok |
| Embeddings | OpenAI text-embedding-3-small |
| Database | PostgreSQL (asyncpg) |
| Private media | Cloudflare R2 (S3-compatible signed GET sources) |
| Language | Python 3.13+ / TypeScript |

---

## How It Works

- **Always-open persona** — Grok is the primary reply model, with Gemini as a validated fallback. Mia is a single forward persona (`personas/mia.yaml`); there is no SFW/NSFW model switch.
- **Message batching** — a debounce window collects the user's messages, then they are processed together.
- **Persistent conversation heat** — one sexual processed batch starts a provocative, non-graphic rising phase; a second intensifies it and a third unlocks high/explicit output. Normal batches do not erase rising momentum, while timeout and consent boundaries cool it deterministically.
- **Time-of-day awareness** (Miami timezone) colours her mood and location. Weather is optional (only if `OPENWEATHER_API_KEY` is set).
- **Input moderation** — obvious violations are hard-blocked locally; every other input receives a strict, fail-closed Grok moderation check. The regex soft tier preserves a precise category when the moderator is unavailable.
- **Output validation** — generated replies, fallback drafts, openings, and AI Help suggestions cross deterministic persona/heat/boundary checks plus semantic moderation before display.
- **"Hear a fantasy" / "Hear a story" cards** — a fantasy is generated fresh each time (tailored to the user + current location; the library serves as a style example), while a story is delivered verbatim from the authored library and never repeated until exhausted (`library/`).
- **AI Help** — drafts a suggested next message for the user to send.
- **Idle re-engagement** — if the user goes quiet while still connected, Mia may send one spontaneous follow-up.
- **Visual paywall** — a deterministic planner selects one real catalog item; Mia receives only a safe commerce brief, while PostgreSQL owns offer rotation, demo tokens and permanent entitlements. Locked previews and unlocked photos/videos render directly in chat without URL messages.

---

## Project Structure

```
server/
├── __init__.py
└── app.py                 # FastAPI entrypoint: WebSocket + REST + lifespan

bot/
├── __init__.py
├── chat_engine.py         # Sexting message processor (batching, cards, re-engagement)
├── config.py              # Env vars, paths, tunable constants
├── persona.py             # YAML persona loader → system prompt
├── prompt_builder.py      # Assembles prompt (persona + time + mood + facts + LTM + STM)
├── router.py              # SFW/NSFW keyword classification (feeds mood)
├── moderation.py          # Three-tier input moderation gate
├── mood.py                # Instant, no-LLM mood from the current message
├── engagement.py          # Per-user message tracking
├── content_library.py     # Authored fantasy/story library + "already shared" tracking
├── time_context.py        # Miami time-of-day (+ optional weather) → location/mood
│
├── memory/
│   ├── db.py              # PostgreSQL schema + connection adapter (asyncpg)
│   ├── stm.py             # Short-term memory CRUD (mode-aware)
│   ├── ltm.py             # Long-term memory: store, retrieve (vector search), compact
│   ├── embeddings.py      # OpenAI embedding calls + cosine similarity
│   ├── summarizer.py      # Summarizes STM batches → LTM entries via Gemini
│   └── facts.py           # Structured user facts (name, preferences, etc.)
│
└── providers/
    ├── base.py            # Abstract LLMProvider interface
    ├── grok_provider.py       # Grok (chat + moderation)
    └── gemini_provider.py     # Gemini Flash (classification, summarization, fallback)

personas/
└── mia.yaml              # The active always-open persona

library/
├── fantasies.yaml         # Style examples for the "Hear a fantasy" card
└── stories.yaml           # Authored stories for the "Hear a story" card

.env                       # API keys + config (gitignored)
.env.example               # Template

frontend/
└── app/
    ├── page.tsx           # Main page: chat + sidebar layout
    ├── layout.tsx         # Root layout (dark theme)
    ├── globals.css        # CSS variables, dark color scheme
    ├── api.ts             # API / WebSocket base URLs (NEXT_PUBLIC_API_URL)
    ├── hooks/
    │   └── useChat.ts     # WebSocket hook: messages, typing, cards
    └── components/
        ├── ChatBubble.tsx
        ├── ChatInput.tsx
        ├── StarterCards.tsx
        ├── NameScreen.tsx
        ├── ProfileGallery.tsx
        ├── ProfileSidebar.tsx
        └── TypingIndicator.tsx
```

---

## Setup

### Prerequisites
- Python 3.13+
- Node.js 18+ (Next.js 16)
- A running PostgreSQL instance

### 1. Install

```bash
cd camsoda-ai-chats-sexting

# Backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### 2. Create the database

The tables are created automatically on startup, but **the database itself must
already exist**:

```bash
createdb mia
# or: psql -c "CREATE DATABASE mia;"
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `XAI_API_KEY` | ✅ | xAI API key for Grok (chat + moderation) |
| `XAI_MODEL` | | Grok model id (default `grok-4.3`) |
| `GOOGLE_API_KEY` | ✅ | Google AI key for Gemini (classifier + summarizer + fallback) |
| `GOOGLE_MODEL` | | Gemini classifier model (default `gemini-3-flash-preview`) |
| `GEMINI_FALLBACK_MODEL` | | Fallback generation model (default `gemini-2.5-flash`) |
| `OPENAI_API_KEY` | ✅ | OpenAI API key (embeddings only) |
| `OPENAI_EMBEDDING_MODEL` | | Embedding model (default `text-embedding-3-small`) |
| `DATABASE_URL` | ✅ | PostgreSQL connection string (default: `postgresql://postgres:postgres@localhost:5432/mia`) |
| `SINGLE_PERSONA_FILE` | | Active persona YAML (default: `personas/mia.yaml`) |
| `SERVER_HOST` / `SERVER_PORT` | | Backend host/port (default `0.0.0.0` / `8000`) |
| `DEFAULT_USER_ID` | | Single-user demo id (default `1`) |
| `OPENWEATHER_API_KEY` | optional | Enables Miami weather in her context; omitted → weather is off |
| `SEXTING_DEBOUNCE_SECONDS` | optional | Debounce before she replies, seconds (default `5`) |
| `MEDIA_CATALOG_FILE` | | Runtime catalog (default `library/media_catalog.yaml`) |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | commerce | Cloudflare R2 S3 credentials; when absent, normal chat stays online and media offers/unlocks are disabled |
| `R2_BUCKET_NAME` | commerce | Private bucket containing the catalog's full, preview and poster keys |
| `R2_UPLOAD_ACCESS_KEY_ID` / `R2_UPLOAD_SECRET_ACCESS_KEY` | offline tooling | Separate bucket-scoped read/write credentials used only by the media publish command |
| `R2_SIGNED_PHOTO_TTL_SECONDS` | | Full photo source lifetime (default `600`) |
| `R2_SIGNED_VIDEO_TTL_SECONDS` | | Full video source lifetime (default `3600`) |
| `R2_SIGNED_PREVIEW_TTL_SECONDS` | | Private teaser/poster source lifetime (default `3600`) |
| `COMMERCE_DEV_RESET_ENABLED` | | Enables the destructive dev-only commerce reset (default `false`) |

Frontend (optional, for non-local backends) — create `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Provision the private R2 media

The demo runtime catalog is tracked at `library/media_catalog.yaml`, while the
full media bytes remain exclusively in the private R2 bucket. Never use anything
under `frontend/public` as paid media: those files are directly reachable
without an entitlement. Add entries only for distinct, approved assets.

Create a **private** Cloudflare R2 bucket (no public/custom domain). Asset
preparation and upload are automated; do not make previews or edit the runtime
catalog by hand:

```powershell
python -m pip install -r requirements-media.txt
New-Item -ItemType Directory -Force .private-media\originals
Copy-Item library\media_manifest.example.yaml .private-media\manifest.yaml

# Put approved originals in .private-media\originals and edit only the
# semantic tags/presentation in the private manifest.
python scripts\media_pipeline.py prepare
python scripts\media_pipeline.py publish
```

The pipeline requires `ffmpeg` and `ffprobe` on `PATH` for video normalization
and the paid-vs-public media safety check.

The ignored `.private-media` source directory and `.media-build` output are
never committed. The generated runtime catalog is tracked in `library`. The
pipeline applies image orientation, strips EXIF/GPS and video
metadata, normalizes videos to browser-compatible H.264/AAC MP4, generates
separate strongly downscaled/pixelated/blurred WebP previews and video posters,
computes every checksum/dimension/duration, and uses immutable content-addressed
R2 keys. A video poster is degraded too because the locked card displays it.
Representative public-video frames are fingerprinted in memory so re-encoding
an asset already under `frontend/public` cannot turn it into paid content.

`publish` uses a separate bucket-scoped read/write token, refuses to overwrite
different bytes with conditional creates, streams the stored object back to
verify its real SHA-256, HEAD-verifies the entire resulting catalog, and only
then installs `library/media_catalog.yaml` under a cross-process lock and
baseline-digest check. A failure leaves the previous catalog unchanged. Commit
the generated catalog with the application so Git-based deployments such as
Railway load the same validated inventory. Full media bytes remain private in
R2 and are still protected by entitlement checks.
The backend token should be read-only.

Because the unlocked player reads the short-lived signed URL directly, set an
R2 CORS policy for the exact frontend origins. A local/deployed example is:

```json
[
  {
    "AllowedOrigins": ["http://localhost:3000", "https://your-frontend.example"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Range"],
    "ExposeHeaders": ["Accept-Ranges", "Content-Length", "Content-Range", "ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Set the R2 credentials from `.env.example`. With runtime credentials present,
backend startup HEAD-checks every catalog object and refuses an incomplete
catalog. Without them, text chat stays available but offers/unlocks are disabled.

### 5. Run

```bash
# Terminal 1 — Backend
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## Architecture

### Message Flow

```
User sends a text message (WebSocket)
       │
       ▼
  Input moderation gate  (hard regex → category hint → Grok check on every input)
       │  (flagged → system notice, stop)
       ▼
  Batched (debounce collect window)
  SFW/NSFW keyword classification  →  mood
       │
       ▼
  Add to STM
  Maybe summarize → LTM  /  Maybe compact LTM
  Retrieve LTM (vector search)
       │
       ▼
  Build prompt:
  [persona + time context + texting style]
  [mood + facts + LTM memories]
  [global guardrails]
  [STM conversation turns]
       │
       ▼
  Generate response (Grok → deterministic guard + Grok moderation)
  Invalid/empty/flagged → corrected Gemini fallback → validate again
  Split into 1–3 message bubbles
       │
       ▼
  WebSocket events:
  typing_start → delay → typing_end
  message (× N bubbles, with pauses)
```

### Visual commerce flow

```text
Processed user batch
  -> deterministic media-intent aliases/tags + same-batch refinements
  -> explicit fresh/live capture intent tracked separately from delivery urgency
  -> direct/contextual request: set Heat high and plan an immediate offer
  -> proactive request: batch/heat/snooze checks
  -> semantic-first catalog planner (requested type, alternate type, then nearest)
  -> exact current, exact saved, or explained fallback presentation
     (unlocked excluded; current location is normally a tie-breaker)
  -> one reserved database offer, or a trusted text-only unavailable action
  -> safe COMMERCE BRIEF for Mia (no key, URL, catalog or price)
  -> persisted teaser text + delivered offer
  -> structured WebSocket media card
  -> atomic token debit + permanent entitlement on Unlock
  -> entitlement-checked, short-lived R2 source inside <img>/<video>
```

Raw WebSocket messages do not drive sales timing. One completed debounce batch
increments `engagement_state.total_messages` once, even if it contains hundreds
of rapidly sent messages. A valid direct or contextual visual request raises
durable Heat to `high` for the current turn and can immediately select eligible
inventory; proactive requests continue to respect the pacing rules.
An exact saved item is offered without a capture excuse unless the user
explicitly asked for a fresh/live visual. Type and semantic fallbacks explain
only the real inventory difference, and live blockers never invent a person
who is absent from Mia's current schedule.

### Memory System

- **STM**: recent turns per user (mode-aware storage; sexting only in this build)
- **LTM**: structured memories with vector embeddings, importance scores, recency
- **Facts**: key-value user facts (name, preferences, kinks) — cumulative for list-like keys
- **Summarization**: every `STM_MAX_TURNS` user turns, Gemini summarizes the oldest batch → LTM
- **Compaction**: at `LTM_COMPACTION_THRESHOLD` entries, Gemini deduplicates and merges
- **Retrieval scoring**: similarity (0.5) + importance (0.3) + recency (0.2), top `LTM_TOP_K`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws/chat` | WebSocket | Real-time bidirectional chat; sends Mia's opening on first connect |
| `/api/history/{mode}` | GET | Mixed `text` / `media_offer` chat history (`sexting`) |
| `/api/suggest` | POST | AI Help — draft a reply for the user |
| `/api/demo/wallet` | GET | Demo token balance (initially 1000) |
| `/api/demo/wallet/refill` | POST | Idempotent +1000 demo-token refill |
| `/api/media/offers/{offer_id}/unlock` | POST | Atomic, idempotent offer unlock |
| `/api/media/gallery` | GET | Entitlement-only unlocked gallery |
| `/api/media/{content_id}/access` | GET | Entitlement-check and short-lived full media source |
| `/api/reset` | POST | Reset chat/memory while preserving commerce state |
| `/api/dev/commerce/reset` | POST | Destructive commerce reset; hidden unless explicitly enabled |

---

## Database Schema

Tables are created on startup by `bot/memory/db.py` (`CREATE TABLE IF NOT EXISTS`).

| Table | Purpose |
|-------|---------|
| `messages` | STM — messages per user per mode |
| `memories` | LTM — structured memories with embeddings |
| `user_facts` | Key-value facts about the user |
| `sent_content` | Tracks generated content sent (dedup, e.g. fantasy themes) |
| `shared_content` | Tracks authored library items already shared |
| `engagement_state` | Per-user message counters / timing |
| `media_offers` | Reserved/delivered/cancelled offer rotation and price snapshots |
| `media_entitlements` | Permanent unique `(user_id, content_id)` unlocks |
| `demo_wallets` | Internal-demo token balances |
| `demo_token_transactions` | Idempotent credit/debit ledger |
| `media_tag_affinity` | Soft preference scores learned from unlocks |
| `media_request_confirmations` | Retained legacy table; the runtime no longer reads or writes confirmation state |

---

## Visual-paywall demo boundaries

This build is for the internal team. Its name-derived `user_id` is not
production authentication, tokens are not real payments, and there are no
subscriptions, bundles or age-verification flow. Every catalog asset must be
pre-approved and rights-cleared. A production integration should replace the
`TokenService` implementation and user-identity adapter while retaining the
catalog planner, entitlement checks and private delivery boundary.

---

## Tunable Constants

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `STM_MAX_TURNS` | `config.py` | 18 | Turns before STM → LTM summarization |
| `STM_SUMMARIZE_BATCH` | `config.py` | 10 | Messages summarized per batch |
| `LTM_TOP_K` | `config.py` | 5 | Memories retrieved per message |
| `LTM_COMPACTION_THRESHOLD` | `config.py` | 500 | Entries before compaction |
| `SEXTING_DEBOUNCE_SECONDS` | `config.py` | 5 | Debounce before she replies (seconds) |
| `LLM_TIMEOUT_SECONDS` | `config.py` | 45 | Max wait for one LLM generation |
