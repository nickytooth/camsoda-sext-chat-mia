# Deploying to Railway

The app is **3 Railway services in one project**:

1. **PostgreSQL** (Railway plugin) — stores all data; browse it in the service's **Data** tab.
2. **Backend** (this repo root) — FastAPI + WebSocket.
3. **Frontend** (`frontend/`) — Next.js.

> The backend keeps in-memory state (message batching, idle timers). **Run a single
> instance** — do not enable horizontal scaling/replicas.

---

## 1. PostgreSQL
Railway → **New → Database → PostgreSQL**. It exposes `DATABASE_URL` automatically.

## 2. Backend service
- **New → GitHub repo** (or deploy this repo). **Root directory: `/`** (repo root).
- Start command comes from the `Procfile`:
  `uvicorn server.app:app --host 0.0.0.0 --port $PORT`
- **Variables** (Settings → Variables):
  - `DATABASE_URL` → reference the Postgres service: `${{Postgres.DATABASE_URL}}`
  - `XAI_API_KEY`, `XAI_MODEL` (e.g. `grok-4.3`)
  - `GOOGLE_API_KEY`, `GOOGLE_MODEL` (e.g. `gemini-3-flash-preview`), `GEMINI_FALLBACK_MODEL` (e.g. `gemini-2.5-flash`)
  - `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL` (e.g. `text-embedding-3-small`)
  - `OPENWEATHER_API_KEY` (optional)
  - private visual commerce: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
    `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`
  - signed-source lifetimes: `R2_SIGNED_PHOTO_TTL_SECONDS=600`,
    `R2_SIGNED_VIDEO_TTL_SECONDS=3600`, `R2_SIGNED_PREVIEW_TTL_SECONDS=3600`
  - keep `COMMERCE_DEV_RESET_ENABLED=false` outside disposable internal demos
  - optional tuning: `SEXTING_DEBOUNCE_SECONDS`, `DEFAULT_USER_ID`
  - `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` are **not required** (Anthropic is not used at runtime)
- Generate a public domain (Settings → Networking). Note it, e.g. `https://mia-backend.up.railway.app`.
- Tables are created automatically on first boot (`init_db`).

### Private Cloudflare R2 media

The checked-in runtime catalog is intentionally empty. Populate it only with
distinct private-only assets; never reuse files from `frontend/public`, because
those are served without an entitlement.

Keep the bucket private. Generate derivatives, upload immutable objects and
atomically build the catalog with `python scripts/media_pipeline.py publish`
before deploying the backend; setup is documented in `README.md`. Use separate
bucket-scoped read/write credentials for that offline command and read-only
credentials for the backend. Configure R2 CORS
for the exact frontend domain with `GET`/`HEAD`, the `Range` request header,
and exposed `Accept-Ranges`, `Content-Length`, and `Content-Range` response
headers. Backend startup validates every referenced object when R2 credentials
are configured; a missing preview/poster/full object fails startup instead of
allowing an unlock that cannot be viewed.

## 3. Frontend service
- **New → same repo**. **Root directory: `frontend`**.
- Build: `npm install && npm run build` — Start: `npm run start` (Next honors Railway's `$PORT`).
- **Variable:** `NEXT_PUBLIC_API_URL = https://mia-backend.up.railway.app` (the backend's public URL, **no trailing slash**).
  - ⚠️ This is baked in at **build time**. Set it **before** the build; if you change it, **redeploy** the frontend.
  - The WebSocket URL is derived automatically (`https` → `wss`).
- Generate a public domain and open it.

---

## Browsing the data
Open the **PostgreSQL** service → **Data** tab to see/query tables:
- `messages` — who wrote what (user_id, role, content, timestamp, mode)
- `memories` — long-term memory (summarised facts + importance)
- `user_facts` — name, location, etc. per user
- `engagement_state` — chat-batch counters plus durable sales snooze/re-ask state
- `media_offers`, `media_entitlements` — offer rotation and permanent unlocks
- `demo_wallets`, `demo_token_transactions` — internal-demo balance + ledger
- `media_tag_affinity` — soft preference scores from purchases
Or connect any client with the Postgres connection string.

---

## Local development
The app is Postgres-only now. Run a local Postgres, then start both apps:

```bash
# 1. Postgres (Docker)
docker run -d --name mia-pg \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=mia \
  -p 5432:5432 postgres:16

# 2. Backend (uses the default DATABASE_URL = localhost:5432/mia)
uvicorn server.app:app --host 127.0.0.1 --port 8000 --app-dir .

# 3. Frontend (NEXT_PUBLIC_API_URL defaults to http://localhost:8000)
cd frontend && npm run dev
```

Override the DB with `DATABASE_URL=postgresql://user:pass@host:port/db` if needed.
