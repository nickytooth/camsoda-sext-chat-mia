// Backend base URL. In production set NEXT_PUBLIC_API_URL to the deployed
// backend (e.g. https://victoria-backend.up.railway.app). Falls back to local
// dev. WS_BASE is derived so https -> wss automatically.
export const API_BASE =
  (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export const WS_BASE = API_BASE.replace(/^http/, "ws");
