import { API_BASE } from "./api";

export type MediaType = "photo" | "video";

export interface MediaOffer {
  offer_id: string;
  content_id: string;
  media_type: MediaType;
  price_tokens: number;
  preview_url: string;
  poster_url: string | null;
  aspect_ratio: number;
  duration_seconds: number | null;
  unlocked: boolean;
}

export interface MediaGalleryItem {
  content_id: string;
  media_type: MediaType;
  aspect_ratio: number;
  duration_seconds: number | null;
  unlocked_at: number;
  preview_url: string;
  poster_url: string | null;
}

export interface MediaAccess {
  content_id: string;
  media_type: MediaType;
  access_url: string;
  mime_type: string;
  expires_at: number;
}

export interface UnlockResult {
  balance: number;
  already_unlocked: boolean;
  entitlement: {
    content_id: string;
    media_type: MediaType;
  };
}

export class CommerceApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "CommerceApiError";
    this.status = status;
  }
}

function withUser(path: string, userId: number): string {
  const separator = path.includes("?") ? "&" : "?";
  return `${API_BASE}${path}${separator}user_id=${encodeURIComponent(userId)}`;
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    cache: "no-store",
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let message = "Something went wrong. Please try again.";
    try {
      const body = (await response.json()) as { detail?: string; message?: string };
      message = body.detail || body.message || message;
    } catch {
      // The status code still gives callers enough information for a useful UI.
    }
    throw new CommerceApiError(message, response.status);
  }

  return (await response.json()) as T;
}

export function createIdempotencyKey(prefix: "unlock" | "refill"): string {
  const randomPart =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${randomPart}`;
}

export async function getWallet(userId: number): Promise<number> {
  const result = await requestJson<{ balance: number }>(
    withUser("/api/demo/wallet", userId),
  );
  return result.balance;
}

export async function refillWallet(userId: number): Promise<{ balance: number; credited: number }> {
  return requestJson(withUser("/api/demo/wallet/refill", userId), {
    method: "POST",
    body: JSON.stringify({ idempotency_key: createIdempotencyKey("refill") }),
  });
}

export async function unlockMediaOffer(
  userId: number,
  offerId: string,
): Promise<UnlockResult> {
  return requestJson(
    withUser(`/api/media/offers/${encodeURIComponent(offerId)}/unlock`, userId),
    {
      method: "POST",
      body: JSON.stringify({ idempotency_key: createIdempotencyKey("unlock") }),
    },
  );
}

export async function getMediaGallery(userId: number): Promise<MediaGalleryItem[]> {
  const result = await requestJson<{ items: MediaGalleryItem[] }>(
    withUser("/api/media/gallery", userId),
  );
  return result.items;
}

export async function getMediaAccess(userId: number, contentId: string): Promise<MediaAccess> {
  return requestJson(
    withUser(`/api/media/${encodeURIComponent(contentId)}/access`, userId),
  );
}

/**
 * The backend may return an absolute signed R2 URL or a relative local/demo
 * media route. Normalising here keeps storage URLs out of message rendering.
 */
export function resolveMediaUrl(url: string | null | undefined): string {
  if (!url) return "";
  try {
    return new URL(url, `${API_BASE}/`).toString();
  } catch {
    return "";
  }
}

export function mediaLabel(type: MediaType): string {
  return type === "photo" ? "Photo" : "Video";
}

export function formatDuration(seconds: number | null): string | null {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return null;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}
