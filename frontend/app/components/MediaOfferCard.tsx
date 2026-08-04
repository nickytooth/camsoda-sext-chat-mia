"use client";

import { useEffect, useRef, useState } from "react";
import { AlertCircle, Check, Coins, Loader2, LockKeyhole, Play } from "lucide-react";
import {
  formatDuration,
  mediaLabel,
  type MediaAccess,
  type MediaOffer,
  resolveMediaUrl,
} from "../commerce";
import MediaSource from "./MediaSource";
import MediaViewerModal from "./MediaViewerModal";

interface Props {
  offer: MediaOffer;
  balance: number | null;
  unlocked: boolean;
  unlocking: boolean;
  error?: string;
  onUnlock: (offer: MediaOffer) => Promise<void>;
  getAccess: (contentId: string) => Promise<MediaAccess>;
  onRefreshPreview?: (offerId: string) => Promise<void>;
}

function signedUrlExpiryMs(url: string): number | null {
  try {
    const parsed = new URL(url);
    const stamp = parsed.searchParams.get("X-Amz-Date");
    const lifetime = Number(parsed.searchParams.get("X-Amz-Expires"));
    const match = stamp?.match(
      /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/,
    );
    if (!match || !Number.isFinite(lifetime) || lifetime <= 0) return null;
    return (
      Date.UTC(
        Number(match[1]),
        Number(match[2]) - 1,
        Number(match[3]),
        Number(match[4]),
        Number(match[5]),
        Number(match[6]),
      ) +
      lifetime * 1000
    );
  } catch {
    return null;
  }
}

export default function MediaOfferCard({
  offer,
  balance,
  unlocked,
  unlocking,
  error,
  onUnlock,
  getAccess,
  onRefreshPreview,
}: Props) {
  const [viewerOpen, setViewerOpen] = useState(false);
  const [failedPreviewUrl, setFailedPreviewUrl] = useState<string | null>(null);
  const previewRefreshAttempted = useRef(false);
  const duration = formatDuration(offer.duration_seconds);
  const insufficient = balance !== null && balance < offer.price_tokens;
  const preview = resolveMediaUrl(
    offer.media_type === "video"
      ? offer.poster_url || offer.preview_url
      : offer.preview_url,
  );
  const safeRatio =
    Number.isFinite(offer.aspect_ratio) && offer.aspect_ratio > 0
      ? offer.aspect_ratio
      : 0.75;
  const previewFailed = !preview || failedPreviewUrl === preview;

  // Locked derivatives are signed too. Refresh shortly before their bearer
  // URL expires so a long-open conversation never falls back to a blank card.
  useEffect(() => {
    if (unlocked || !preview || !onRefreshPreview) return;
    const expiresAt = signedUrlExpiryMs(preview);
    if (expiresAt === null) return;
    const refreshIn = Math.max(expiresAt - Date.now() - 15_000, 1_000);
    const timer = window.setTimeout(
      () => void onRefreshPreview(offer.offer_id),
      Math.min(refreshIn, 2_000_000_000),
    );
    return () => window.clearTimeout(timer);
  }, [offer.offer_id, onRefreshPreview, preview, unlocked]);

  const handlePreviewError = () => {
    if (!preview) return;
    setFailedPreviewUrl(preview);
    if (!onRefreshPreview || previewRefreshAttempted.current) return;
    previewRefreshAttempted.current = true;
    void onRefreshPreview(offer.offer_id);
  };

  if (unlocked) {
    return (
      <div className="w-full overflow-hidden rounded-2xl border border-pink-400/20 bg-[#14131f] shadow-[0_18px_55px_rgba(0,0,0,0.28)]">
        <div className="flex items-center justify-between border-b border-white/8 px-3 py-2.5">
          <div className="flex items-center gap-2 text-xs font-medium text-white/80">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-400/15 text-emerald-300">
              <Check size={14} />
            </span>
            Unlocked {mediaLabel(offer.media_type).toLowerCase()}
          </div>
          {duration && <span className="text-[11px] text-white/45">{duration}</span>}
        </div>
        <MediaSource
          contentId={offer.content_id}
          mediaType={offer.media_type}
          posterUrl={offer.poster_url}
          getAccess={getAccess}
          onExpand={() => setViewerOpen(true)}
        />
        {viewerOpen && (
          <MediaViewerModal
            contentId={offer.content_id}
            mediaType={offer.media_type}
            posterUrl={offer.poster_url}
            getAccess={getAccess}
            onClose={() => setViewerOpen(false)}
          />
        )}
      </div>
    );
  }

  return (
    <div className="w-full overflow-hidden rounded-2xl border border-pink-400/20 bg-[#171522] shadow-[0_18px_55px_rgba(0,0,0,0.28)]">
      <div
        className="relative w-full max-h-[30rem] min-h-64 overflow-hidden bg-gradient-to-br from-purple-950 via-[#1b1328] to-black"
        style={{ aspectRatio: safeRatio }}
      >
        {!previewFailed ? (
          // Preview assets are intentionally separate from the protected original.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt={`Locked ${offer.media_type} preview from Mia`}
            referrerPolicy="no-referrer"
            onError={handlePreviewError}
            className="h-full w-full scale-[1.03] object-cover brightness-[0.58] blur-[2px]"
          />
        ) : (
          <div className="h-full w-full bg-[radial-gradient(circle_at_50%_25%,rgba(236,72,153,0.28),transparent_48%)]" />
        )}

        <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/20 to-black/25" />
        {offer.media_type === "video" && (
          <div className="absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full bg-black/65 px-2.5 py-1 text-[11px] font-medium text-white backdrop-blur">
            <Play size={12} className="fill-current" />
            {duration || "Video"}
          </div>
        )}
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-5 text-center">
          <span className="flex h-14 w-14 items-center justify-center rounded-full border border-white/20 bg-black/55 text-white shadow-xl backdrop-blur-md">
            <LockKeyhole size={25} />
          </span>
          <div>
            <p className="text-sm font-semibold text-white">Private {mediaLabel(offer.media_type)}</p>
            <p className="mt-1 text-xs text-white/60">Unlock to view it here in chat</p>
          </div>
        </div>
      </div>

      <div className="space-y-2.5 p-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-white">
              {mediaLabel(offer.media_type)} · {offer.price_tokens} tokens
            </p>
            <p className="mt-0.5 text-[11px] text-white/45">Yours permanently after unlock</p>
          </div>
          <button
            type="button"
            onClick={() => void onUnlock(offer)}
            disabled={unlocking || insufficient}
            className="inline-flex min-w-24 items-center justify-center gap-1.5 rounded-xl bg-gradient-to-r from-purple-600 to-pink-500 px-3.5 py-2.5 text-xs font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-45"
          >
            {unlocking ? <Loader2 size={15} className="animate-spin" /> : <Coins size={15} />}
            {unlocking ? "Unlocking" : "Unlock"}
          </button>
        </div>

        {insufficient && !error && (
          <div className="flex items-center gap-2 rounded-lg bg-amber-400/10 px-2.5 py-2 text-[11px] text-amber-200">
            <AlertCircle size={14} className="shrink-0" />
            Not enough tokens — use Refill in the header.
          </div>
        )}
        {error && (
          <div className="flex items-center gap-2 rounded-lg bg-rose-400/10 px-2.5 py-2 text-[11px] text-rose-200" role="alert">
            <AlertCircle size={14} className="shrink-0" />
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
