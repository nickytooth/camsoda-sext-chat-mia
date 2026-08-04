"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, Loader2, Maximize2, RefreshCw } from "lucide-react";
import { MediaAccess, MediaType, resolveMediaUrl } from "../commerce";

interface Props {
  contentId: string;
  mediaType: MediaType;
  posterUrl?: string | null;
  getAccess: (contentId: string) => Promise<MediaAccess>;
  onExpand?: () => void;
  viewer?: boolean;
}

export default function MediaSource({
  contentId,
  mediaType,
  posterUrl,
  getAccess,
  onExpand,
  viewer = false,
}: Props) {
  const [access, setAccess] = useState<MediaAccess | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const requestNumber = useRef(0);
  const lastRecovery = useRef(0);

  const refresh = useCallback(async () => {
    const request = ++requestNumber.current;
    setLoading(true);
    setError(null);
    try {
      const next = await getAccess(contentId);
      if (request === requestNumber.current) setAccess(next);
    } catch (reason) {
      console.error("Failed to load unlocked media:", reason);
      if (request === requestNumber.current) {
        setError("This private media couldn’t be loaded.");
      }
    } finally {
      if (request === requestNumber.current) setLoading(false);
    }
  }, [contentId, getAccess]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => {
      window.clearTimeout(timer);
      requestNumber.current += 1;
    };
  }, [refresh]);

  // Access is permanent, while its browser source is intentionally short-lived.
  // Refresh it quietly before expiry so an open image/player keeps working.
  useEffect(() => {
    if (!access) return;
    const refreshIn = Math.max(access.expires_at * 1000 - Date.now() - 15_000, 1_000);
    const timer = window.setTimeout(() => void refresh(), Math.min(refreshIn, 2_000_000_000));
    return () => window.clearTimeout(timer);
  }, [access, refresh]);

  const recoverFromMediaError = () => {
    const now = Date.now();
    if (now - lastRecovery.current < 5_000) return;
    lastRecovery.current = now;
    void refresh();
  };

  const frameClass = viewer
    ? "w-full h-full min-h-[45vh] max-h-[82vh]"
    : "w-full min-h-64 max-h-[32rem]";

  if (loading && !access) {
    return (
      <div className={`${frameClass} flex items-center justify-center bg-black/40`}>
        <div className="flex items-center gap-2 text-sm text-white/65">
          <Loader2 size={18} className="animate-spin text-pink-400" />
          Opening private {mediaType}…
        </div>
      </div>
    );
  }

  if (error || !access) {
    return (
      <div className={`${frameClass} flex flex-col items-center justify-center gap-3 bg-black/40 px-6 text-center`}>
        <AlertCircle size={24} className="text-rose-400" />
        <p className="text-sm text-white/70">{error || "Media unavailable"}</p>
        <button
          type="button"
          onClick={() => void refresh()}
          className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-4 py-2 text-xs text-white hover:bg-white/10"
        >
          <RefreshCw size={14} />
          Try again
        </button>
      </div>
    );
  }

  const source = resolveMediaUrl(access.access_url);
  const expandButton = onExpand ? (
    <button
      type="button"
      onClick={onExpand}
      aria-label="Open fullscreen"
      className="absolute right-2 top-2 z-10 rounded-full bg-black/65 p-2 text-white backdrop-blur-sm transition-colors hover:bg-black/85"
    >
      <Maximize2 size={16} />
    </button>
  ) : null;

  return (
    <div className={`relative ${viewer ? "flex items-center justify-center" : ""} bg-black`}>
      {expandButton}
      {mediaType === "photo" ? (
        // Signed bearer URLs should go straight to R2/the backend rather than
        // through Next's public image optimiser and its shared cache.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={source}
          alt="Unlocked private photo from Mia"
          referrerPolicy="no-referrer"
          onError={recoverFromMediaError}
          className={`${frameClass} object-contain`}
        />
      ) : (
        <video
          key={source}
          controls
          playsInline
          preload="metadata"
          poster={resolveMediaUrl(posterUrl)}
          onError={recoverFromMediaError}
          aria-label="Unlocked private video from Mia"
          className={`${frameClass} object-contain`}
        >
          <source src={source} type={access.mime_type || "video/mp4"} />
          Your browser does not support this video.
        </video>
      )}
      {loading && access && (
        <div className="pointer-events-none absolute left-2 top-2 rounded-full bg-black/65 p-2 text-pink-300">
          <Loader2 size={14} className="animate-spin" />
        </div>
      )}
    </div>
  );
}
