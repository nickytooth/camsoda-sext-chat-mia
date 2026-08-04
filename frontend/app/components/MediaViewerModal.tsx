"use client";

import { useEffect } from "react";
import { X } from "lucide-react";
import type { MediaAccess, MediaType } from "../commerce";
import MediaSource from "./MediaSource";

interface Props {
  contentId: string;
  mediaType: MediaType;
  posterUrl?: string | null;
  getAccess: (contentId: string) => Promise<MediaAccess>;
  onClose: () => void;
}

export default function MediaViewerModal({
  contentId,
  mediaType,
  posterUrl,
  getAccess,
  onClose,
}: Props) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Fullscreen private ${mediaType}`}
      className="fixed inset-0 z-[70] flex items-center justify-center bg-black/90 p-3 backdrop-blur-sm sm:p-8"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <button
        type="button"
        onClick={onClose}
        aria-label="Close fullscreen viewer"
        className="absolute right-3 top-3 z-20 rounded-full border border-white/15 bg-black/70 p-2.5 text-white transition-colors hover:bg-white/15 sm:right-6 sm:top-6"
      >
        <X size={20} />
      </button>
      <div className="h-full w-full max-w-6xl overflow-hidden rounded-2xl border border-white/10 bg-black shadow-2xl">
        <MediaSource
          contentId={contentId}
          mediaType={mediaType}
          posterUrl={posterUrl}
          getAccess={getAccess}
          viewer
        />
      </div>
    </div>
  );
}
