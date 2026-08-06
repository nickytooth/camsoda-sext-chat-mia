"use client";

import { useEffect, useState } from "react";
import { Image as ImageIcon, LockOpen, X } from "lucide-react";
import type { MediaAccess, MediaGalleryItem } from "../commerce";
import MediaSource from "./MediaSource";
import MediaViewerModal from "./MediaViewerModal";

interface Props {
  items: MediaGalleryItem[];
  error?: string | null;
  getAccess: (contentId: string) => Promise<MediaAccess>;
  onClose: () => void;
}

function unlockedDate(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleDateString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function UnlockedGallery({ items, error, getAccess, onClose }: Props) {
  const [selected, setSelected] = useState<MediaGalleryItem | null>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (selected) setSelected(null);
      else onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, selected]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Unlocked Gallery"
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 p-0 backdrop-blur-sm sm:p-6"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section className="flex h-full w-full max-w-6xl flex-col overflow-hidden bg-[#101018] shadow-2xl sm:h-[88vh] sm:rounded-2xl sm:border sm:border-white/10">
        <header className="flex items-center justify-between border-b border-white/10 px-4 py-4 sm:px-6">
          <div>
            <div className="flex items-center gap-2">
              <LockOpen size={18} className="text-pink-400" />
              <h2 className="text-base font-semibold text-white">Unlocked Gallery</h2>
            </div>
            <p className="mt-1 text-xs text-white/45">
              {items.length === 1 ? "1 private item" : `${items.length} private items`}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close gallery"
            className="rounded-full p-2 text-white/70 transition-colors hover:bg-white/10 hover:text-white"
          >
            <X size={20} />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-4 sm:p-6">
          {error && (
            <div className="mb-4 rounded-xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-200" role="alert">
              {error}
            </div>
          )}
          {!error && items.length === 0 ? (
            <div className="flex h-full min-h-72 flex-col items-center justify-center text-center">
              <span className="flex h-16 w-16 items-center justify-center rounded-full bg-pink-400/10 text-pink-300">
                <ImageIcon size={28} />
              </span>
              <h3 className="mt-4 text-base font-semibold text-white">Nothing unlocked yet</h3>
              <p className="mt-1 max-w-xs text-sm text-white/45">
                Photos and videos you unlock in chat will stay here.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
              {items.map((item) => (
                <article
                  key={item.content_id}
                  className="overflow-hidden rounded-2xl border border-white/10 bg-[#171522]"
                >
                  <MediaSource
                    contentId={item.content_id}
                    mediaType={item.media_type}
                    posterUrl={item.poster_url}
                    getAccess={getAccess}
                    onExpand={() => setSelected(item)}
                  />
                  <div className="flex items-center justify-between gap-3 px-3 py-2.5 text-xs">
                    <span className="font-medium text-white/80">
                      {item.media_type === "photo" ? "Photo" : "Video"}
                    </span>
                    <span className="text-white/40">Unlocked {unlockedDate(item.unlocked_at)}</span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      </section>

      {selected && (
        <MediaViewerModal
          contentId={selected.content_id}
          mediaType={selected.media_type}
          posterUrl={selected.poster_url}
          getAccess={getAccess}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
