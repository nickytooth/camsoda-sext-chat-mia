"use client";

import type { MediaAccess, MediaOffer } from "../commerce";
import type { MediaOfferChatMessage } from "../hooks/useChat";
import MediaOfferCard from "./MediaOfferCard";

function formatTime(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

interface Props {
  message: MediaOfferChatMessage;
  balance: number | null;
  unlocked: boolean;
  unlocking: boolean;
  error?: string;
  onUnlock: (offer: MediaOffer) => Promise<void>;
  getAccess: (contentId: string) => Promise<MediaAccess>;
  onRefreshPreview: (offerId: string) => Promise<void>;
}

export default function MediaOfferBubble({
  message,
  balance,
  unlocked,
  unlocking,
  error,
  onUnlock,
  getAccess,
  onRefreshPreview,
}: Props) {
  return (
    <div className="mb-4 flex justify-start px-3 sm:px-4">
      <div className="flex w-full max-w-[27rem] items-end gap-2">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-purple-600 to-pink-500 text-[10px] font-bold text-white">
          M
        </div>
        <div className="min-w-0 flex-1">
          <MediaOfferCard
            offer={message.offer}
            balance={balance}
            unlocked={unlocked}
            unlocking={unlocking}
            error={error}
            onUnlock={onUnlock}
            getAccess={getAccess}
            onRefreshPreview={onRefreshPreview}
          />
          <div className="mt-1 pl-1 text-[11px] text-[var(--muted)]">
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    </div>
  );
}
