"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CommerceApiError,
  getMediaAccess,
  getMediaGallery,
  getWallet,
  MediaAccess,
  MediaGalleryItem,
  MediaOffer,
  refillWallet,
  unlockMediaOffer,
} from "../commerce";

function unlockErrorMessage(error: unknown): string {
  if (error instanceof CommerceApiError) {
    if (error.status === 402) return "You need more tokens to unlock this.";
    if (error.status === 404) return "This private drop is no longer available.";
    return error.message;
  }
  return "Couldn’t unlock this right now. Try again.";
}

export function useCommerce(userId: number, externalRevision = 0) {
  const [balance, setBalance] = useState<number | null>(null);
  const [gallery, setGallery] = useState<MediaGalleryItem[]>([]);
  const [optimisticUnlocked, setOptimisticUnlocked] = useState<Set<string>>(
    () => new Set(),
  );
  const [unlocking, setUnlocking] = useState<Set<string>>(() => new Set());
  const [unlockErrors, setUnlockErrors] = useState<Record<string, string>>({});
  const [isRefilling, setIsRefilling] = useState(false);
  const [walletError, setWalletError] = useState<string | null>(null);
  const [galleryError, setGalleryError] = useState<string | null>(null);
  const refillInFlight = useRef(false);
  const unlocksInFlight = useRef<Set<string>>(new Set());

  const refreshWallet = useCallback(async () => {
    try {
      const nextBalance = await getWallet(userId);
      setBalance(nextBalance);
      setWalletError(null);
    } catch (error) {
      console.error("Failed to load token wallet:", error);
      setWalletError("Token balance unavailable");
    }
  }, [userId]);

  const refreshGallery = useCallback(async () => {
    try {
      const items = await getMediaGallery(userId);
      setGallery(items);
      setGalleryError(null);
      setOptimisticUnlocked((current) => {
        if (current.size === 0) return current;
        const persisted = new Set(items.map((item) => item.content_id));
        const pending = new Set([...current].filter((id) => !persisted.has(id)));
        return pending.size === current.size ? current : pending;
      });
    } catch (error) {
      console.error("Failed to load unlocked gallery:", error);
      setGalleryError("Couldn’t load your gallery");
    }
  }, [userId]);

  useEffect(() => {
    void Promise.all([refreshWallet(), refreshGallery()]);
  }, [refreshWallet, refreshGallery, externalRevision]);

  const unlockedContentIds = useMemo(() => {
    const ids = new Set(gallery.map((item) => item.content_id));
    optimisticUnlocked.forEach((id) => ids.add(id));
    return ids;
  }, [gallery, optimisticUnlocked]);

  const refill = useCallback(async () => {
    if (refillInFlight.current) return;
    refillInFlight.current = true;
    setIsRefilling(true);
    try {
      const result = await refillWallet(userId);
      setBalance(result.balance);
      setWalletError(null);
    } catch (error) {
      console.error("Failed to refill token wallet:", error);
      setWalletError(
        error instanceof CommerceApiError ? error.message : "Couldn’t refill tokens",
      );
    } finally {
      refillInFlight.current = false;
      setIsRefilling(false);
    }
  }, [userId]);

  const unlock = useCallback(
    async (offer: MediaOffer) => {
      if (unlocksInFlight.current.has(offer.offer_id)) return;
      unlocksInFlight.current.add(offer.offer_id);

      setUnlocking((current) => new Set(current).add(offer.offer_id));
      setUnlockErrors((current) => {
        const next = { ...current };
        delete next[offer.offer_id];
        return next;
      });

      try {
        const result = await unlockMediaOffer(userId, offer.offer_id);
        setBalance(result.balance);
        setOptimisticUnlocked((current) =>
          new Set(current).add(result.entitlement.content_id),
        );
        void refreshGallery();
      } catch (error) {
        setUnlockErrors((current) => ({
          ...current,
          [offer.offer_id]: unlockErrorMessage(error),
        }));
      } finally {
        unlocksInFlight.current.delete(offer.offer_id);
        setUnlocking((current) => {
          const next = new Set(current);
          next.delete(offer.offer_id);
          return next;
        });
      }
    },
    [refreshGallery, userId],
  );

  const access = useCallback(
    (contentId: string): Promise<MediaAccess> => getMediaAccess(userId, contentId),
    [userId],
  );

  return {
    balance,
    gallery,
    unlockedContentIds,
    unlocking,
    unlockErrors,
    isRefilling,
    walletError,
    galleryError,
    refill,
    unlock,
    getAccess: access,
    refreshGallery,
  };
}
