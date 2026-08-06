"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { API_BASE, WS_BASE } from "../api";
import type { MediaOffer } from "../commerce";

interface ChatMessageBase {
  id: string;
  role: "user" | "assistant" | "system";
  timestamp: number;
  mode: "sexting" | "story";
}

export interface TextChatMessage extends ChatMessageBase {
  type: "text";
  content: string;
}

export interface MediaOfferChatMessage extends ChatMessageBase {
  type: "media_offer";
  role: "assistant";
  offer: MediaOffer;
}

export type ChatMessage = TextChatMessage | MediaOfferChatMessage;

const HISTORY_LIVE_MATCH_WINDOW_SECONDS = 45;

function textRepresentations(message: TextChatMessage): string[] {
  const lines = message.content
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length > 1 ? [message.content, ...lines] : [message.content];
}

/**
 * Reconcile the durable mixed feed with messages that arrived over the socket
 * while the history request was in flight.
 *
 * History is canonical for persisted rows (and has the timestamps needed to
 * keep a teaser immediately before its media card). Live assistant responses
 * arrive as separate bubbles while the database stores those bubbles joined
 * by newlines, so the bounded representation match also handles that shape.
 */
function mergeHistoryMessages(
  current: ChatMessage[],
  history: ChatMessage[],
): ChatMessage[] {
  const currentOffers = new Map(
    current
      .filter((message): message is MediaOfferChatMessage => message.type === "media_offer")
      .map((message) => [message.offer.offer_id, message]),
  );
  const historyOfferIds = new Set<string>();
  const historyTextIds = new Set<string>();
  const historyTextRepresentations: Array<{
    role: ChatMessage["role"];
    content: string;
    timestamp: number;
    used: boolean;
  }> = [];

  const merged = history.map((message) => {
    if (message.type === "media_offer") {
      historyOfferIds.add(message.offer.offer_id);
      const live = currentOffers.get(message.offer.offer_id);
      if (live?.offer.unlocked && !message.offer.unlocked) {
        return {
          ...message,
          offer: { ...message.offer, unlocked: true },
        };
      }
      return message;
    }

    historyTextIds.add(message.id);
    for (const content of textRepresentations(message)) {
      historyTextRepresentations.push({
        role: message.role,
        content,
        timestamp: message.timestamp,
        used: false,
      });
    }
    return message;
  });

  for (const message of current) {
    if (message.type === "media_offer") {
      if (!historyOfferIds.has(message.offer.offer_id)) merged.push(message);
      continue;
    }

    if (historyTextIds.has(message.id)) continue;
    const representation = historyTextRepresentations.find(
      (candidate) =>
        !candidate.used &&
        candidate.role === message.role &&
        candidate.content === message.content &&
        Math.abs(candidate.timestamp - message.timestamp) <=
          HISTORY_LIVE_MATCH_WINDOW_SECONDS,
    );
    if (representation) {
      representation.used = true;
      continue;
    }
    merged.push(message);
  }

  return merged.sort((left, right) => {
    const timestampOrder = left.timestamp - right.timestamp;
    if (timestampOrder !== 0) return timestampOrder;
    if (left.type === right.type) return 0;
    return left.type === "text" ? -1 : 1;
  });
}

interface UseChatOptions {
  wsUrl?: string;
  userId?: number;
  userName?: string;
  // While true, the one-time opening is NOT requested on connect — the intro
  // popup is showing. Dismissing it calls releaseOpening() to kick it off.
  holdOpening?: boolean;
}

interface HistoryResponse {
  messages?: Array<
    | {
        type: "text";
        id?: string;
        role: ChatMessage["role"];
        content: string;
        timestamp: number;
        mode?: ChatMessage["mode"];
      }
    | {
        type: "media_offer";
        id?: string;
        role: "assistant";
        timestamp: number;
        mode?: ChatMessage["mode"];
        offer: MediaOffer;
      }
  >;
}

export function useChat({ wsUrl = `${WS_BASE}/ws/chat`, userId = 1, userName = "", holdOpening = false }: UseChatOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const mode = "sexting" as const;
  const [isConnected, setIsConnected] = useState(false);
  const [isOpening, setIsOpening] = useState(false);
  const [isCardDelivering, setIsCardDelivering] = useState(false);
  const [commerceRevision, setCommerceRevision] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  // Invalidates sockets, reconnect timers and history requests created by an
  // older hook lifecycle (including React StrictMode's mount/cleanup probe).
  const lifecycleGeneration = useRef(0);
  const disposed = useRef(false);
  // Delay before her typing bubble appears after a send, so it doesn't pop in
  // instantly (feels more human).
  const typingTimer = useRef<NodeJS.Timeout | null>(null);
  const idCounter = useRef(0);
  const openingAnimating = useRef(false);
  const liveOpeningInProgress = useRef(false);
  // The opening is animated once per session. Without this, a reconnect/reload
  // would replay the first bubbles every time, because the (still unanswered)
  // history is all-assistant and looks "fresh" again.
  const sextingOpeningPlayed = useRef(false);
  // Latest holdOpening value, readable from inside ws.onopen (which closes
  // over the first render otherwise).
  const holdOpeningRef = useRef(holdOpening);
  useEffect(() => {
    holdOpeningRef.current = holdOpening;
  }, [holdOpening]);

  const genId = () => `msg-${Date.now()}-${idCounter.current++}`;

  // Load history for current mode
  const loadHistory = useCallback(async (
    m: ChatMessage["mode"],
    generation = lifecycleGeneration.current,
  ) => {
    try {
      const res = await fetch(`${API_BASE}/api/history/${m}?user_id=${userId}`);
      if (!res.ok) return;
      const data = (await res.json()) as HistoryResponse;
      if (disposed.current || generation !== lifecycleGeneration.current) return;
      const loaded: ChatMessage[] = [];
      for (const msg of data.messages || []) {
        if (msg.type === "media_offer") {
          if (!msg.offer?.offer_id || !msg.offer?.content_id) continue;
          loaded.push({
            type: "media_offer" as const,
            id: msg.id || genId(),
            role: "assistant" as const,
            timestamp: msg.timestamp,
            mode: msg.mode || m,
            offer: msg.offer,
          });
          continue;
        }
        if (msg.type !== "text" || typeof msg.content !== "string") continue;
        loaded.push({
          type: "text" as const,
          id: msg.id || genId(),
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp,
          mode: msg.mode || m,
        });
      }

      // Fresh opening: Mia initiated and the user hasn't replied yet
      // (every loaded message is from her). Play it out with the typing
      // indicator and send the bubbles one by one, instead of dumping them.
      const isFreshOpening =
        m === "sexting" &&
        loaded.length > 0 &&
        loaded.every((msg) => msg.type === "text" && msg.role === "assistant");

      if (isFreshOpening) {
        // The backend already owns the paced opening delivery on this socket.
        // Its DB rows can become visible to this in-flight history request
        // before the last live bubble arrives, so do not replay or reveal them.
        if (liveOpeningInProgress.current) return;
        // Guard against double-run (React StrictMode mounts effects twice in
        // dev, which would otherwise append every bubble twice).
        if (openingAnimating.current) return;
        // Already animated once this session (e.g. switched away and back):
        // just show the bubbles statically instead of replaying them.
        if (sextingOpeningPlayed.current) {
          setMessages((current) => mergeHistoryMessages(current, loaded));
          return;
        }
        openingAnimating.current = true;
        sextingOpeningPlayed.current = true;
        setIsOpening(true);
        try {
          setMessages([]);
          for (let i = 0; i < loaded.length; i++) {
            if (disposed.current || generation !== lifecycleGeneration.current) return;
            setIsTyping(true);
            // Longer pause before the first bubble, a little shorter between the rest
            await new Promise((r) => setTimeout(r, i === 0 ? 2500 : 1300));
            if (disposed.current || generation !== lifecycleGeneration.current) return;
            setIsTyping(false);
            setMessages((prev) => [...prev, loaded[i]]);
            // small gap so consecutive bubbles don't appear in the same frame
            if (i < loaded.length - 1) {
              await new Promise((r) => setTimeout(r, 350));
            }
          }
        } finally {
          openingAnimating.current = false;
          if (!disposed.current && generation === lifecycleGeneration.current) {
            setIsOpening(false);
          }
        }
        return;
      }

      setMessages((current) => mergeHistoryMessages(current, loaded));
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }, [userId]);

  // Locked previews use short-lived signed sources too. If one expires while
  // the chat is open, refresh only that offer from history without replacing
  // the live conversation or replaying the opening animation.
  const refreshMediaOffer = useCallback(async (offerId: string) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/history/sexting?user_id=${userId}`,
        { cache: "no-store" },
      );
      if (!response.ok) return;
      const data = (await response.json()) as HistoryResponse;
      const refreshed = data.messages?.find(
        (item) => item.type === "media_offer" && item.offer.offer_id === offerId,
      );
      if (!refreshed || refreshed.type !== "media_offer") return;
      setMessages((current) =>
        current.map((item) =>
          item.type === "media_offer" && item.offer.offer_id === offerId
            ? {
                ...item,
                offer: {
                  ...refreshed.offer,
                  unlocked: item.offer.unlocked || refreshed.offer.unlocked,
                },
              }
            : item,
        ),
      );
    } catch (error) {
      console.error("Failed to refresh media preview:", error);
    }
  }, [userId]);

  // Connect WebSocket
  const connect = useCallback(function connectSocket(
    generation = lifecycleGeneration.current,
    reloadHistory = false,
  ) {
    if (disposed.current || generation !== lifecycleGeneration.current) return;
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) return;

    const ws = new WebSocket(`${wsUrl}?user_id=${userId}&user_name=${encodeURIComponent(userName)}`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (
        disposed.current ||
        generation !== lifecycleGeneration.current ||
        wsRef.current !== ws
      ) {
        ws.onclose = null;
        ws.close();
        return;
      }
      setIsConnected(true);
      console.log("WebSocket connected");
      if (reloadHistory) {
        void loadHistory(mode, generation);
      }
      // Ask the backend for the one-time opening — unless the intro popup is
      // up, in which case releaseOpening() sends this on dismiss. Idempotent
      // server-side, so reconnects are safe.
      if (!holdOpeningRef.current) {
        ws.send(JSON.stringify({ type: "start" }));
      }
    };

    ws.onmessage = (event) => {
      if (
        disposed.current ||
        generation !== lifecycleGeneration.current ||
        wsRef.current !== ws
      ) return;
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "opening_start":
          // Backend is delivering the one-time opening over the socket — lock
          // the input for its whole duration (input is gated on isOpening).
          liveOpeningInProgress.current = true;
          sextingOpeningPlayed.current = true;
          setIsOpening(true);
          break;
        case "opening_end":
          liveOpeningInProgress.current = false;
          setIsOpening(false);
          break;
        case "card_start":
          setIsCardDelivering(true);
          break;
        case "card_end":
          setIsCardDelivering(false);
          break;
        case "typing_start":
          if (typingTimer.current) clearTimeout(typingTimer.current);
          setIsTyping(true);
          break;
        case "typing_end":
          if (typingTimer.current) { clearTimeout(typingTimer.current); typingTimer.current = null; }
          setIsTyping(false);
          break;
        case "message":
          if (typingTimer.current) { clearTimeout(typingTimer.current); typingTimer.current = null; }
          setIsTyping(false);
          const newMsg: ChatMessage = {
            type: "text",
            id: genId(),
            role: data.role || "assistant",
            content: data.content,
            timestamp: data.timestamp || Date.now() / 1000,
            mode: data.mode || "sexting",
          };
          setMessages((prev) => [...prev, newMsg]);
          break;
        case "media_offer": {
          if (!data.offer?.offer_id || !data.offer?.content_id) break;
          const mediaMessage: MediaOfferChatMessage = {
            type: "media_offer",
            id: data.id || genId(),
            role: "assistant",
            timestamp: data.timestamp || Date.now() / 1000,
            mode: data.mode || "sexting",
            offer: data.offer as MediaOffer,
          };
          setMessages((previous) => {
            const duplicate = previous.some(
              (item) =>
                item.type === "media_offer" &&
                item.offer.offer_id === mediaMessage.offer.offer_id,
            );
            return duplicate ? previous : [...previous, mediaMessage];
          });
          break;
        }
        case "media_entitlement_updated":
          setMessages((previous) =>
            previous.map((item) =>
              item.type === "media_offer" &&
              (item.offer.offer_id === data.offer_id ||
                item.offer.content_id === data.content_id)
                ? { ...item, offer: { ...item.offer, unlocked: true } }
                : item,
            ),
          );
          setCommerceRevision((revision) => revision + 1);
          break;
        case "flagged":
          if (typingTimer.current) { clearTimeout(typingTimer.current); typingTimer.current = null; }
          setIsTyping(false);
          setMessages((prev) => [...prev, {
            type: "text",
            id: genId(),
            role: "system",
            content: "This message was flagged as inappropriate. Please try again.",
            timestamp: Date.now() / 1000,
            mode: data.mode || "sexting",
          }]);
          break;
      }
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      wsRef.current = null;
      if (disposed.current || generation !== lifecycleGeneration.current) return;
      liveOpeningInProgress.current = false;
      setIsOpening(false);
      setIsCardDelivering(false);
      setIsTyping(false);
      setIsConnected(false);
      console.log("WebSocket disconnected, reconnecting in 3s...");
      reconnectTimer.current = setTimeout(
        () => connectSocket(generation, true),
        3000,
      );
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
    };
  }, [wsUrl, userId, userName, loadHistory, mode]);

  // Send message
  const sendMessage = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      const now = Date.now() / 1000;

      const newMessage: ChatMessage = {
        type: "text",
        id: genId(),
        role: "user",
        content: text,
        timestamp: now,
        mode,
      };
      setMessages((prev) => [...prev, newMessage]);

      // Send via WebSocket
      wsRef.current.send(
        JSON.stringify({
          type: "message",
          content: text,
          mode,
        })
      );
    },
    [mode]
  );

  // Card request (Hear a fantasy / Hear a story): show the user's request as a
  // bubble, then ask the backend to pull a (non-repeating) item from the library.
  const triggerCard = useCallback(
    (kind: "fantasy" | "story", requestText: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      setMessages((prev) => [
        ...prev,
        { type: "text", id: genId(), role: "user", content: requestText, timestamp: Date.now() / 1000, mode },
      ]);
      setIsTyping(true);
      wsRef.current.send(JSON.stringify({ type: "card", kind, content: requestText, mode }));
    },
    [mode]
  );

  // Intro popup dismissed — request Mia's one-time opening now.
  const releaseOpening = useCallback(() => {
    holdOpeningRef.current = false;
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "start" }));
    }
    // If the socket isn't open yet, ws.onopen will send it (hold is cleared).
  }, []);

  // AI Help — ask the backend to draft a reply the user can approve/edit
  const suggestReply = useCallback(async (): Promise<string> => {
    try {
      const res = await fetch(
        `${API_BASE}/api/suggest?user_id=${userId}&mode=${mode}`,
        { method: "POST" }
      );
      if (!res.ok) return "";
      const data = await res.json();
      return (data.suggestion || "").trim();
    } catch (e) {
      console.error("Suggest failed:", e);
      return "";
    }
  }, [userId, mode]);

  // Connect and load history. connect/loadHistory are memoised on the
  // connection identity (wsUrl/userId/userName) and userId respectively, so this
  // re-runs — reconnecting and reloading — whenever the user changes.
  useEffect(() => {
    const generation = lifecycleGeneration.current + 1;
    lifecycleGeneration.current = generation;
    disposed.current = false;
    liveOpeningInProgress.current = false;
    openingAnimating.current = false;
    sextingOpeningPlayed.current = false;
    setMessages([]);
    setIsConnected(false);
    connect(generation);
    const historyTimer = setTimeout(() => {
      void loadHistory(mode, generation);
    }, 0);
    return () => {
      disposed.current = true;
      if (lifecycleGeneration.current === generation) {
        lifecycleGeneration.current += 1;
      }
      clearTimeout(historyTimer);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (typingTimer.current) clearTimeout(typingTimer.current);
      const socket = wsRef.current;
      wsRef.current = null;
      if (socket) {
        // An intentional lifecycle close must never enter ws.onclose's retry
        // path and create a detached "ghost" connection after unmount/reset.
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
      }
    };
  }, [connect, loadHistory, mode]);

  return {
    messages,
    isTyping,
    isConnected,
    isOpening,
    isCardDelivering,
    commerceRevision,
    mode,
    sendMessage,
    suggestReply,
    triggerCard,
    releaseOpening,
    refreshMediaOffer,
  };
}
