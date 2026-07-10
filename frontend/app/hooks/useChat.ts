"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { API_BASE, WS_BASE } from "../api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  mode: "sexting" | "story";
  imageUrl?: string;
}

interface UseChatOptions {
  wsUrl?: string;
  userId?: number;
  userName?: string;
  // While true, the one-time opening is NOT requested on connect — the intro
  // popup is showing. Dismissing it calls releaseOpening() to kick it off.
  holdOpening?: boolean;
}

export function useChat({ wsUrl = `${WS_BASE}/ws/chat`, userId = 1, userName = "", holdOpening = false }: UseChatOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const mode = "sexting" as const;
  const [isConnected, setIsConnected] = useState(false);
  const [isOpening, setIsOpening] = useState(false);
  const [isCardDelivering, setIsCardDelivering] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  // Delay before her typing bubble appears after a send, so it doesn't pop in
  // instantly (feels more human).
  const typingTimer = useRef<NodeJS.Timeout | null>(null);
  const idCounter = useRef(0);
  const openingAnimating = useRef(false);
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
  const loadHistory = useCallback(async (m: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/history/${m}?user_id=${userId}`);
      if (!res.ok) return;
      const data = await res.json();
      const loaded: ChatMessage[] = (data.messages || []).map((msg: any) => ({
        id: genId(),
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp,
        mode: m,
        // Stored as a relative path ("/uploads/..") — make it absolute against
        // the backend so the <img> resolves regardless of the frontend origin.
        imageUrl: msg.image_url
          ? msg.image_url.startsWith("http")
            ? msg.image_url
            : `${API_BASE}${msg.image_url}`
          : undefined,
      }));

      // Fresh opening: Mia initiated and the user hasn't replied yet
      // (every loaded message is from her). Play it out with the typing
      // indicator and send the bubbles one by one, instead of dumping them.
      const isFreshOpening =
        m === "sexting" &&
        loaded.length > 0 &&
        loaded.every((msg) => msg.role === "assistant");

      if (isFreshOpening) {
        // Guard against double-run (React StrictMode mounts effects twice in
        // dev, which would otherwise append every bubble twice).
        if (openingAnimating.current) return;
        // Already animated once this session (e.g. switched away and back):
        // just show the bubbles statically instead of replaying them.
        if (sextingOpeningPlayed.current) {
          setMessages(loaded);
          return;
        }
        openingAnimating.current = true;
        sextingOpeningPlayed.current = true;
        setIsOpening(true);
        try {
          setMessages([]);
          for (let i = 0; i < loaded.length; i++) {
            setIsTyping(true);
            // Longer pause before the first bubble, a little shorter between the rest
            await new Promise((r) => setTimeout(r, i === 0 ? 2500 : 1300));
            setIsTyping(false);
            setMessages((prev) => [...prev, loaded[i]]);
            // small gap so consecutive bubbles don't appear in the same frame
            if (i < loaded.length - 1) {
              await new Promise((r) => setTimeout(r, 350));
            }
          }
        } finally {
          openingAnimating.current = false;
          setIsOpening(false);
        }
        return;
      }

      setMessages(loaded);
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }, [userId]);

  // Connect WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(`${wsUrl}?user_id=${userId}&user_name=${encodeURIComponent(userName)}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log("WebSocket connected");
      // Ask the backend for the one-time opening — unless the intro popup is
      // up, in which case releaseOpening() sends this on dismiss. Idempotent
      // server-side, so reconnects are safe.
      if (!holdOpeningRef.current) {
        ws.send(JSON.stringify({ type: "start" }));
      }
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "opening_start":
          // Backend is delivering the one-time opening over the socket — lock
          // the input for its whole duration (input is gated on isOpening).
          setIsOpening(true);
          break;
        case "opening_end":
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
            id: genId(),
            role: data.role || "assistant",
            content: data.content,
            timestamp: data.timestamp || Date.now() / 1000,
            mode: data.mode || "sexting",
          };
          setMessages((prev) => [...prev, newMsg]);
          break;
        case "image":
          if (typingTimer.current) { clearTimeout(typingTimer.current); typingTimer.current = null; }
          setIsTyping(false);
          const imageUrl = data.url?.startsWith("http") ? data.url : `${API_BASE}${data.url}`;
          const imgMsg: ChatMessage = {
            id: genId(),
            role: "assistant",
            content: `[image:${imageUrl}]`,
            timestamp: data.timestamp || Date.now() / 1000,
            mode: data.mode || "sexting",
          };
          setMessages((prev) => [...prev, imgMsg]);
          break;
        case "flagged":
          if (typingTimer.current) { clearTimeout(typingTimer.current); typingTimer.current = null; }
          setIsTyping(false);
          setMessages((prev) => [...prev, {
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
      setIsConnected(false);
      console.log("WebSocket disconnected, reconnecting in 3s...");
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
    };
  }, [wsUrl, userId, userName]);

  // Send message
  const sendMessage = useCallback(
    (text: string, imageDataUrl?: string) => {
      if (!text.trim() && !imageDataUrl) return;
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      const now = Date.now() / 1000;

      // Show the user's messages immediately: the photo as its own image
      // bubble (so it actually renders), plus a text bubble if there's text.
      const newMsgs: ChatMessage[] = [];
      if (imageDataUrl) {
        newMsgs.push({
          id: genId(),
          role: "user",
          content: `[image:${imageDataUrl}]`,
          timestamp: now,
          mode,
        });
      }
      if (text.trim()) {
        newMsgs.push({
          id: genId(),
          role: "user",
          content: text,
          timestamp: now,
          mode,
        });
      }
      setMessages((prev) => [...prev, ...newMsgs]);

      // Backend expects raw base64 (no "data:...;base64," prefix) for vision.
      const imageBase64 = imageDataUrl ? imageDataUrl.split(",")[1] : undefined;

      // Send via WebSocket
      wsRef.current.send(
        JSON.stringify({
          type: "message",
          content: text,
          mode,
          image: imageBase64 || undefined,
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
        { id: genId(), role: "user", content: requestText, timestamp: Date.now() / 1000, mode },
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
    connect();
    loadHistory(mode);
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (typingTimer.current) clearTimeout(typingTimer.current);
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connect, loadHistory]);

  return {
    messages,
    isTyping,
    isConnected,
    isOpening,
    isCardDelivering,
    mode,
    sendMessage,
    suggestReply,
    triggerCard,
    releaseOpening,
  };
}
