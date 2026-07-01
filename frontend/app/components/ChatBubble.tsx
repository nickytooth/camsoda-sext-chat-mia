"use client";

import React, { useState, useEffect } from "react";
import { Check, CheckCheck } from "lucide-react";
import { ChatMessage } from "../hooks/useChat";

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

interface Props {
  message: ChatMessage;
  showReceipt?: boolean;
  read?: boolean;
}

export default function ChatBubble({ message, showReceipt, read }: Props) {
  // System messages (moderation flags) — centered grey bubble, no avatar
  if (message.role === "system") {
    return (
      <div className="flex justify-center mb-3 px-4">
        <div className="bg-[var(--border)]/40 text-[var(--muted)] text-[12px] text-center rounded-lg px-3 py-2 max-w-[80%]">
          {message.content}
        </div>
      </div>
    );
  }

  const isUser = message.role === "user";

  // WhatsApp-style read receipts (sexting only), shown under EVERY user message
  // and kept there. A fresh message animates: one grey tick the instant it's
  // sent, then after ~1s flips to blue "Read" with double ticks — right as her
  // typing bubble appears. Already-read history messages render blue immediately.
  const [autoRead, setAutoRead] = useState(false);
  useEffect(() => {
    if (!isUser || !showReceipt || read) return;
    setAutoRead(false);
    const t = setTimeout(() => setAutoRead(true), 1000);
    return () => clearTimeout(t);
  }, [isUser, showReceipt, read, message.id]);

  const isRead = read || autoRead;

  let receipt: React.ReactNode = null;
  if (isUser && showReceipt) {
    receipt = isRead ? (
      <span className="inline-flex items-center gap-1">
        <span className="text-sky-400 font-medium">Read</span>
        <span>{formatTime(message.timestamp)}</span>
        <CheckCheck size={14} className="text-sky-400" />
      </span>
    ) : (
      <span className="inline-flex items-center gap-1">
        <span>{formatTime(message.timestamp)}</span>
        <Check size={14} className="text-[var(--muted)]" />
      </span>
    );
  }

  // A persisted upload (imageUrl, from history) or a live preview bubble whose
  // content is an [image:...] marker — both render as an image.
  const isImage = !!message.imageUrl || message.content.startsWith("[image:");

  if (isImage) {
    const url = message.imageUrl ?? message.content.slice(7, -1);

    return (
      <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3 px-4`}>
        <div className="max-w-[320px]">
          <div className="relative overflow-hidden rounded-2xl border border-[var(--border)]">
            <img
              src={url}
              alt=""
              className="block w-full"
            />
          </div>
          <div className={`text-[11px] text-[var(--muted)] mt-1 flex items-center gap-1 ${isUser ? "justify-end" : "justify-start"}`}>
            {receipt ?? formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3 px-4`}>
      <div className="flex flex-col max-w-[70%]">
        {/* Avatar for her messages */}
        {!isUser && (
          <div className="flex items-end gap-2">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-600 to-pink-500 flex-shrink-0 flex items-center justify-center text-[10px] font-bold text-white">
              V
            </div>
            <div className="bg-[var(--her-bubble)] rounded-2xl rounded-bl-md px-4 py-2.5 text-[14px] leading-relaxed">
              {message.content}
            </div>
          </div>
        )}

        {/* User messages */}
        {isUser && (
          <div className="bg-[var(--user-bubble)] rounded-2xl rounded-br-md px-4 py-2.5 text-[14px] leading-relaxed">
            {message.content}
          </div>
        )}

        {/* Timestamp */}
        <div
          className={`text-[11px] text-[var(--muted)] mt-1 flex items-center gap-1 ${
            isUser ? "justify-end pr-1" : "justify-start pl-9"
          }`}
        >
          {receipt ?? formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
}
