"use client";

import React from "react";
import { Send } from "lucide-react";

interface Props {
  value: string;
  onChange: (text: string) => void;
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ value, onChange, onSend, disabled, placeholder }: Props) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim() || disabled) return;
    onSend(value.trim());
    onChange("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center gap-2 px-4 py-3 bg-[#111118] border-t border-[var(--border)]"
    >
      {/* Text input */}
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder || "Write a message..."}
        disabled={disabled}
        className="flex-1 bg-[#1a1a2e] text-[var(--foreground)] placeholder-[var(--muted)] rounded-xl px-4 py-2.5 text-[14px] outline-none focus:ring-1 focus:ring-[var(--accent)]/50 disabled:opacity-50"
      />

      {/* Send */}
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="p-2.5 bg-gradient-to-r from-purple-600 to-pink-500 rounded-xl text-white disabled:opacity-30 hover:opacity-90 transition-opacity"
      >
        <Send size={18} />
      </button>
    </form>
  );
}
