"use client";

import React, { useRef } from "react";
import { Send, Paperclip } from "lucide-react";

interface Props {
  value: string;
  onChange: (text: string) => void;
  onSend: (text: string, imageBase64?: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ value, onChange, onSend, disabled, placeholder }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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

  const handleFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      // Pass the full data URL (keeps the real mime type) so the UI can render
      // a preview; useChat strips the prefix back to raw base64 for the backend.
      onSend(value.trim() || "", reader.result as string);
      onChange("");
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center gap-2 px-4 py-3 bg-[#111118] border-t border-[var(--border)]"
    >
      {/* Attachment */}
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        className="p-2 text-[var(--muted)] hover:text-[var(--accent)] transition-colors"
        title="Send photo"
      >
        <Paperclip size={20} />
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFile}
      />

      {/* Text input */}
      <input
        ref={inputRef}
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
