"use client";

import { useState } from "react";

interface Props {
  onSubmit: (name: string) => void;
}

export default function NameScreen({ onSubmit }: Props) {
  const [name, setName] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (name.trim()) {
      onSubmit(name.trim());
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-[#0d0d0d]">
      <form
        onSubmit={handleSubmit}
        className="flex flex-col items-center gap-6 p-8 rounded-2xl bg-[#111118] border border-[var(--border)] max-w-sm w-full mx-4"
      >
        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-600 to-pink-500 flex items-center justify-center text-3xl font-bold text-white">
          V
        </div>
        <div className="text-center">
          <h1 className="text-xl font-bold text-white">Victoria Donovan</h1>
          <p className="text-[13px] text-[var(--muted)] mt-1">
            Enter your name to start chatting
          </p>
        </div>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Your name..."
          autoFocus
          className="w-full bg-[#1a1a2e] text-[var(--foreground)] placeholder-[var(--muted)] rounded-xl px-4 py-3 text-[14px] outline-none focus:ring-1 focus:ring-[var(--accent)]/50 text-center"
        />
        <button
          type="submit"
          disabled={!name.trim()}
          className="w-full py-3 bg-gradient-to-r from-purple-600 to-pink-500 rounded-xl text-white font-medium disabled:opacity-30 hover:opacity-90 transition-opacity"
        >
          Start Chat
        </button>
      </form>
    </div>
  );
}
