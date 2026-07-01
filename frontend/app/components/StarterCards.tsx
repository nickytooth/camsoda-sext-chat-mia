"use client";

import React, { useState } from "react";
import { Flame, BookOpen, Crown, Sparkles, Loader2 } from "lucide-react";

interface Props {
  onStarter: (topic: string) => void;
  onDraft: () => Promise<void>;
  disabled?: boolean;
}

type Card = {
  key: string;
  label: string;
  icon: React.ReactNode;
  action: "starter" | "draft";
};

const CARDS: Card[] = [
  { key: "fantasy", label: "Hear a fantasy", icon: <Flame size={15} />, action: "starter" },
  { key: "story", label: "Hear a story", icon: <BookOpen size={15} />, action: "starter" },
  { key: "lead", label: "Let her lead", icon: <Crown size={15} />, action: "starter" },
  { key: "draft", label: "Draft my reply", icon: <Sparkles size={15} />, action: "draft" },
];

export default function StarterCards({ onStarter, onDraft, disabled }: Props) {
  const [busyKey, setBusyKey] = useState<string | null>(null);

  const handleClick = async (card: Card) => {
    if (disabled || busyKey) return;
    setBusyKey(card.key);
    try {
      if (card.action === "draft") {
        await onDraft();
      } else {
        onStarter(card.key);
      }
    } finally {
      setBusyKey(null);
    }
  };

  return (
    <div className="flex gap-2 overflow-x-auto px-4 py-2 bg-[var(--chat-bg)] border-t border-[var(--border)]">
      {CARDS.map((card) => (
        <button
          key={card.key}
          type="button"
          onClick={() => handleClick(card)}
          disabled={disabled || busyKey !== null}
          title={card.label}
          className="flex items-center gap-1.5 shrink-0 px-3 py-1.5 rounded-full text-[13px] text-[var(--foreground)] bg-[#1a1a2e] border border-[var(--border)] hover:bg-[#2d1b4e] hover:border-[var(--accent)]/50 hover:text-[var(--accent-pink)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {busyKey === card.key ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            card.icon
          )}
          <span className="whitespace-nowrap">{card.label}</span>
        </button>
      ))}
    </div>
  );
}
