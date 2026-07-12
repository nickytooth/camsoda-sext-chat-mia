"use client";

import { useEffect, useRef, useState } from "react";
import { useChat } from "./hooks/useChat";
import ChatBubble from "./components/ChatBubble";
import ChatInput from "./components/ChatInput";
import StarterCards from "./components/StarterCards";
import TypingIndicator from "./components/TypingIndicator";
import ProfileSidebar from "./components/ProfileSidebar";
import NameScreen from "./components/NameScreen";
import IntroModal from "./components/IntroModal";
import { API_BASE } from "./api";
import { Circle } from "lucide-react";

// Canned messages sent AS the user when a starter card is clicked. One is
// picked at random so it doesn't read identically every time.
const STARTER_MESSAGES: Record<string, string[]> = {
  fantasy: [
    "tell me one of your fantasies",
    "what's a fantasy you can't stop thinking about?",
    "I want to hear one of your fantasies",
  ],
  story: [
    "tell me a story from your past",
    "tell me about something wild you've done",
    "I want to hear a story about you",
  ],
  lead: [
    "take the lead tonight... I'm all yours",
    "I want you to take the lead",
    "show me what you want",
  ],
};

const PROFILE = {
  name: "Mia",
  tagline:
    "Your best friend's girlfriend — zero filter, zero shame, and zero intention of stopping.",
  bio:
    "Miami party girl, 26, dating Tyler (your best friend). She's been wanting you since his birthday party and doesn't feel a shred of guilt about it. She thinks cheating is hot, sends nudes unprompted, and says exactly what she wants in the most vulgar way possible. That she's not supposed to want you? That's exactly the thrill.",
  profile: {
    age: "26",
    body: "Curvy",
    ethnicity: "European",
    language: "English",
    relationship: "Taken (Tyler's girl)",
    occupation: "Hairdresser / party girl",
    hobbies: "Clubbing, gym, brunch, TikTok",
    personality: "Shameless, bratty, extremely crude",
  },
};

function nameToId(name: string): number {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = (hash * 31 + name.charCodeAt(i)) & 0x7fffffff;
  }
  return Math.max(hash, 1);
}

export default function Home() {
  const [userName, setUserName] = useState<string | null>(null);
  const [userId, setUserId] = useState<number>(1);
  const [ready, setReady] = useState(false);
  // Intro popup: shown right after the name screen — i.e. on a first visit or
  // after a reset (both flow through it). Returning users skip it.
  const [showIntro, setShowIntro] = useState(false);

  // Load saved user from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("mia_user");
    if (saved) {
      const { name, id } = JSON.parse(saved);
      setUserName(name);
      setUserId(id);
    }
    setReady(true);
  }, []);

  const handleNameSubmit = (name: string) => {
    const id = nameToId(name);
    setUserName(name);
    setUserId(id);
    localStorage.setItem("mia_user", JSON.stringify({ name, id }));
    setShowIntro(true);
  };

  const handleReset = async () => {
    if (!userName) return;
    try {
      await fetch(`${API_BASE}/api/reset?user_id=${userId}`, { method: "POST" });
    } catch (e) {}
    localStorage.removeItem("mia_user");
    setUserName(null);
    setUserId(1);
  };

  if (!ready) return null;
  if (!userName) return <NameScreen onSubmit={handleNameSubmit} />;

  return (
    <ChatView
      userName={userName}
      userId={userId}
      onReset={handleReset}
      showIntro={showIntro}
      onIntroClose={() => setShowIntro(false)}
    />
  );
}

function ChatView({
  userName,
  userId,
  onReset,
  showIntro,
  onIntroClose,
}: {
  userName: string;
  userId: number;
  onReset: () => void;
  showIntro: boolean;
  onIntroClose: () => void;
}) {
  const {
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
  } = useChat({ userId, userName, holdOpening: showIntro });

  const scrollRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState("");

  // Cards show a natural-looking request AS the user, then she responds.
  // "Hear a fantasy"/"Hear a story" pull from the (non-repeating) library; "lead" is improvised.
  const handleStarter = (topic: string) => {
    const opts = STARTER_MESSAGES[topic];
    if (!opts || opts.length === 0) return;
    const msg = opts[Math.floor(Math.random() * opts.length)];
    if (topic === "fantasy" || topic === "story") {
      triggerCard(topic, msg);
    } else {
      sendMessage(msg);
    }
  };

  // "Draft my reply" card — fetch an AI suggestion and drop it in the input box.
  const handleDraft = async () => {
    const suggestion = await suggestReply();
    if (suggestion) setDraft(suggestion);
  };

  // Auto-scroll on new messages / typing
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const inputDisabled = isOpening || isCardDelivering;

  // Sexting messages only (the backend persists per mode; this build is sexting-only).
  const visibleMessages = messages.filter((msg) => msg.mode === mode);

  // Read receipts (sexting only): every user message keeps its ticks. A message
  // is "Read" (blue) once she has a reply after it; until then it shows the grey
  // sent/delivered ticks (the bubble animates them in).
  const readByIndex = visibleMessages.map((msg, i) =>
    msg.role === "user" &&
    visibleMessages.slice(i + 1).some((m) => m.role === "assistant")
  );

  return (
    <div className="flex h-screen">
      {/* ---- Chat area ---- */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-3 bg-[#111118] border-b border-[var(--border)]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-purple-600 to-pink-500 flex items-center justify-center text-sm font-bold text-white">
              M
            </div>
            <div>
              <span className="text-[15px] font-semibold text-white">
                Mia
              </span>
              <div className="flex items-center gap-1.5 mt-0.5">
                <Circle
                  size={8}
                  className={`fill-current ${
                    isConnected ? "text-green-400" : "text-red-400"
                  }`}
                />
                <span className="text-[11px] text-[var(--muted)]">
                  {isTyping
                    ? "typing..."
                    : isConnected
                    ? "online"
                    : "offline"}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onReset}
              className="px-3 py-1.5 text-[11px] rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors"
            >
              Reset
            </button>
          </div>
        </div>

        {/* Messages */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto py-4 bg-[var(--chat-bg)]"
        >
          {visibleMessages.length === 0 && (
            <div className="flex items-center justify-center h-full text-[var(--muted)] text-sm">
              Start a conversation...
            </div>
          )}
          {visibleMessages.map((msg, i) => (
            <ChatBubble
              key={msg.id}
              message={msg}
              showReceipt={true}
              read={readByIndex[i]}
            />
          ))}
          {isTyping && <TypingIndicator />}
        </div>

        {/* Conversation-starter cards */}
        <StarterCards
          onStarter={handleStarter}
          onDraft={handleDraft}
          disabled={inputDisabled || isTyping}
        />

        {/* Input */}
        <ChatInput
          value={draft}
          onChange={setDraft}
          onSend={sendMessage}
          disabled={inputDisabled}
          placeholder="Write a message..."
        />
      </div>

      {/* ---- Sidebar ---- */}
      <ProfileSidebar
        name={PROFILE.name}
        tagline={PROFILE.tagline}
        bio={PROFILE.bio}
        profile={PROFILE.profile}
      />

      {/* Intro popup — dismissing it kicks off Mia's opening messages */}
      {showIntro && (
        <IntroModal
          onClose={() => {
            onIntroClose();
            releaseOpening();
          }}
        />
      )}
    </div>
  );
}
