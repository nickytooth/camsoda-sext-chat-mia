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
import MediaOfferBubble from "./components/MediaOfferBubble";
import UnlockedGallery from "./components/UnlockedGallery";
import { API_BASE } from "./api";
import { useCommerce } from "./hooks/useCommerce";
import { Circle, Coins, Images, Loader2, Plus } from "lucide-react";

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
    "Miami party girl, 26, dating Tyler (your best friend). She's been wanting you since his birthday party and doesn't feel a shred of guilt about it. She thinks cheating is hot, keeps private photos and videos for the right moment, and says exactly what she wants with zero filter. That she's not supposed to want you? That's exactly the thrill.",
  profile: {
    age: "26",
    body: "Curvy",
    ethnicity: "European",
    language: "English",
    relationship: "Taken (Tyler's girl)",
    occupation: "Part-time bartender / party girl",
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
    const hydrationTimer = window.setTimeout(() => {
      const saved = localStorage.getItem("mia_user");
      if (saved) {
        const { name, id } = JSON.parse(saved);
        setUserName(name);
        setUserId(id);
      }
      setReady(true);
    }, 0);
    return () => window.clearTimeout(hydrationTimer);
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
    } catch {}
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
    commerceRevision,
    mode,
    sendMessage,
    suggestReply,
    triggerCard,
    releaseOpening,
    refreshMediaOffer,
  } = useChat({ userId, userName, holdOpening: showIntro });

  const commerce = useCommerce(userId, commerceRevision);

  const scrollRef = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState("");
  const [galleryOpen, setGalleryOpen] = useState(false);

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

  // Never clear a user's draft into a closed socket. The controls become
  // available automatically as soon as the reconnect succeeds.
  const inputDisabled = !isConnected || isOpening || isCardDelivering;

  // Sexting messages only (the backend persists per mode; this build is sexting-only).
  const visibleMessages = messages.filter((msg) => msg.mode === mode);

  // Read receipts (sexting only): every user message keeps its ticks. A message
  // is "Read" (blue) once she has a reply after it; until then it shows the grey
  // sent/delivered ticks (the bubble animates them in).
  const readByIndex = visibleMessages.map((msg, i) =>
    msg.type === "text" &&
    msg.role === "user" &&
    visibleMessages.slice(i + 1).some((m) => m.role === "assistant")
  );

  return (
    <div className="flex h-dvh">
      {/* ---- Chat area ---- */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] bg-[#111118] px-3 py-3 sm:flex-nowrap sm:px-4">
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

          <div className="flex min-w-0 w-full items-center justify-between gap-1.5 sm:w-auto sm:justify-start sm:gap-2">
            <div
              className="flex h-8 items-center gap-1.5 rounded-lg border border-amber-300/15 bg-amber-300/8 px-2.5 text-xs font-semibold text-amber-100"
              title={commerce.walletError || "Demo token balance"}
            >
              <Coins size={15} className="text-amber-300" />
              <span>{commerce.balance === null ? "—" : commerce.balance.toLocaleString()}</span>
            </div>
            <button
              type="button"
              onClick={() => void commerce.refill()}
              disabled={commerce.isRefilling}
              className="inline-flex h-8 items-center gap-1 rounded-lg bg-purple-500/12 px-2.5 text-[11px] font-medium text-purple-200 transition-colors hover:bg-purple-500/20 disabled:cursor-wait disabled:opacity-60"
              aria-label="Refill 1000 demo tokens"
            >
              {commerce.isRefilling ? <Loader2 size={13} className="animate-spin" /> : <Plus size={13} />}
              <span className="hidden sm:inline">Refill </span>+1000
            </button>
            <button
              type="button"
              onClick={() => setGalleryOpen(true)}
              className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-white/5 px-2.5 text-[11px] font-medium text-white/75 transition-colors hover:bg-white/10 hover:text-white"
              aria-label={`Open Unlocked Gallery with ${commerce.gallery.length} items`}
            >
              <Images size={14} />
              <span className="hidden md:inline">Gallery</span>
              {commerce.gallery.length > 0 && (
                <span className="rounded-full bg-pink-500/20 px-1.5 py-0.5 text-[10px] text-pink-200">
                  {commerce.gallery.length}
                </span>
              )}
            </button>
            <button
              onClick={onReset}
              className="h-8 rounded-lg bg-red-500/10 px-2.5 text-[11px] text-red-400 transition-colors hover:bg-red-500/20 sm:px-3"
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
          {visibleMessages.map((msg, i) =>
            msg.type === "media_offer" ? (
              <MediaOfferBubble
                key={msg.id}
                message={msg}
                balance={commerce.balance}
                unlocked={
                  msg.offer.unlocked ||
                  commerce.unlockedContentIds.has(msg.offer.content_id)
                }
                unlocking={commerce.unlocking.has(msg.offer.offer_id)}
                error={commerce.unlockErrors[msg.offer.offer_id]}
                onUnlock={commerce.unlock}
                getAccess={commerce.getAccess}
                onRefreshPreview={refreshMediaOffer}
              />
            ) : (
              <ChatBubble
                key={msg.id}
                message={msg}
                showReceipt={true}
                read={readByIndex[i]}
              />
            ),
          )}
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
          placeholder={isConnected ? "Write a message..." : "Connecting..."}
        />
      </div>

      {/* ---- Sidebar ---- */}
      <div className="hidden xl:block">
        <ProfileSidebar
          name={PROFILE.name}
          tagline={PROFILE.tagline}
          bio={PROFILE.bio}
          profile={PROFILE.profile}
        />
      </div>

      {galleryOpen && (
        <UnlockedGallery
          items={commerce.gallery}
          error={commerce.galleryError}
          getAccess={commerce.getAccess}
          onClose={() => setGalleryOpen(false)}
        />
      )}

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
