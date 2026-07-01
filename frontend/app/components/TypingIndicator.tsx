"use client";

export default function TypingIndicator() {
  return (
    <div className="flex justify-start mb-3 px-4">
      <div className="flex items-end gap-2">
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-600 to-pink-500 flex-shrink-0 flex items-center justify-center text-[10px] font-bold text-white">
          V
        </div>
        <div className="bg-[var(--her-bubble)] rounded-2xl rounded-bl-md px-4 py-3 flex gap-1.5 items-center">
          <span className="w-2 h-2 bg-purple-300/60 rounded-full animate-bounce [animation-delay:0ms]" />
          <span className="w-2 h-2 bg-purple-300/60 rounded-full animate-bounce [animation-delay:150ms]" />
          <span className="w-2 h-2 bg-purple-300/60 rounded-full animate-bounce [animation-delay:300ms]" />
        </div>
      </div>
    </div>
  );
}
