"use client";

// One-time intro shown on a fresh chat (first visit or right after a reset —
// both flow through the name screen). Explains who Mia is before her opening
// messages start rolling in behind the overlay.
export default function IntroModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-2xl bg-[#111118] border border-[var(--border)] p-8 text-center shadow-2xl">
        <div className="mx-auto mb-5 w-16 h-16 rounded-full bg-gradient-to-br from-purple-600 to-pink-500 flex items-center justify-center text-2xl font-bold text-white">
          M
        </div>

        <h2 className="text-xl font-semibold text-white mb-1">Meet Mia</h2>
        <p className="text-sm text-pink-400/90 mb-4">
          Your best friend&apos;s girlfriend
        </p>

        <p className="text-[14px] leading-relaxed text-gray-300 mb-6">
          Zero filter, and more than a little naughty. The two of you have been
          trading looks for way too long &mdash; maybe it&apos;s finally time to
          say things the way they are.
          <br />
          <span className="text-gray-500 text-[13px]">
            What happens here, stays here. 😏
          </span>
        </p>

        <button
          onClick={onClose}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-purple-600 to-pink-500 text-white text-sm font-semibold hover:opacity-90 transition-opacity"
        >
          Say hi to Mia
        </button>
      </div>
    </div>
  );
}
