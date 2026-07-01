"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";

type Slide = { type: "video" | "image"; src: string };

// Slide 1 is the video; the rest are photos. Drop the files in frontend/public/.
const SLIDES: Slide[] = [
  { type: "video", src: "/victoria.mp4" },
  { type: "image", src: "/victoria1.png" },
  { type: "image", src: "/victoria2.png" },
  { type: "image", src: "/victoria3.png" },
];

export default function ProfileGallery() {
  const [index, setIndex] = useState(0);
  const [failed, setFailed] = useState<Record<number, boolean>>({});
  const videoRef = useRef<HTMLVideoElement>(null);

  const go = (i: number) => setIndex((i + SLIDES.length) % SLIDES.length);

  // Whenever the video slide becomes active, play it from the start.
  useEffect(() => {
    const v = videoRef.current;
    if (v && SLIDES[index].type === "video") {
      v.currentTime = 0;
      v.play().catch(() => {});
    }
  }, [index]);

  // When the video ends, freeze on the first frame as a thumbnail.
  const handleEnded = () => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = 0;
    v.pause();
  };

  // Hovering the video replays it from the start.
  const handleHoverVideo = () => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = 0;
    v.play().catch(() => {});
  };

  const markFailed = (i: number) =>
    setFailed((f) => ({ ...f, [i]: true }));

  const slide = SLIDES[index];

  return (
    <div className="relative">
      <div className="w-full aspect-[3/4] bg-gradient-to-b from-purple-900/40 to-black flex items-center justify-center overflow-hidden">
        {failed[index] ? (
          <div className="w-32 h-32 rounded-full bg-gradient-to-br from-purple-600 to-pink-500 flex items-center justify-center text-4xl font-bold text-white">
            V
          </div>
        ) : slide.type === "video" ? (
          <video
            ref={videoRef}
            src={slide.src}
            className="w-full h-full object-cover"
            autoPlay
            muted
            playsInline
            preload="auto"
            onEnded={handleEnded}
            onMouseEnter={handleHoverVideo}
            onError={() => markFailed(index)}
          />
        ) : (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={slide.src}
            alt=""
            className="w-full h-full object-cover"
            onError={() => markFailed(index)}
          />
        )}
      </div>

      {/* Arrows */}
      <button
        type="button"
        onClick={() => go(index - 1)}
        aria-label="Previous"
        className="absolute left-2 top-1/2 -translate-y-1/2 p-1.5 rounded-full bg-black/45 text-white hover:bg-black/75 transition-colors"
      >
        <ChevronLeft size={20} />
      </button>
      <button
        type="button"
        onClick={() => go(index + 1)}
        aria-label="Next"
        className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-full bg-black/45 text-white hover:bg-black/75 transition-colors"
      >
        <ChevronRight size={20} />
      </button>

      {/* Dots */}
      <div className="absolute bottom-3 left-0 right-0 flex justify-center gap-1.5">
        {SLIDES.map((_, i) => (
          <button
            key={i}
            type="button"
            onClick={() => go(i)}
            aria-label={`Go to slide ${i + 1}`}
            className={`w-2 h-2 rounded-full transition-colors ${
              i === index ? "bg-white" : "bg-white/40 hover:bg-white/70"
            }`}
          />
        ))}
      </div>
    </div>
  );
}
