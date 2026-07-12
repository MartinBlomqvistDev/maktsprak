"use client";

import { useEffect, useState } from "react";

export interface RedactSegment {
  text: string;
  /** If true, this segment renders as a black bar until its turn to reveal. */
  redact?: boolean;
}

interface RedactRevealProps {
  segments: RedactSegment[];
  /** Delay before the first segment declassifies, in ms. */
  startDelayMs?: number;
  /** Gap between each subsequent reveal, in ms. */
  staggerMs?: number;
  /** Fired once every redacted segment has been revealed. */
  onComplete?: () => void;
  className?: string;
}

/**
 * The site's signature moment: a sentence that loads with key phrases
 * redacted, then declassifies word-group by word-group — performing, quite
 * literally, what the model does to political language. Instant + fully
 * visible when prefers-reduced-motion is set.
 */
export function RedactReveal({
  segments,
  startDelayMs = 500,
  staggerMs = 260,
  onComplete,
  className,
}: RedactRevealProps) {
  const redactableCount = segments.filter((s) => s.redact).length;
  const [revealed, setRevealed] = useState(0);

  useEffect(() => {
    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    if (reduceMotion || redactableCount === 0) {
      // Deciding this requires matchMedia (browser-only), so it can't be a
      // useState initializer — same justified exception as the theme toggle.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setRevealed(redactableCount);
      onComplete?.();
      return;
    }

    const timers: ReturnType<typeof setTimeout>[] = [];
    for (let i = 1; i <= redactableCount; i++) {
      timers.push(
        setTimeout(() => {
          setRevealed(i);
          if (i === redactableCount) onComplete?.();
        }, startDelayMs + i * staggerMs)
      );
    }
    return () => timers.forEach(clearTimeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  let redactIndex = 0;

  return (
    <span className={className}>
      {segments.map((seg, i) => {
        if (!seg.redact) return <span key={i}>{seg.text}</span>;
        const myIndex = ++redactIndex;
        const isRevealed = myIndex <= revealed;
        return (
          <span
            key={i}
            className={`redacted inline-block px-1 transition-[color,background-color] duration-300 ${
              isRevealed ? "!bg-transparent !text-inherit" : ""
            }`}
            aria-hidden={!isRevealed || undefined}
          >
            {seg.text}
          </span>
        );
      })}
    </span>
  );
}
