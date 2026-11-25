"use client";

import { useState } from "react";

/**
 * A long quotation folded to its first lines, with a button that shows the
 * rest. The text is all there either way; only its height changes.
 */
export function Clamp({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <div className={open ? "" : "relative max-h-56 overflow-hidden"}>
        {children}
        {!open && (
          <div
            className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-card to-transparent"
            aria-hidden="true"
          />
        )}
      </div>
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="mt-2 text-xs text-accent underline underline-offset-2"
      >
        {open ? "Visa mindre" : "Visa hela citatet"}
      </button>
    </div>
  );
}
