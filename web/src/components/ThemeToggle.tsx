"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";

function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    // Reading localStorage/matchMedia requires the browser, so this can't be
    // a useState initializer (SSR has no window), the one legitimate case
    // for setState-in-effect, same pattern libraries like next-themes use.
    const stored = window.localStorage.getItem("maktsprak-theme") as Theme | null;
    const initial = stored ?? systemTheme();
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTheme(initial);
    document.documentElement.setAttribute("data-theme", initial);
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    window.localStorage.setItem("maktsprak-theme", next);
  }

  return (
    <button
      onClick={toggle}
      aria-label="Växla mellan ljust och mörkt tema"
      className="font-data flex items-center gap-2 rounded-full border border-line-2 px-3 py-1.5 text-[11px] uppercase tracking-widest text-ink-2 transition-colors hover:border-accent hover:text-accent"
    >
      <span
        className="h-1.5 w-1.5 rounded-full bg-accent"
        aria-hidden="true"
      />
      {theme === "dark" ? "Mörkt" : "Ljust"}
    </button>
  );
}
