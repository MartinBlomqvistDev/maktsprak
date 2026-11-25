"use client";

import { useId, useMemo, useRef, useState } from "react";
import { fold } from "./format";

/** One entry of /fraga/talare.json, written by search/pipeline/15_build_page.py. */
export interface Speaker {
  id: string;
  name: string;
  parties: string[];
  first: number;
  last: number;
  speeches: number;
}

const MAX_OPTIONS = 8;

let cached: Promise<Speaker[]> | null = null;

/** The speaker list, fetched once per page load and only when the field is used. */
function loadSpeakers(): Promise<Speaker[]> {
  cached ??= fetch("/fraga/talare.json")
    .then((response) => {
      if (!response.ok) throw new Error(`talare.json: ${response.status}`);
      return response.json() as Promise<Speaker[]>;
    })
    .catch((error) => {
      cached = null;
      throw error;
    });
  return cached;
}

function years(speaker: Speaker): string {
  return speaker.first === speaker.last ? `${speaker.first}` : `${speaker.first}-${speaker.last}`;
}

/**
 * A combobox over the speakers in the corpus: type part of a name, pick one.
 *
 * The search filters by the Riksdag's person_id, so a member keeps one entry
 * whatever title a speech gives them. The parent owns the chosen speaker; give
 * the picker a new `key` to clear what has been typed.
 */
export function SpeakerPicker({
  value,
  onChange,
  disabled,
}: {
  value: Speaker | null;
  onChange: (speaker: Speaker | null) => void;
  disabled?: boolean;
}) {
  const [typed, setTyped] = useState("");
  const text = value ? value.name : typed;
  const [speakers, setSpeakers] = useState<Speaker[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();
  const inputRef = useRef<HTMLInputElement>(null);

  const matches = useMemo(() => {
    if (!speakers) return [];
    const typed = fold(text.trim());
    if (!typed) return [];
    const parts = typed.split(/\s+/);
    return speakers
      .filter((speaker) => {
        const name = fold(speaker.name);
        return parts.every((part) => name.includes(part));
      })
      .slice(0, MAX_OPTIONS);
  }, [speakers, text]);

  function ensureLoaded() {
    if (speakers || failed) return;
    loadSpeakers()
      .then(setSpeakers)
      .catch(() => setFailed(true));
  }

  function choose(speaker: Speaker) {
    onChange(speaker);
    setTyped("");
    setOpen(false);
  }

  function clear() {
    onChange(null);
    setTyped("");
    setOpen(false);
    inputRef.current?.focus();
  }

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-expanded={open && matches.length > 0}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={open && matches[active] ? `${listId}-${active}` : undefined}
          value={text}
          disabled={disabled}
          placeholder={failed ? "Talarlistan gick inte att läsa in" : "Alla talare"}
          onFocus={ensureLoaded}
          onChange={(event) => {
            setTyped(event.target.value);
            setOpen(true);
            setActive(0);
            if (value) onChange(null);
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setOpen(true);
              setActive((i) => Math.min(i + 1, Math.max(matches.length - 1, 0)));
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              setActive((i) => Math.max(i - 1, 0));
            } else if (event.key === "Enter" && open && matches[active]) {
              event.preventDefault();
              choose(matches[active]);
            } else if (event.key === "Escape") {
              setOpen(false);
            }
          }}
          onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          className="w-full rounded-sm border border-line-2 bg-paper px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:border-accent focus:outline-none disabled:opacity-50"
        />
        {value && !disabled && (
          <button
            type="button"
            onClick={clear}
            className="shrink-0 text-xs text-accent underline underline-offset-2"
          >
            Rensa
          </button>
        )}
      </div>
      {open && matches.length > 0 && (
        <ul
          id={listId}
          role="listbox"
          className="absolute inset-x-0 top-full z-20 mt-1 max-h-72 overflow-auto rounded-sm border border-line-2 bg-card py-1 shadow-sm"
        >
          {matches.map((speaker, i) => (
            <li
              key={speaker.id}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              onMouseDown={(event) => {
                event.preventDefault();
                choose(speaker);
              }}
              onMouseEnter={() => setActive(i)}
              className={`cursor-pointer px-3 py-2 text-sm ${i === active ? "bg-accent-soft text-accent-soft-ink" : "text-ink"}`}
            >
              <span className="font-medium">{speaker.name}</span>{" "}
              <span className="text-ink-3">
                ({speaker.parties.join(", ")}) {years(speaker)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
