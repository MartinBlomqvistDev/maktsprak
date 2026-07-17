"use client";

import { useEffect, useRef, useState } from "react";
import { YEARS, yearSignatures } from "@/lib/site-data";

// One-line gloss per year. Every caption must be grounded in that year's actual
// top distinctive words (data/site/year_signatures.json) — the caption sits
// directly beside the word list, so a claim the words don't support reads as
// the site contradicting its own data. RE-VERIFY these against the words
// whenever the corpus is rebuilt; they are hand-written and will otherwise drift.
const CAPTIONS: Record<number, string> = {
  2002: "EMU-folkomröstningen. Maxtaxa och buffertfonder.",
  2003: "EMU och Irakkriget. Saddam Hussein.",
  2005: "Stormen Gudrun. Apatiska barn och amnestikravet.",
  2006: "Alliansen vinner. Fastighetsskatten och Citybanan.",
  2008: "Lissabonfördraget och vårdnadsbidraget. Varslen börjar.",
  2009: "Finanskrisen. Bonusar och lågkonjunktur.",
  2010: "De rödgröna mot Alliansen. Vattenfall och Arbetsförmedlingen.",
  2011: "Arabiska våren. Libyen, Egypten, Gaddafi.",
  2013: "Jobbskatteavdraget och arbetslösheten.",
  2014: "Bromma flygplats och jobben. Förbifarten.",
  2015: "RUT, ROT och jobben. Bromma och Paris.",
  2016: "Flyktingmottagandet. Nyanlända, Daish, ensamkommande.",
  2017: "Metoo. Nyanlända, poliser och flygskatt.",
  2018: "Lång regeringsbildning. CETA och barnäktenskap.",
  2019: "Januariavtalet präglar allt. Värnskatten slopas. Brexit.",
  2020: "Pandemin slår in med full kraft. Nära nog inget annat.",
  2021: "Vaccin och restriktioner. Pandemin fortsätter.",
  2022: "Rysslands invasion av Ukraina. Tidöavtalet efter valet.",
  2023: "Inflationen och elstödet. Tidöavtalet.",
  2024: "Gaza och Ukraina parallellt. Kärnvapenfrågan tillbaka.",
  2025: "Gaza och UNRWA. Ukraina och gängkriminaliteten.",
  2026: "Ukraina och Tidöregeringen. Förvarsfrågan och alunskiffern.",
};

const MAX_WORDS = 10;
const PLAY_INTERVAL_MS = 1300;

export function YearSlider() {
  const [yearIdx, setYearIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const year = YEARS[yearIdx];
  const words = (yearSignatures[String(year)] ?? []).slice(0, MAX_WORDS);
  const maxZ = Math.max(...words.map((w) => w.z), 1);

  useEffect(() => {
    if (!playing) return;
    timerRef.current = setInterval(() => {
      setYearIdx((i) => {
        if (i >= YEARS.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, PLAY_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing]);

  function stopAndSeek(idx: number) {
    setPlaying(false);
    setYearIdx(idx);
  }

  function togglePlay() {
    if (!playing && yearIdx >= YEARS.length - 1) setYearIdx(0);
    setPlaying((p) => !p);
  }

  // Nearest caption at or before the current year.
  const caption =
    CAPTIONS[year] ??
    CAPTIONS[
      Object.keys(CAPTIONS)
        .map(Number)
        .filter((y) => y <= year)
        .sort((a, b) => b - a)[0]
    ];

  return (
    <div className="overflow-hidden rounded-[var(--radius-lg)] border border-line bg-card shadow-sm">
      <div className="flex items-end gap-5 px-6 pb-2 pt-7 sm:px-8">
        <div className="tabular font-display text-[3.2rem] leading-none tracking-tight sm:text-[4.5rem]">
          {year}
        </div>
        <p className="mb-1 min-h-10 max-w-md text-sm italic leading-snug text-ink-2">
          {caption}
        </p>
      </div>

      <div className="flex items-center gap-4 px-6 pb-6 pt-3 sm:px-8">
        <button
          type="button"
          onClick={togglePlay}
          className="font-data shrink-0 whitespace-nowrap rounded-[var(--radius)] bg-accent px-4 py-2 text-[11px] uppercase tracking-widest text-accent-ink transition-opacity hover:opacity-90"
        >
          {playing ? "❚❚ Pausa" : "► Spela upp"}
        </button>
        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={YEARS.length - 1}
            step={1}
            value={yearIdx}
            onChange={(e) => stopAndSeek(Number(e.target.value))}
            className="w-full accent-[var(--accent)]"
            aria-label="Välj år"
          />
          <div className="font-data mt-1.5 flex justify-between text-[10px] uppercase tracking-widest text-ink-3">
            <span>{YEARS[0]}</span>
            <span>{YEARS[YEARS.length - 1]}</span>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-1 border-t border-line px-6 py-6 sm:px-8">
        {words.length === 0 && (
          <p className="text-sm text-ink-3">Ingen data för {year}.</p>
        )}
        {words.map((w) => (
          <div key={w.word} className="grid grid-cols-[9rem_1fr] items-center gap-3">
            <span className="font-data truncate text-right text-sm">{w.word}</span>
            <div className="h-3">
              <div
                className="h-3 rounded-[1px] bg-accent transition-[width] duration-500 ease-out"
                style={{ width: `${Math.max(4, (w.z / maxZ) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
