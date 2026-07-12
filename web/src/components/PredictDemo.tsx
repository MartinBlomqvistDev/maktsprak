"use client";

import { useState } from "react";
import { PredictionBars } from "./PredictionBars";

type Status = "idle" | "loading" | "done" | "not-connected" | "error";

const PLACEHOLDER =
  "Klistra in ett citat, ett pressmeddelande eller en riksdagsreplik…";

export function PredictDemo() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [probs, setProbs] = useState<Partial<Record<string, number>> | null>(
    null
  );

  async function handleSubmit() {
    if (!text.trim()) return;
    setStatus("loading");
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (res.status === 501) {
        setStatus("not-connected");
        return;
      }
      if (!res.ok) {
        setStatus("error");
        return;
      }
      const data = await res.json();
      setProbs(data.probabilities ?? data);
      setStatus("done");
    } catch {
      setStatus("error");
    }
  }

  return (
    <div className="rounded-card border border-line bg-card p-5 sm:p-6">
      <div className="anf-marker mb-4">
        <span className="tabular">ANF. LIVE</span>
        <span className="text-line-2">·</span>
        <b>TESTA MODELLEN</b>
        <span className="h-px flex-1 bg-line" aria-hidden="true" />
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={PLACEHOLDER}
        rows={4}
        className="w-full resize-none rounded-sm border border-line-2 bg-paper px-4 py-3 font-body text-[15px] text-ink placeholder:text-ink-3 focus:border-accent focus:outline-none"
      />

      <div className="mt-3 flex items-center justify-between gap-4">
        <p className="text-xs text-ink-3">
          Texten skickas till modellen och sparas inte.
        </p>
        <button
          onClick={handleSubmit}
          disabled={status === "loading" || !text.trim()}
          className="shrink-0 rounded-sm bg-accent px-5 py-2 font-data text-xs uppercase tracking-widest text-accent-ink transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {status === "loading" ? "Analyserar…" : "Avkoda"}
        </button>
      </div>

      {status === "done" && probs && (
        <div className="mt-6 border-t border-line pt-5">
          <p className="mb-3 font-data text-[11px] uppercase tracking-widest text-ink-3">
            Läses som —
          </p>
          <PredictionBars probabilities={probs} />
        </div>
      )}

      {status === "not-connected" && (
        <div className="mt-6 rounded-sm border border-dashed border-line-2 bg-paper-2 px-4 py-3">
          <p className="text-sm text-ink-2">
            <span className="font-data uppercase text-warn">Ej ansluten ännu</span>
            {" — "}
            inferens-tjänsten publiceras i nästa fas av ombyggnationen. Se{" "}
            <a href="/metod" className="text-accent underline underline-offset-2">
              metodsidan
            </a>{" "}
            för status.
          </p>
        </div>
      )}

      {status === "error" && (
        <p className="mt-4 text-sm text-crit">
          Något gick fel. Försök igen om en stund.
        </p>
      )}
    </div>
  );
}
