import { Fragment } from "react";
import { PARTY_ORDER } from "@/lib/parties";
import type { ModelProfile } from "@/lib/llm-study";

/**
 * Diverging heatmap for the LLM writing-style study. Cell colour is a tint of
 * --hm-pos (above baseline) or --hm-neg (below) mixed into the card surface;
 * the signed value is printed in every cell, so direction never rides on
 * colour alone and the grid doubles as its own table view.
 */

// Full tint at the strongest observed deviation, so the scale uses its range.
const MAX_DEV = 0.3;

function cellStyle(v: number): React.CSSProperties {
  const pole = v >= 0 ? "var(--hm-pos)" : "var(--hm-neg)";
  const pct = Math.round(Math.min(1, Math.abs(v) / MAX_DEV) * 55);
  return { backgroundColor: `color-mix(in oklab, ${pole} ${pct}%, var(--card))` };
}

function fmt(v: number): string {
  return (v > 0 ? "+" : "") + v.toFixed(2);
}

export function LlmHeatmap({ rows, label }: { rows: ModelProfile[]; label: string }) {
  return (
    <figure>
      <figcaption className="font-data mb-3 text-[11px] uppercase tracking-widest text-ink-3">
        {label}
      </figcaption>

      <div className="overflow-x-auto">
        <div
          className="grid min-w-[640px] gap-px overflow-hidden rounded-card border border-line bg-line"
          style={{ gridTemplateColumns: "minmax(10rem, auto) repeat(8, minmax(3.2rem, 1fr))" }}
        >
          {/* header row */}
          <div className="bg-card" />
          {PARTY_ORDER.map((p) => (
            <div
              key={p.code}
              className="font-data flex items-center justify-center gap-1.5 bg-card py-2 text-[11px] uppercase tracking-wider text-ink-2"
            >
              <span
                className="h-2 w-2 rounded-[2px]"
                style={{ backgroundColor: `var(${p.colorVar})` }}
                aria-hidden
              />
              {p.code}
            </div>
          ))}

          {/* data rows */}
          {rows.map((m) => (
            <Fragment key={m.id}>
              <div className="flex items-center bg-card py-1.5 pl-3 pr-4 text-[13px] text-ink-2">
                {m.name}
              </div>
              {PARTY_ORDER.map((p) => {
                const v = m.dev[p.code];
                return (
                  <div
                    key={m.id + p.code}
                    className="tabular font-data flex items-center justify-center py-1.5 text-[11px] text-ink"
                    style={cellStyle(v)}
                    title={`${m.name} · ${p.name}: ${fmt(v)} mot baslinjen`}
                  >
                    {fmt(v)}
                  </div>
                );
              })}
            </Fragment>
          ))}
        </div>
      </div>

      {/* scale legend */}
      <div className="mt-3 flex items-center gap-3">
        <span className="font-data text-[10px] uppercase tracking-widest text-ink-3">
          Mindre än baslinjen
        </span>
        <span
          className="h-2 w-36 rounded-full border border-line"
          style={{
            background:
              "linear-gradient(to right, color-mix(in oklab, var(--hm-neg) 55%, var(--card)), var(--card), color-mix(in oklab, var(--hm-pos) 55%, var(--card)))",
          }}
          aria-hidden
        />
        <span className="font-data text-[10px] uppercase tracking-widest text-ink-3">Mer</span>
      </div>
    </figure>
  );
}
