"use client";

import { useMemo, useState } from "react";
import {
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PARTY_ORDER } from "@/lib/parties";
import { partyFrames } from "@/lib/site-data";

interface TooltipPayloadItem {
  color?: string;
  name?: string;
  value?: number;
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: string | number;
}) {
  if (!active || !payload?.length) return null;
  const sorted = [...payload].sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
  return (
    <div className="rounded-[var(--radius)] border border-line bg-card px-3 py-2 text-xs shadow-md">
      <p className="font-data mb-1 text-ink-3">{label}</p>
      {sorted.map((p) => (
        <div key={p.name} className="flex items-center gap-1.5">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: p.color }}
          />
          <span className="w-7">{p.name}</span>
          <span className="tabular font-medium">{p.value?.toFixed(1)}</span>
        </div>
      ))}
    </div>
  );
}

export function PartyFrameChart() {
  const frames = partyFrames.frames;
  const [frame, setFrame] = useState(frames[0]);
  // Three party colours are all blue (L/M/KD), two green (C/MP), two red
  // (S/V) — solid lines alone don't reliably separate them on an 8-line
  // chart. Rather than dash patterns (tried, unreadable at this density),
  // hovering a legend entry isolates that one line: every other line dims,
  // so any single line — including the near-identical blues — can be traced
  // precisely on demand, while the chart stays clean (all solid) at rest.
  const [hovered, setHovered] = useState<string | null>(null);

  const data = useMemo(() => {
    const series = partyFrames.series[frame] ?? {};
    const years = new Set<string>();
    for (const p of PARTY_ORDER) {
      for (const y of Object.keys(series[p.code] ?? {})) years.add(y);
    }
    return Array.from(years)
      .map(Number)
      .sort((a, b) => a - b)
      .map((year) => {
        const row: Record<string, number | string> = { year };
        for (const p of PARTY_ORDER) {
          row[p.code] = series[p.code]?.[String(year)] ?? 0;
        }
        return row;
      });
  }, [frame]);

  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {frames.map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFrame(f)}
            className={`font-data rounded-full border px-3 py-1.5 text-[11px] uppercase tracking-wide transition-colors ${
              f === frame
                ? "border-accent bg-accent text-accent-ink"
                : "border-line bg-card text-ink-2 hover:border-line-2"
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="mt-5 h-80 w-full rounded-[var(--radius-lg)] border border-line bg-card p-4 shadow-sm">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -18 }}>
            <XAxis
              dataKey="year"
              tick={{ fontSize: 11, fill: "var(--ink-3)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--line)" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "var(--ink-3)" }}
              tickLine={false}
              axisLine={false}
              width={36}
            />
            <Tooltip content={<ChartTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 11, fontFamily: "var(--font-data)", cursor: "pointer" }}
              iconSize={16}
              iconType="plainline"
              onMouseEnter={(entry) => setHovered(entry.value ?? null)}
              onMouseLeave={() => setHovered(null)}
            />
            {PARTY_ORDER.map((p) => {
              const isDimmed = hovered !== null && hovered !== p.code;
              return (
                <Line
                  key={p.code}
                  type="monotone"
                  dataKey={p.code}
                  name={p.code}
                  stroke={`var(${p.colorVar})`}
                  strokeWidth={hovered === p.code ? 3 : 2.25}
                  strokeOpacity={isDimmed ? 0.15 : 1}
                  dot={false}
                  isAnimationActive={false}
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="font-data mt-2 text-[11px] text-ink-3">
        Andel av partiets ord som hör till frågan &quot;{frame}&quot;, per 10 000 ord och år.
      </p>
    </div>
  );
}
