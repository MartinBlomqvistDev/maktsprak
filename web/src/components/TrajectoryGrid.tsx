"use client";

import { Area, AreaChart, ReferenceDot, ResponsiveContainer, YAxis } from "recharts";
import { trajectories } from "@/lib/site-data";

interface TrajectoryGridProps {
  terms: string[];
  direction: "up" | "down";
  notes: Record<string, string>;
}

interface Point {
  year: number;
  value: number;
}

function seriesFor(term: string): Point[] {
  const years = Object.keys(trajectories.series)
    .map(Number)
    .sort((a, b) => a - b);
  return years.map((y) => ({
    year: y,
    value: trajectories.series[String(y)]?.[term] ?? 0,
  }));
}

function peakPoint(data: Point[]): Point {
  return data.reduce((best, d) => (d.value > best.value ? d : best), data[0]);
}

export function TrajectoryGrid({ terms, direction, notes }: TrajectoryGridProps) {
  const color = direction === "up" ? "var(--accent)" : "var(--crit)";
  const allSeries = terms.map((term) => ({ term, data: seriesFor(term) }));

  // Shared y-domain across every card in this group, so bar/curve HEIGHT is
  // comparable between terms — without this each card auto-scales to its own
  // max and a modest rise looks exactly as dramatic as a huge one.
  const sharedMax = Math.max(
    ...allSeries.flatMap(({ data }) => data.map((d) => d.value)),
    0.1
  );

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {allSeries.map(({ term, data }) => {
        const peak = peakPoint(data);
        const first = data[0]?.year;
        const last = data[data.length - 1]?.year;

        return (
          <div
            key={term}
            className="rounded-[var(--radius)] border border-line bg-card px-3 pb-2 pt-3 shadow-sm"
          >
            <div className="flex items-baseline justify-between gap-2">
              <p className="font-data truncate text-[13px]">{term}</p>
              {/* The peak year is stated here, in plain text next to the value —
                  not positioned inside the chart. A chart-coordinate label (DOM
                  percentage math, and separately Recharts' own ReferenceDot
                  label) both proved unreliable at this card size: either it
                  drifted from the dot's true position or silently failed to
                  render. Plain text next to a number can't misalign. */}
              <p className="tabular font-data shrink-0 text-[11px] text-ink-3">
                {peak.value.toFixed(1)} · {peak.year}
              </p>
            </div>
            <p className="mb-1 min-h-8 text-[11px] italic leading-snug text-ink-3">
              {notes[term] ?? ""}
            </p>
            <div className="h-14 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 6, right: 2, bottom: 0, left: 2 }}>
                  <YAxis hide domain={[0, sharedMax]} />
                  <defs>
                    <linearGradient id={`grad-${term}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={color} stopOpacity={0.25} />
                      <stop offset="100%" stopColor={color} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke={color}
                    strokeWidth={2}
                    fill={`url(#grad-${term})`}
                    isAnimationActive={false}
                    dot={false}
                  />
                  {/* Dot marks the peak's true position on the curve; the year
                      itself is stated in the header above, not here. */}
                  <ReferenceDot x={peak.year} y={peak.value} r={2.6} fill={color} stroke="none" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="font-data flex justify-between text-[10px] text-ink-3">
              <span>{first}</span>
              <span>{last}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
