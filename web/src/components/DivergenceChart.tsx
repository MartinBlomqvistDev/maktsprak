"use client";

import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { divergence } from "@/lib/site-data";

export function DivergenceChart() {
  const years = Object.keys(divergence)
    .map(Number)
    .sort((a, b) => a - b);
  const data = years.map((y) => ({ year: y, value: divergence[String(y)] }));
  const first = data[0];
  const last = data[data.length - 1];

  return (
    <div className="rounded-[var(--radius-lg)] border border-line bg-card p-4 shadow-sm">
      <div className="h-40 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 18, right: 24, bottom: 0, left: 16 }}>
            <defs>
              <linearGradient id="grad-div" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.22} />
                <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="year"
              tick={{ fontSize: 10, fill: "var(--ink-3)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--line)" }}
              interval="preserveStartEnd"
              minTickGap={28}
            />
            <YAxis domain={[0.08, 0.2]} hide />
            <Area
              type="monotone"
              dataKey="value"
              stroke="var(--accent)"
              strokeWidth={2}
              fill="url(#grad-div)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="font-data mt-1 flex justify-between text-[11px]">
        <span className="text-ink-2">
          {first.year}: <b className="text-ink">{first.value.toFixed(2)}</b>
        </span>
        <span className="text-ink-2">
          {last.year}: <b className="text-ink">{last.value.toFixed(2)}</b>
        </span>
      </div>
    </div>
  );
}
