import { PARTY_ORDER } from "@/lib/parties";

interface PredictionBarsProps {
  /** Party code → probability (0–1). Missing codes render as 0. */
  probabilities: Partial<Record<string, number>>;
}

/**
 * Horizontal probability bars for a single prediction, ranked highest first.
 * The "reads as..." moment after the redact→reveal hero sequence.
 */
export function PredictionBars({ probabilities }: PredictionBarsProps) {
  const ranked = [...PARTY_ORDER]
    .map((p) => ({ ...p, prob: probabilities[p.code] ?? 0 }))
    .sort((a, b) => b.prob - a.prob);

  return (
    <div className="space-y-2">
      {ranked.map((p, i) => (
        <div key={p.code} className="flex items-center gap-3">
          <span
            className={`w-8 shrink-0 font-data text-xs uppercase ${
              i === 0 ? "text-ink font-medium" : "text-ink-3"
            }`}
          >
            {p.code}
          </span>
          <div className="relative h-5 flex-1 overflow-hidden rounded-sm bg-paper-2">
            <div
              className="h-full rounded-sm transition-[width] duration-700 ease-out"
              style={{
                width: `${Math.max(p.prob * 100, 1.5)}%`,
                backgroundColor: `var(${p.colorVar})`,
                opacity: i === 0 ? 1 : 0.55,
              }}
            />
          </div>
          <span className="tabular w-12 shrink-0 text-right font-data text-xs text-ink-2">
            {(p.prob * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}
