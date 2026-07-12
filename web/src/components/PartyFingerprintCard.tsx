import { partyByCode } from "@/lib/parties";
import type { WordScore } from "@/lib/site-data";

interface PartyFingerprintCardProps {
  code: string;
  words: WordScore[];
  maxWords?: number;
}

/**
 * A party's distinctive-vocabulary "fingerprint": its most over-represented
 * words versus every other party (Fightin' Words), ranked by z-score.
 * Deliberately a ranked list rather than a literal word cloud — this is meant
 * to read as an analytical instrument, not a decorative graphic.
 */
export function PartyFingerprintCard({
  code,
  words,
  maxWords = 12,
}: PartyFingerprintCardProps) {
  const party = partyByCode(code);
  const shown = words.slice(0, maxWords);
  const maxZ = Math.max(...shown.map((w) => w.z), 1);

  return (
    <div className="rounded-[var(--radius-lg)] border border-line bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2">
        <span
          className="h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: party ? `var(${party.colorVar})` : undefined }}
        />
        <span className="font-display text-lg">{party?.name ?? code}</span>
        <span className="font-data ml-auto text-[11px] uppercase tracking-widest text-ink-3">
          {code}
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {shown.map((w) => (
          <div key={w.word} className="grid grid-cols-[7rem_1fr] items-center gap-2">
            <span className="font-data truncate text-right text-[13px]">{w.word}</span>
            <div className="h-2">
              <div
                className="h-2 rounded-[1px]"
                style={{
                  width: `${Math.max(6, (w.z / maxZ) * 100)}%`,
                  backgroundColor: party ? `var(${party.colorVar})` : "var(--accent)",
                }}
              />
            </div>
          </div>
        ))}
        {shown.length === 0 && (
          <p className="text-sm text-ink-3">Ingen data.</p>
        )}
      </div>
    </div>
  );
}
