import { PARTY_ORDER } from "@/lib/parties";

/**
 * The eight-party spectrum bar, the site's recurring signature mark.
 * Colour is used only as data here, never as UI chrome elsewhere.
 */
export function PartySpectrum({ withLabels = true }: { withLabels?: boolean }) {
  return (
    <div>
      <div
        className="flex h-2 overflow-hidden rounded-full"
        role="img"
        aria-label="De åtta riksdagspartierna"
      >
        {PARTY_ORDER.map((p) => (
          <span
            key={p.code}
            className="flex-1"
            style={{ backgroundColor: `var(${p.colorVar})` }}
          />
        ))}
      </div>
      {withLabels && (
        <div className="mt-1.5 flex justify-between font-data text-[10px] uppercase tracking-widest text-ink-3">
          {PARTY_ORDER.map((p) => (
            <span key={p.code}>{p.code}</span>
          ))}
        </div>
      )}
    </div>
  );
}
