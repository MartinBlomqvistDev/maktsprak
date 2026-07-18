import { partyByCode } from "@/lib/parties";

interface ProtokollMarkerProps {
  /** Speech number, mirroring the Riksdag's "Anf. N" convention. */
  n: number;
  /** Speaker name, or a section label like "METOD" / "BENCHMARK". */
  name: string;
  /** Party code, if this marker represents an actual speaker. */
  party?: string;
}

/**
 * A transcript-marker divider styled after the Riksdag protocol's own
 * "Anf. 42 NAMN (PARTI):" convention. Used as a structural device throughout
 * the site instead of generic section headers, the record's own grammar.
 */
export function ProtokollMarker({ n, name, party }: ProtokollMarkerProps) {
  const p = party ? partyByCode(party) : undefined;
  return (
    <div className="flex items-center gap-3 anf-marker">
      <span className="tabular">ANF. {n}</span>
      <span aria-hidden="true" className="text-line-2">
        ·
      </span>
      <b>{name}</b>
      {p && (
        <span
          className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium normal-case"
          style={{
            backgroundColor: `color-mix(in srgb, var(${p.colorVar}) 16%, transparent)`,
            color: `var(${p.colorVar})`,
          }}
        >
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: `var(${p.colorVar})` }}
          />
          {p.code}
        </span>
      )}
      <span className="h-px flex-1 bg-line" aria-hidden="true" />
    </div>
  );
}
