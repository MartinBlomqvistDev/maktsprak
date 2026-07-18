import { ProtokollMarker } from "./ProtokollMarker";

interface ComingNextProps {
  protNr: number;
  section: string;
  title: string;
  lede: string;
  planned: string[];
}

/**
 * A designed "not built yet" state for routes ahead in the roadmap, reads
 * as a deliberate placeholder with real content about what's coming, not a
 * broken or forgotten page.
 */
export function ComingNext({
  protNr,
  section,
  title,
  lede,
  planned,
}: ComingNextProps) {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 sm:py-20">
      <p className="anf-marker mb-4">
        <span className="tabular">PROT. 2026/27:{protNr}</span>
        <span className="text-line-2">·</span>
        <span>{section}</span>
      </p>
      <h1 className="font-display text-4xl leading-tight tracking-tight sm:text-5xl">
        {title}
      </h1>
      <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-2">{lede}</p>

      <section className="mt-14">
        <ProtokollMarker n={protNr * 10} name="UNDER ARBETE" />
        <ul className="mt-5 space-y-3">
          {planned.map((item) => (
            <li key={item} className="flex gap-3 text-ink-2">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
