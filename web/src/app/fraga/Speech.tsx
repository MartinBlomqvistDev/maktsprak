import { partyByCode } from "@/lib/parties";
import type { Hit, Quote } from "@/lib/fraga";
import { longDate } from "./format";

/**
 * The speaker field without what the card shows on its own: the party in
 * parentheses becomes the party chip and a trailing "replik" the reply badge.
 * The name and title stay as the Riksdag publishes them.
 */
export function speakerName(speaker: string): string {
  let name = speaker.trim();
  if (/\sreplik$/i.test(name)) name = name.replace(/\s+replik$/i, "");
  return name.replace(/\s*\([^()]*\)$/, "");
}

/** The speech's number, the part of its key after the protocol. */
export function speechNumber(anfId: string): string {
  return anfId.slice(anfId.lastIndexOf("-") + 1);
}

export function PartyChip({ party }: { party: string }) {
  const p = partyByCode(party);
  if (!p) return <span className="font-data text-xs text-ink-3">{party}</span>;
  return (
    <span
      className="inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium"
      style={{
        backgroundColor: `color-mix(in srgb, var(${p.colorVar}) 16%, transparent)`,
        color: `var(${p.colorVar})`,
      }}
      title={p.name}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: `var(${p.colorVar})` }}
        aria-hidden="true"
      />
      {p.code}
    </span>
  );
}

/** The record line, speaker and debate heading above a quotation or a result. */
export function SpeechHeader({ speech }: { speech: Hit }) {
  return (
    <div>
      <p className="anf-marker flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="tabular">
          Prot. {speech.rm}:{speech.dok_nummer}
        </span>
        <span className="text-line-2" aria-hidden="true">
          ·
        </span>
        <span className="tabular">Anf. {speechNumber(speech.anf_id)}</span>
        <span className="text-line-2" aria-hidden="true">
          ·
        </span>
        <span>{longDate(speech.date)}</span>
        {speech.is_reply && (
          <span className="rounded-sm border border-line-2 px-1.5 py-px text-[11px] text-ink-2">
            Replik
          </span>
        )}
      </p>
      <p className="mt-2 flex flex-wrap items-center gap-2">
        <PartyChip party={speech.party} />
        <span className="font-medium text-ink">{speakerName(speech.speaker)}</span>
      </p>
      {speech.title && <p className="mt-1 text-sm text-ink-3">{speech.title}</p>}
    </div>
  );
}

/** A link to the speech's own page at the Riksdag. */
export function SpeechLink({ url }: { url: string }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="text-sm text-accent underline underline-offset-2"
    >
      Hela anförandet hos riksdagen
    </a>
  );
}

/** A speaker's paragraphs as plain text, one paragraph each. */
export function Paragraphs({ paragraphs }: { paragraphs: string[] }) {
  return (
    <div className="space-y-3 text-[15px] leading-relaxed text-ink">
      {paragraphs.map((paragraph, i) => (
        <p key={i}>{paragraph}</p>
      ))}
    </div>
  );
}

/** One quotation: the record line, the quoted paragraphs and the link. */
export function QuoteCard({ quote }: { quote: Quote }) {
  return (
    <article className="rounded-card border border-line bg-card p-5">
      <SpeechHeader speech={quote} />
      <blockquote className="mt-4 border-l-[3px] border-accent pl-4">
        <Paragraphs paragraphs={quote.paragraphs} />
      </blockquote>
      <p className="mt-4">
        <SpeechLink url={quote.url} />
      </p>
    </article>
  );
}
