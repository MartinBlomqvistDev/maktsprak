import { PARTY_ORDER } from "@/lib/parties";
import { words } from "./format";
import type { AnswerLine, Hit, PartiesLine, PartyAnswer } from "@/lib/fraga";
import { Clamp } from "./Clamp";
import { Paragraphs, PartyChip, QuoteCard, SpeechHeader, SpeechLink } from "./Speech";

/** The opening of a passage, cut at a word, for a result that is not opened. */
function opening(paragraphs: string[], limit = 220): string {
  const text = paragraphs.join(" ");
  if (text.length <= limit) return text;
  const cut = text.lastIndexOf(" ", limit);
  return text.slice(0, cut > 0 ? cut : limit) + " …";
}

/** The list of results: every passage the search ranked, best first. */
export function HitList({ hits }: { hits: Hit[] }) {
  if (hits.length === 0) {
    return <p className="text-sm text-ink-3">Inget anförande matchade frågan under filtren.</p>;
  }
  return (
    <ol className="space-y-3">
      {hits.map((hit) => (
        <li key={hit.passage_id} className="rounded-card border border-line bg-card px-5 py-4">
          <details className="group">
            <summary className="cursor-pointer list-none [&::-webkit-details-marker]:hidden">
              <SpeechHeader speech={hit} />
              <p className="mt-3 text-sm leading-relaxed text-ink-2 group-open:hidden">
                {opening(hit.paragraphs)}
              </p>
              <span className="mt-2 inline-block text-xs text-accent group-open:hidden">
                Visa hela passagen
              </span>
            </summary>
            <div className="mt-3">
              <Paragraphs paragraphs={hit.paragraphs} />
              <p className="mt-3">
                <SpeechLink url={hit.url} />
              </p>
            </div>
          </details>
        </li>
      ))}
    </ol>
  );
}

/** Who chose the quotations, said once above them. */
export function SelectorNote({
  line,
  hits,
}: {
  line: Pick<AnswerLine, "selector" | "fallback">;
  hits: number;
}) {
  if (line.selector === "claude") {
    return (
      <p className="text-sm text-ink-3">
        Claude Opus 5.5 valde styckena bland de {words(hits)} träffarna nedan. Texten kommer
        ordagrant ur riksdagens data.
      </p>
    );
  }
  return (
    <p className="text-sm text-ink-3">
      {line.fallback === "budget"
        ? "Månadens budget för språkmodellen är slut, så baslinjen valde citatet: den bästa träffens stycken med frågans ord, utan språkmodell."
        : line.fallback === "failed"
          ? "Språkmodellen svarade inte, så baslinjen valde citatet: den bästa träffens stycken med frågans ord, utan språkmodell."
          : "Baslinjen valde citatet: den bästa träffens stycken med frågans ord, utan språkmodell."}
    </p>
  );
}

/** The answer step's quotations, or the notice that none answers the question. */
export function AnswerView({ line, hits }: { line: AnswerLine; hits: number }) {
  if (line.quotes.length === 0) {
    return (
      <div className="rounded-card border border-dashed border-line-2 bg-paper-2 px-5 py-4">
        <p className="font-medium text-ink">Inget citat svarar på frågan.</p>
        <p className="mt-1 text-sm text-ink-2">
          {line.selector === "claude"
            ? `Claude Opus 5.5 läste de ${words(hits)} bästa träffarna och fann inget stycke som besvarar den.`
            : "Ingen träff nådde baslinjens tröskel."}
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-4">
      <SelectorNote line={line} hits={hits} />
      {line.quotes.map((quote) => (
        <QuoteCard key={`${quote.passage_id}:${quote.first_paragraph}`} quote={quote} />
      ))}
    </div>
  );
}

function PartyCell({ answer }: { answer: PartyAnswer }) {
  const party = PARTY_ORDER.find((p) => p.code === answer.party);
  const [first, ...more] = answer.quotes;
  return (
    <section className="flex flex-col rounded-card border border-line bg-card p-4">
      <h3 className="flex items-center gap-2">
        <PartyChip party={answer.party} />
        <span className="label text-ink-2">{party?.name ?? answer.party}</span>
      </h3>
      {first ? (
        <div className="mt-3 flex-1">
          <SpeechHeader speech={first} />
          <blockquote className="mt-3 border-l-[3px] border-accent pl-3">
            <Clamp>
              <Paragraphs paragraphs={first.paragraphs} />
            </Clamp>
          </blockquote>
          <p className="mt-3">
            <SpeechLink url={first.url} />
          </p>
          {more.length > 0 && (
            <details className="mt-3">
              <summary className="cursor-pointer text-xs text-accent">
                {more.length === 1 ? "Ett citat till" : `${more.length} citat till`}
              </summary>
              <div className="mt-3 space-y-3">
                {more.map((quote) => (
                  <QuoteCard key={`${quote.passage_id}:${quote.first_paragraph}`} quote={quote} />
                ))}
              </div>
            </details>
          )}
        </div>
      ) : (
        <p className="mt-3 flex-1 text-sm text-ink-3">
          {answer.selector === "claude"
            ? "Inga av partiets bästa träffar svarar på frågan."
            : "Ingen träff från partiet nådde tröskeln."}
        </p>
      )}
    </section>
  );
}

/** The question searched under each party in turn, in the site's party order. */
export function PartyGrid({ line }: { line: PartiesLine }) {
  const byParty = new Map(line.parties.map((answer) => [answer.party, answer]));
  const selectors = new Set(line.parties.map((answer) => answer.selector));
  return (
    <div className="space-y-4">
      <p className="text-sm text-ink-3">
        {selectors.has("claude") && !selectors.has("baseline")
          ? "Frågan ställd en gång per parti, bara bland partiets anföranden. Claude Opus 5.5 valde styckena."
          : "Frågan ställd en gång per parti, bara bland partiets anföranden. Baslinjen valde citaten, utan språkmodell: den bästa träffens stycken med frågans ord."}
      </p>
      <div className="grid gap-4 sm:grid-cols-2">
        {PARTY_ORDER.map((p) => {
          const answer = byParty.get(p.code);
          return answer ? <PartyCell key={p.code} answer={answer} /> : null;
        })}
      </div>
    </div>
  );
}
