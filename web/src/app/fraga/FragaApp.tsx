"use client";

import { useEffect, useRef, useState } from "react";
import { PARTY_ORDER, type PartyCode } from "@/lib/parties";
import {
  askFraga,
  FragaHttpError,
  QUESTION_MAX_CHARS,
  type AnswerLine,
  type FragaLine,
  type FragaRequest,
  type Mode,
  type PartiesLine,
  type SearchLine,
} from "@/lib/fraga";
import { int, longDate, seconds, words } from "./format";
import { AnswerView, HitList, PartyGrid } from "./Results";
import { SpeakerPicker, type Speaker } from "./SpeakerPicker";
import { loadExample, type Example, type ExampleEntry } from "./types";

// The search service scales to zero, so the first question after an idle spell
// waits for it to start. A warm search answers within about a second and a half
// in most cases, but the question's vector comes from an outside service and
// can take several seconds now and then. The hint appears once nothing has come
// back for longer than that, and it names both causes.
const COLD_START_HINT_MS = 3500;

type View = "svar" | "partier";

type Failure =
  | { kind: "invalid" }
  | { kind: "rate"; retryAfterS: number | null }
  | { kind: "not-connected" }
  | { kind: "failed" }
  | { kind: "interrupted" };

interface Shown {
  source: "live" | "example";
  key: string;
  question: string;
  filters: Filters;
  search?: SearchLine;
  answer?: AnswerLine;
  parties?: PartiesLine;
  /** Milliseconds from the question to its last line, for a live answer. */
  tookMs?: number;
}

type Phase = "idle" | "waiting" | "answering" | "done" | "error";

interface Filters {
  party: PartyCode | "";
  yearFrom: number | "";
  yearTo: number | "";
  speaker: Speaker | null;
}

const NO_FILTERS: Filters = { party: "", yearFrom: "", yearTo: "", speaker: null };

/** Milliseconds on the page's clock, for how long an answer took. */
function clock(): number {
  return performance.now();
}

/** Whether a shown result was asked with the question and filters now in the form. */
function sameQuery(shown: Shown, question: string, filters: Filters, view: View): boolean {
  const asked = shown.filters;
  const years = asked.yearFrom === filters.yearFrom && asked.yearTo === filters.yearTo;
  if (shown.question !== question.trim() || !years) return false;
  if (view === "partier") return true;
  return asked.party === filters.party && asked.speaker?.id === filters.speaker?.id;
}

function failureOf(error: unknown): Failure | null {
  if (error instanceof DOMException && error.name === "AbortError") return null;
  if (error instanceof FragaHttpError) {
    if (error.status === 400) return { kind: "invalid" };
    if (error.status === 429) return { kind: "rate", retryAfterS: error.retryAfterS };
    if (error.status === 501) return { kind: "not-connected" };
  }
  return { kind: "failed" };
}

function FailureNotice({ failure }: { failure: Failure }) {
  const text = {
    invalid: `Frågan får vara högst ${int(QUESTION_MAX_CHARS)} tecken och kan inte vara tom.`,
    rate: "Många frågor på kort tid. Vänta en stund och försök igen.",
    "not-connected": "Sökningen är inte ansluten ännu. Exemplen går att läsa ändå.",
    failed: "Något gick fel. Försök igen om en stund.",
    interrupted:
      "Svaret bröts innan citaten kom. Träffarna stämmer; ställ frågan igen för att få citaten.",
  }[failure.kind];
  const detail =
    failure.kind === "rate" && failure.retryAfterS
      ? ` Det går igen om ${failure.retryAfterS} sekunder.`
      : "";
  return (
    <p
      className={`text-sm ${failure.kind === "not-connected" ? "text-ink-2" : "text-crit"}`}
      role="alert"
    >
      {text}
      {detail}
    </p>
  );
}

function RetrievalNote({
  line,
  vectorLimitS,
}: {
  line: SearchLine | PartiesLine;
  vectorLimitS: number;
}) {
  return (
    <p className="text-sm text-ink-3">
      {line.retrieval === "hybrid" ? (
        <>
          Sökt med nyckelord och vektorer.
          {line.vector_wait_ms !== null && (
            <> Frågans vektor kom efter {seconds(line.vector_wait_ms)}.</>
          )}
        </>
      ) : (
        <>
          Frågans vektor kom inte inom {int(vectorLimitS)} sekunder, så sökningen gjordes med
          nyckelord enbart (BM25).
        </>
      )}{" "}
      Indexet går till och med {longDate(line.index.last_date)}.
    </p>
  );
}

const selectClass =
  "w-full rounded-sm border border-line-2 bg-paper px-3 py-2 text-sm text-ink focus:border-accent focus:outline-none disabled:opacity-50";

function exampleShown(example: Example): Shown {
  return {
    source: "example",
    key: example.id,
    question: example.question,
    filters: NO_FILTERS,
    search: example.search,
    answer: example.answer,
    parties: example.parties,
  };
}

export function FragaApp({
  entries,
  initial,
  generated,
  firstYear,
  lastYear,
  answerStep,
  vectorLimitS,
}: {
  entries: ExampleEntry[];
  /** The first example, shown before anything is asked. */
  initial: Example;
  generated: string;
  firstYear: number;
  lastYear: number;
  /** False once the launch rule has failed: the page then searches without the answer step. */
  answerStep: boolean;
  /** How long the service waits for a question's vector before BM25 answers alone. */
  vectorLimitS: number;
}) {
  const [question, setQuestion] = useState(initial.question);
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [view, setView] = useState<View>("svar");
  const [shown, setShown] = useState<Shown | null>(() => exampleShown(initial));
  const [phase, setPhase] = useState<Phase>("done");
  const [failure, setFailure] = useState<Failure | null>(null);
  const [warming, setWarming] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const hintTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loaded = useRef(new Map<string, Example>([[initial.id, initial]]));
  const [pickerKey, setPickerKey] = useState(0);

  useEffect(
    () => () => {
      controller.current?.abort();
      if (hintTimer.current) clearTimeout(hintTimer.current);
    },
    [],
  );

  const yearOptions: number[] = [];
  for (let year = firstYear; year <= lastYear; year++) yearOptions.push(year);
  const busy = phase === "waiting" || phase === "answering";

  function stopHint() {
    if (hintTimer.current) clearTimeout(hintTimer.current);
    hintTimer.current = null;
    setWarming(false);
  }

  async function ask(mode: Mode, keep?: Shown) {
    const text = question.trim();
    if (!text || text.length > QUESTION_MAX_CHARS) {
      setFailure({ kind: "invalid" });
      setPhase("error");
      return;
    }
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    const started = clock();
    const base: Shown = keep ?? { source: "live", key: "live", question: text, filters };
    setShown(base);
    setFailure(null);
    setPhase("waiting");
    stopHint();
    hintTimer.current = setTimeout(() => setWarming(true), COLD_START_HINT_MS);

    const request: FragaRequest = {
      question: text,
      mode,
      filters: {
        party: mode === "parties" || !filters.party ? null : filters.party,
        year_from: filters.yearFrom || null,
        year_to: filters.yearTo || null,
        speaker: mode === "parties" ? null : (filters.speaker?.id ?? null),
      },
    };
    let current = base;
    const onLine = (line: FragaLine) => {
      stopHint();
      if (line.type === "error") {
        setFailure({ kind: "interrupted" });
        setPhase("error");
        return;
      }
      if (line.type === "search") current = { ...current, search: line };
      if (line.type === "answer") current = { ...current, answer: line };
      if (line.type === "parties") current = { ...current, parties: line };
      const finished = line.type !== "search" || mode === "search";
      if (finished) current = { ...current, tookMs: clock() - started };
      setShown(current);
      setPhase(finished ? "done" : "answering");
    };

    try {
      const scenario =
        process.env.NODE_ENV !== "production"
          ? (await import("./stub")).scenarioFrom(window.location.search)
          : null;
      if (scenario) {
        const { stubAsk } = await import("./stub");
        await stubAsk(scenario, entries, request, onLine, abort.signal);
      } else {
        await askFraga(request, onLine, abort.signal);
      }
    } catch (error) {
      const found = failureOf(error);
      if (found) {
        setFailure(found);
        setPhase("error");
      }
    } finally {
      if (controller.current === abort) stopHint();
    }
  }

  function submit(event: React.FormEvent) {
    event.preventDefault();
    if (view === "partier") void ask("parties");
    else void ask(answerStep ? "answer" : "search");
  }

  async function showExample(entry: ExampleEntry) {
    controller.current?.abort();
    controller.current = null;
    stopHint();
    setQuestion(entry.question);
    setFilters(NO_FILTERS);
    setPickerKey((n) => n + 1);
    setFailure(null);
    let example = loaded.current.get(entry.id);
    if (!example) {
      setShown(null);
      setPhase("waiting");
      try {
        example = await loadExample(entry.id);
        loaded.current.set(entry.id, example);
      } catch {
        setFailure({ kind: "failed" });
        setPhase("error");
        return;
      }
    }
    setShown(exampleShown(example));
    setPhase("done");
  }

  function switchView(next: View) {
    setView(next);
    if (!shown || shown.source === "example" || busy) return;
    if (!sameQuery(shown, question, filters, next)) return;
    if (next === "partier" && !shown.parties) void ask("parties", { ...shown });
    if (next === "svar" && !shown.search) void ask(answerStep ? "answer" : "search", { ...shown });
  }

  const example = shown?.source === "example";
  const line = view === "partier" ? shown?.parties : shown?.search;

  return (
    <div>
      <div className="rounded-card border border-line bg-card p-5 sm:p-6">
        <div className="anf-marker mb-4">
          <span className="tabular">Anf. live</span>
          <span className="text-line-2">·</span>
          <b>Fråga protokollet</b>
          <span className="h-px flex-1 bg-line" aria-hidden="true" />
        </div>

        <form onSubmit={submit}>
          <label htmlFor="fraga-question" className="sr-only">
            Din fråga
          </label>
          <textarea
            id="fraga-question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            maxLength={QUESTION_MAX_CHARS}
            rows={2}
            placeholder="Ställ en fråga om något riksdagen har debatterat"
            className="w-full resize-none rounded-sm border border-line-2 bg-paper px-4 py-3 font-body text-[15px] text-ink placeholder:text-ink-3 focus:border-accent focus:outline-none"
          />

          <div className="mt-3 grid gap-3 sm:grid-cols-[1fr_1fr_1fr_1.6fr]">
            <label className="block">
              <span className="label mb-1 block text-ink-3">Parti</span>
              <select
                value={filters.party}
                disabled={view === "partier"}
                onChange={(event) =>
                  setFilters({ ...filters, party: event.target.value as PartyCode | "" })
                }
                className={selectClass}
              >
                <option value="">Alla partier</option>
                {PARTY_ORDER.map((p) => (
                  <option key={p.code} value={p.code}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="label mb-1 block text-ink-3">Från år</span>
              <select
                value={filters.yearFrom}
                onChange={(event) => {
                  const value = event.target.value ? Number(event.target.value) : "";
                  setFilters({
                    ...filters,
                    yearFrom: value,
                    yearTo:
                      value && filters.yearTo && filters.yearTo < value ? value : filters.yearTo,
                  });
                }}
                className={selectClass}
              >
                <option value="">{firstYear}</option>
                {yearOptions.slice(1).map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="label mb-1 block text-ink-3">Till år</span>
              <select
                value={filters.yearTo}
                onChange={(event) => {
                  const value = event.target.value ? Number(event.target.value) : "";
                  setFilters({
                    ...filters,
                    yearTo: value,
                    yearFrom:
                      value && filters.yearFrom && filters.yearFrom > value
                        ? value
                        : filters.yearFrom,
                  });
                }}
                className={selectClass}
              >
                <option value="">{lastYear}</option>
                {yearOptions.slice(0, -1).map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
            </label>
            <div>
              <span className="label mb-1 block text-ink-3">Talare</span>
              <SpeakerPicker
                key={pickerKey}
                value={filters.speaker}
                disabled={view === "partier"}
                onChange={(speaker) => setFilters({ ...filters, speaker })}
              />
            </div>
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <div role="tablist" aria-label="Visa" className="flex rounded-sm border border-line-2 p-0.5">
              {(
                [
                  ["svar", "Svaret"],
                  ["partier", "Alla åtta partier"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  role="tab"
                  aria-selected={view === value}
                  onClick={() => switchView(value)}
                  className={`rounded-[2px] px-3 py-1.5 label transition-colors ${view === value ? "bg-accent text-accent-ink" : "text-ink-2 hover:text-accent"}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <button
              type="submit"
              disabled={busy || !question.trim()}
              className="shrink-0 rounded-sm bg-accent px-5 py-2 label text-accent-ink transition-opacity hover:opacity-90 disabled:opacity-40"
            >
              {busy ? "Söker…" : "Fråga"}
            </button>
          </div>
          <p className="mt-3 text-xs text-ink-3">
            Sajten sparar inte frågan. Den skickas till OpenRouter för vektorn och till Anthropic
            när Opus väljer citat.
          </p>
          {view === "partier" && (
            <p className="mt-2 text-xs text-ink-3">
              Frågan ställs en gång per parti, så parti och talare väljs inte här.
            </p>
          )}
        </form>

        <div className="mt-5 border-t border-line pt-4">
          <p className="label mb-2 text-ink-3">Eller läs ett exempel</p>
          <ul className="flex flex-wrap gap-2">
            {entries.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => void showExample(item)}
                  aria-pressed={shown?.source === "example" && shown.key === item.id}
                  className="rounded-full border border-line-2 px-3 py-1 text-left text-sm text-ink-2 transition-colors hover:border-accent hover:text-accent aria-pressed:border-accent aria-pressed:text-accent"
                >
                  {item.question}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="mt-8 space-y-6" aria-live="polite">
        {phase === "waiting" && (
          <div className="space-y-2">
            <p className="text-sm text-ink-2">Söker i protokollen…</p>
            {warming && (
              <p className="text-xs text-ink-3">
                Sökningen körs på en tjänst som stängs av när ingen använder den, och den första
                frågan får vänta medan den startar. Frågans vektor kommer från en annan tjänst.
                Dröjer den mer än {int(vectorLimitS)} sekunder söker sidan med nyckelord enbart.
              </p>
            )}
          </div>
        )}

        {failure && <FailureNotice failure={failure} />}

        {shown && line && (
          <>
            {example && (
              <p className="text-sm text-ink-3">
                Exempel, framtaget {longDate(generated)} med samma sökning och samma språkmodell
                som sidan kör.
              </p>
            )}
            <RetrievalNote line={line} vectorLimitS={vectorLimitS} />
          </>
        )}

        {view === "svar" && shown?.search && (
          <>
            {answerStep && (
              <section>
                <h2 className="anf-marker mb-3">
                  <b>Citaten</b>
                </h2>
                {shown.answer ? (
                  <AnswerView line={shown.answer} hits={shown.search.hits.length} />
                ) : phase === "answering" ? (
                  <p className="text-sm text-ink-2">
                    Väljer citat ur de {words(shown.search.hits.length)} träffarna…
                  </p>
                ) : null}
              </section>
            )}
            <section>
              <h2 className="anf-marker mb-3">
                <b>
                  {shown.search.hits.length > 1
                    ? `De ${words(shown.search.hits.length)} bästa träffarna`
                    : "Träffen"}
                </b>
              </h2>
              <HitList hits={shown.search.hits} />
            </section>
          </>
        )}

        {view === "partier" && shown?.parties && <PartyGrid line={shown.parties} />}

        {shown?.tookMs !== undefined && phase === "done" && (
          <p className="text-xs text-ink-3">Svaret kom efter {seconds(shown.tookMs)}.</p>
        )}
      </div>
    </div>
  );
}
