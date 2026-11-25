import {
  FragaHttpError,
  type AnswerLine,
  type FragaLine,
  type FragaRequest,
  type PartiesLine,
  type SearchLine,
} from "@/lib/fraga";
import { loadExample, type Example, type ExampleEntry } from "./types";

/**
 * A stand-in for /api/fraga during development, so every state of the page
 * can be looked at before the search service answers. Only `next dev` loads
 * it: the page imports it behind a check on NODE_ENV, and only when the
 * address carries `?stub=<scenario>`. Its lines are the stored example
 * answers, so the stand-in never shows a text the corpus does not hold.
 */

export const SCENARIOS = [
  "svar",
  "kallstart",
  "bm25",
  "budget",
  "misslyckat",
  "avstar",
  "tomt",
  "avbrott",
  "fel",
  "natverk",
  "ej-ansluten",
  "hastighet",
  "ogiltig",
] as const;

export type Scenario = (typeof SCENARIOS)[number];

export function scenarioFrom(search: string): Scenario | null {
  const value = new URLSearchParams(search).get("stub");
  return value && (SCENARIOS as readonly string[]).includes(value) ? (value as Scenario) : null;
}

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      window.clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });
}

function baselineOf(line: PartiesLine): PartiesLine {
  return {
    ...line,
    parties: line.parties.map((answer) => ({
      ...answer,
      selector: "baseline",
      model: null,
      quotes: answer.quotes.slice(0, 1),
    })),
  };
}

export async function stubAsk(
  scenario: Scenario,
  entries: ExampleEntry[],
  request: FragaRequest,
  onLine: (line: FragaLine) => void,
  signal?: AbortSignal,
): Promise<void> {
  const entry = entries.find((item) => item.question === request.question.trim()) ?? entries[0];
  const example: Example = await loadExample(entry.id);
  if (scenario === "fel") {
    await wait(700, signal);
    throw new FragaHttpError(502, "search_failed", null);
  }
  if (scenario === "natverk") {
    await wait(400, signal);
    throw new TypeError("Failed to fetch");
  }
  if (scenario === "ej-ansluten") throw new FragaHttpError(501, "search_not_connected", null);
  if (scenario === "hastighet") throw new FragaHttpError(429, "rate_limited", 60);
  if (scenario === "ogiltig") throw new FragaHttpError(400, "invalid_request", null);

  const cold = scenario === "kallstart";
  await wait(cold ? 6000 : 800, signal);

  if (request.mode === "parties") {
    onLine(baselineOf(example.parties));
    return;
  }

  let search: SearchLine = example.search;
  if (scenario === "bm25") search = { ...search, retrieval: "bm25", vector_wait_ms: null };
  if (scenario === "tomt") search = { ...search, hits: [] };
  onLine(search);
  if (request.mode === "search") return;

  if (scenario === "avbrott") {
    await wait(1500, signal);
    onLine({ type: "error", error: "answer_failed" });
    return;
  }
  await wait(cold ? 3000 : 2500, signal);
  let answer: AnswerLine = example.answer;
  if (scenario === "budget" || scenario === "misslyckat") {
    answer = {
      ...answer,
      selector: "baseline",
      model: null,
      fallback: scenario === "budget" ? "budget" : "failed",
      quotes: answer.quotes.slice(0, 1),
    };
  }
  if (scenario === "avstar" || scenario === "tomt") {
    answer = { ...answer, answerable: false, quotes: [] };
  }
  onLine(answer);
}
