import type { PartyCode } from "./parties";

/**
 * The contract of /api/fraga, the site's proxy to the search service.
 *
 * Mirrors search/page/contract.py, which writes the page's example answers in
 * the same shape. A request names the question, its filters and a mode. The
 * answer is JSON Lines: a `search` line with the list of results, then, for
 * mode `answer`, an `answer` line with the quotations; mode `parties` answers
 * with one `parties` line. Every text in a line is a speaker's paragraphs as
 * the corpus holds them, so the page renders them as plain text and never as
 * markup.
 */

export type Mode = "answer" | "search" | "parties";

export interface FragaFilters {
  party?: PartyCode | null;
  year_from?: number | null;
  year_to?: number | null;
  /** The Riksdag's person_id for the speaker. */
  speaker?: string | null;
}

export interface FragaRequest {
  question: string;
  filters: FragaFilters;
  mode: Mode;
}

export interface Hit {
  rank: number;
  passage_id: string;
  anf_id: string;
  /** The session, such as "2024/25". */
  rm: string;
  dok_nummer: number;
  party: PartyCode;
  /** The speaker as the Riksdag publishes the name, title included. */
  speaker: string;
  /** ISO date of the protocol. */
  date: string;
  title: string | null;
  is_reply: boolean;
  url: string;
  paragraphs: string[];
}

export interface Quote extends Hit {
  /** First paragraph quoted, numbered from 1 within the passage. */
  first_paragraph: number;
  last_paragraph: number;
}

export interface IndexFacts {
  last_date: string;
  speeches: number;
  passages: number;
}

export type Retrieval = "hybrid" | "bm25";
export type SelectorName = "claude" | "baseline";
export type Fallback = "budget" | "failed";

export interface SearchLine {
  type: "search";
  retrieval: Retrieval;
  vector_wait_ms: number | null;
  index: IndexFacts;
  hits: Hit[];
}

export interface AnswerLine {
  type: "answer";
  selector: SelectorName;
  model: string | null;
  fallback: Fallback | null;
  answerable: boolean | null;
  quotes: Quote[];
}

export interface PartyAnswer {
  party: PartyCode;
  selector: SelectorName;
  model: string | null;
  fallback: Fallback | null;
  answerable: boolean | null;
  quotes: Quote[];
}

export interface PartiesLine {
  type: "parties";
  retrieval: Retrieval;
  vector_wait_ms: number | null;
  index: IndexFacts;
  parties: PartyAnswer[];
}

export interface ErrorLine {
  type: "error";
  error: string;
}

export type FragaLine = SearchLine | AnswerLine | PartiesLine | ErrorLine;

/** Error codes the route answers with before any line is sent. */
export type FragaErrorCode =
  | "invalid_request"
  | "rate_limited"
  | "search_not_connected"
  | "search_failed"
  | "answer_step_off";

/** The page's limits on a question, the same as the service's. */
export const QUESTION_MAX_CHARS = 400;

/** An answer from /api/fraga that came back as an HTTP error, not as lines. */
export class FragaHttpError extends Error {
  constructor(
    readonly status: number,
    readonly code: FragaErrorCode | "unknown",
    readonly retryAfterS: number | null,
  ) {
    super(`/api/fraga answered ${status} (${code})`);
  }
}

/** Read a response body of JSON Lines, one parsed line at a time. */
export async function* readLines(response: Response): AsyncGenerator<FragaLine> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";
  for (;;) {
    const { done, value } = await reader.read();
    buffered += decoder.decode(value, { stream: !done });
    let newline = buffered.indexOf("\n");
    while (newline >= 0) {
      const line = buffered.slice(0, newline).trim();
      buffered = buffered.slice(newline + 1);
      if (line) yield JSON.parse(line) as FragaLine;
      newline = buffered.indexOf("\n");
    }
    if (done) break;
  }
  const rest = buffered.trim();
  if (rest) yield JSON.parse(rest) as FragaLine;
}

/** Turn an error response into a FragaHttpError, reading its code when it has one. */
export async function httpError(response: Response): Promise<FragaHttpError> {
  let code: FragaErrorCode | "unknown" = "unknown";
  try {
    const body = (await response.json()) as { error?: string };
    if (body.error) code = body.error as FragaErrorCode;
  } catch {
    // A body that is not JSON leaves the code unknown; the status still tells.
  }
  const header = response.headers.get("Retry-After");
  const retryAfter = header && /^\d+$/.test(header) ? Number(header) : null;
  return new FragaHttpError(response.status, code, retryAfter);
}

/**
 * Ask /api/fraga and hand each line to `onLine` as it arrives.
 *
 * Throws a FragaHttpError when the route answers with an error status, and
 * lets a network failure or an aborted request throw as fetch does.
 */
export async function askFraga(
  request: FragaRequest,
  onLine: (line: FragaLine) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/fraga", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) throw await httpError(response);
  for await (const line of readLines(response)) onLine(line);
}
