import type { AnswerLine, PartiesLine, SearchLine } from "@/lib/fraga";

/**
 * One example as web/public/fraga/exempel/<id>.json holds it: the lines
 * /api/fraga would have answered with, written once by
 * search/pipeline/15_build_page.py with the chain the page runs.
 */
export interface Example {
  id: string;
  question: string;
  search: SearchLine;
  /** Left out when the page runs without the answer step. */
  answer?: AnswerLine;
  parties?: PartiesLine;
}

/** One example in the page's list. */
export interface ExampleEntry {
  id: string;
  question: string;
}

/** What the examples were answered with. */
export interface ExampleChain {
  embedding_model: string;
  provider: string;
  rrf_k: number;
  model: string;
  prompt_sha256: string;
  effort: string;
  transport: string;
  requests: number;
  cost_usd: number;
}

/** web/src/data/fraga/examples.json: the list of examples and their chain. */
export interface ExamplesFile {
  generated: string;
  chain: ExampleChain;
  examples: ExampleEntry[];
}

/** Fetch one example's lines from the site's static files. */
export async function loadExample(id: string): Promise<Example> {
  const response = await fetch(`/fraga/exempel/${encodeURIComponent(id)}.json`);
  if (!response.ok) throw new Error(`exempel/${id}.json: ${response.status}`);
  return (await response.json()) as Example;
}
