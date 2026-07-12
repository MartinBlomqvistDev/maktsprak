/**
 * Typed access to the precomputed analysis JSON in src/data/.
 *
 * All of this is generated offline by scripts/build_site_data.py (in the
 * Python pipeline repo) from the archived speech corpus — the site itself
 * never queries a database. Re-run that script and copy its output here to
 * refresh the numbers.
 */
import distinctiveRaw from "@/data/distinctive.json";
import divergenceRaw from "@/data/divergence.json";
import metaRaw from "@/data/meta.json";
import moversRaw from "@/data/movers.json";
import partyFramesRaw from "@/data/party_frames.json";
import trajectoriesRaw from "@/data/trajectories.json";
import yearSignaturesRaw from "@/data/year_signatures.json";
import type { PartyCode } from "@/lib/parties";

export interface WordScore {
  word: string;
  z: number;
}

export interface Meta {
  count: number;
  first_year: number;
  last_year: number;
  per_year: Record<string, number>;
  generated_at: string;
}

export interface Movers {
  split_year: number;
  risers: WordScore[];
  fallers: WordScore[];
}

export interface Trajectories {
  terms: string[];
  series: Record<string, Record<string, number>>;
}

/** {year: divergence} */
export type Divergence = Record<string, number>;

/** {year: [{word, z}, ...]} */
export type YearSignatures = Record<string, WordScore[]>;

/** {party: [{word, z}, ...]} */
export type Distinctive = Partial<Record<PartyCode, WordScore[]>>;

export interface PartyFrames {
  frames: string[];
  /** {frame: {party: {year: rate}}} */
  series: Record<string, Partial<Record<PartyCode, Record<string, number>>>>;
}

export const meta = metaRaw as Meta;
export const movers = moversRaw as Movers;
export const trajectories = trajectoriesRaw as Trajectories;
export const divergence = divergenceRaw as Divergence;
export const yearSignatures = yearSignaturesRaw as YearSignatures;
export const distinctive = distinctiveRaw as Distinctive;
export const partyFrames = partyFramesRaw as PartyFrames;

export const YEARS: number[] = Object.keys(yearSignatures)
  .map(Number)
  .sort((a, b) => a - b);
