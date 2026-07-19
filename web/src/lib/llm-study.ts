import type { PartyCode } from "./parties";

/**
 * Results of the LLM writing-style study (research/llm_language_profile.py).
 * Numbers are transcribed from research/out/profiles_speech.csv and
 * profiles_neutral.csv, run of 2026-07-19: 14 models, 8 topics, 2 conditions,
 * 3 samples per cell, 669 generations total.
 *
 * dev = mean classifier probability minus the instrument baseline (the same
 * classifier averaged over a party-balanced sample of real speeches). rawM and
 * rawL are the uncorrected mean shares for the two headline parties.
 */

export interface ModelProfile {
  /** Display name */
  name: string;
  /** OpenRouter model id, kept for the dataset cross-reference */
  id: string;
  /** Deviation from instrument baseline per party */
  dev: Record<PartyCode, number>;
  /** Uncorrected mean share for M and L (speech condition only) */
  rawM?: number;
  rawL?: number;
}

/** Instrument baseline: mean classifier output on party-balanced real speech. */
export const REFERENCE: Record<PartyCode, number> = {
  V: 0.12,
  MP: 0.13,
  S: 0.11,
  C: 0.09,
  L: 0.07,
  KD: 0.1,
  M: 0.21,
  SD: 0.15,
};

/** Speech condition, sorted by deviation toward M (the headline axis). */
export const SPEECH: ModelProfile[] = [
  {
    name: "Claude Opus 4.8",
    id: "anthropic/claude-opus-4.8",
    dev: { V: -0.1, MP: -0.08, S: 0.04, C: -0.08, L: 0.12, KD: -0.08, M: 0.3, SD: -0.11 },
    rawM: 0.51,
    rawL: 0.2,
  },
  {
    name: "Claude Sonnet 4.6",
    id: "anthropic/claude-sonnet-4.6",
    dev: { V: -0.1, MP: -0.06, S: -0.02, C: -0.08, L: 0.03, KD: -0.04, M: 0.28, SD: -0.02 },
    rawM: 0.5,
    rawL: 0.11,
  },
  {
    name: "Kimi K2.6",
    id: "moonshotai/kimi-k2.6",
    dev: { V: -0.07, MP: -0.01, S: 0.04, C: -0.07, L: -0.01, KD: -0.09, M: 0.19, SD: 0.02 },
    rawM: 0.41,
  },
  {
    name: "GPT-5.4",
    id: "openai/gpt-5.4",
    dev: { V: -0.09, MP: -0.07, S: 0.06, C: -0.03, L: 0.14, KD: -0.08, M: 0.18, SD: -0.1 },
    rawM: 0.39,
  },
  {
    name: "Gemma 4 31B",
    id: "google/gemma-4-31b-it",
    dev: { V: -0.1, MP: -0.1, S: -0.03, C: -0.06, L: 0.14, KD: -0.09, M: 0.18, SD: 0.06 },
    rawM: 0.39,
  },
  {
    name: "DeepSeek V4 Pro",
    id: "deepseek/deepseek-v4-pro",
    dev: { V: -0.09, MP: -0.04, S: 0.0, C: -0.08, L: 0.13, KD: -0.09, M: 0.16, SD: 0.0 },
    rawM: 0.38,
  },
  {
    name: "Nemotron 3 Ultra",
    id: "nvidia/nemotron-3-ultra-550b-a55b",
    dev: { V: -0.09, MP: -0.08, S: 0.08, C: -0.08, L: 0.11, KD: -0.08, M: 0.16, SD: -0.02 },
    rawM: 0.38,
  },
  {
    name: "GLM 5.2",
    id: "z-ai/glm-5.2",
    dev: { V: -0.06, MP: -0.04, S: 0.0, C: -0.08, L: -0.01, KD: -0.04, M: 0.14, SD: 0.08 },
    rawM: 0.36,
  },
  {
    name: "GPT-5.5",
    id: "openai/gpt-5.5",
    dev: { V: -0.08, MP: -0.01, S: 0.11, C: -0.03, L: 0.09, KD: -0.08, M: 0.14, SD: -0.13 },
    rawM: 0.35,
  },
  {
    name: "Grok 4.3",
    id: "x-ai/grok-4.3",
    dev: { V: -0.08, MP: -0.02, S: 0.04, C: -0.07, L: 0.1, KD: -0.05, M: 0.13, SD: -0.05 },
    rawM: 0.34,
  },
  {
    name: "MiniMax M3",
    id: "minimax/minimax-m3",
    dev: { V: -0.09, MP: 0.1, S: 0.05, C: -0.08, L: 0.08, KD: -0.09, M: 0.04, SD: -0.02 },
    rawM: 0.26,
  },
  {
    name: "Qwen 3.6 Plus",
    id: "qwen/qwen3.6-plus",
    dev: { V: -0.09, MP: -0.02, S: 0.03, C: -0.08, L: 0.05, KD: -0.07, M: 0.01, SD: 0.17 },
    rawM: 0.22,
  },
  {
    name: "Gemini 3.5 Flash",
    id: "google/gemini-3.5-flash",
    dev: { V: 0.0, MP: -0.03, S: -0.02, C: -0.01, L: 0.18, KD: -0.01, M: -0.03, SD: -0.08 },
    rawM: 0.19,
    rawL: 0.25,
  },
  {
    name: "Gemini 3.1 Pro",
    id: "google/gemini-3.1-pro-preview",
    dev: { V: -0.01, MP: -0.04, S: -0.02, C: -0.02, L: 0.23, KD: 0.0, M: -0.05, SD: -0.09 },
    rawM: 0.17,
    rawL: 0.3,
  },
];

/** Neutral control (civil-servant register), same row order as SPEECH. */
export const NEUTRAL: ModelProfile[] = [
  {
    name: "Claude Opus 4.8",
    id: "anthropic/claude-opus-4.8",
    dev: { V: -0.08, MP: -0.03, S: 0.07, C: -0.07, L: -0.04, KD: -0.04, M: -0.06, SD: 0.25 },
  },
  {
    name: "Claude Sonnet 4.6",
    id: "anthropic/claude-sonnet-4.6",
    dev: { V: -0.09, MP: 0.01, S: 0.03, C: -0.07, L: -0.06, KD: -0.05, M: -0.06, SD: 0.3 },
  },
  {
    name: "Kimi K2.6",
    id: "moonshotai/kimi-k2.6",
    dev: { V: -0.08, MP: 0.05, S: 0.16, C: -0.07, L: -0.05, KD: -0.01, M: -0.02, SD: 0.03 },
  },
  {
    name: "GPT-5.4",
    id: "openai/gpt-5.4",
    dev: { V: -0.07, MP: 0.01, S: 0.21, C: -0.03, L: -0.05, KD: -0.01, M: -0.04, SD: -0.02 },
  },
  {
    name: "Gemma 4 31B",
    id: "google/gemma-4-31b-it",
    dev: { V: -0.08, MP: -0.01, S: 0.08, C: -0.06, L: -0.05, KD: -0.05, M: -0.05, SD: 0.2 },
  },
  {
    name: "DeepSeek V4 Pro",
    id: "deepseek/deepseek-v4-pro",
    dev: { V: -0.09, MP: 0.06, S: 0.08, C: -0.06, L: -0.03, KD: -0.03, M: -0.03, SD: 0.1 },
  },
  {
    name: "Nemotron 3 Ultra",
    id: "nvidia/nemotron-3-ultra-550b-a55b",
    dev: { V: -0.08, MP: 0.04, S: 0.03, C: -0.06, L: -0.05, KD: -0.04, M: -0.12, SD: 0.27 },
  },
  {
    name: "GLM 5.2",
    id: "z-ai/glm-5.2",
    dev: { V: -0.07, MP: -0.01, S: 0.11, C: -0.06, L: -0.05, KD: -0.03, M: -0.07, SD: 0.19 },
  },
  {
    name: "GPT-5.5",
    id: "openai/gpt-5.5",
    dev: { V: -0.05, MP: 0.03, S: 0.08, C: -0.05, L: -0.04, KD: 0.01, M: -0.1, SD: 0.13 },
  },
  {
    name: "Grok 4.3",
    id: "x-ai/grok-4.3",
    dev: { V: -0.08, MP: -0.02, S: 0.09, C: -0.05, L: -0.05, KD: -0.04, M: -0.05, SD: 0.2 },
  },
  {
    name: "MiniMax M3",
    id: "minimax/minimax-m3",
    dev: { V: -0.08, MP: -0.03, S: 0.13, C: -0.06, L: -0.05, KD: -0.05, M: -0.05, SD: 0.19 },
  },
  {
    name: "Qwen 3.6 Plus",
    id: "qwen/qwen3.6-plus",
    dev: { V: -0.09, MP: 0.02, S: 0.04, C: -0.06, L: -0.05, KD: -0.06, M: -0.1, SD: 0.28 },
  },
  {
    name: "Gemini 3.5 Flash",
    id: "google/gemini-3.5-flash",
    dev: { V: 0.02, MP: -0.01, S: -0.03, C: 0.01, L: 0.06, KD: -0.01, M: -0.04, SD: 0.0 },
  },
  {
    name: "Gemini 3.1 Pro",
    id: "google/gemini-3.1-pro-preview",
    dev: { V: 0.01, MP: 0.0, S: 0.01, C: -0.02, L: 0.02, KD: -0.02, M: -0.02, SD: 0.03 },
  },
];

export const STUDY_META = {
  models: 14,
  providers: 10,
  topics: 8,
  samplesPerCell: 3,
  generations: 669,
  calibrationAccuracy: 0.488,
  calibrationN: 480,
  runDate: "2026-07-19",
};
