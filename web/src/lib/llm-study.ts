import type { PartyCode } from "./parties";

/**
 * Results of the LLM writing-style study (research/llm_language_profile.py).
 * Re-classified 2026-07-19 with the DEPLOYED classifier (maktsprak_classifier_clean),
 * the same model the site serves, after an earlier run used a stale local checkpoint.
 * 14 models, 8 topics, 2 conditions, 3 samples per cell, 669 generations.
 *
 * dev = mean classifier probability minus the instrument baseline (the same model
 * averaged over a party-balanced sample of real speech). rawM/rawL are the
 * uncorrected mean shares (speech condition).
 */

export interface ModelProfile {
  name: string;
  id: string;
  dev: Record<PartyCode, number>;
  rawM?: number;
  rawL?: number;
}

/** Instrument baseline: mean output of the deployed model on balanced real speech. */
export const REFERENCE: Record<PartyCode, number> = {
  V: 0.09, MP: 0.1, S: 0.25, C: 0.08, L: 0.06, KD: 0.08, M: 0.21, SD: 0.14,
};

/** Speech condition, sorted by deviation toward M. */
export const SPEECH: ModelProfile[] = [
  { name: "Claude Opus 4.8", id: "anthropic/claude-opus-4.8", dev: { C: -0.07, KD: -0.08, L: -0.05, M: 0.42, MP: -0.06, S: -0.06, SD: -0.02, V: -0.08 }, rawM: 0.63, rawL: 0.01 },
  { name: "DeepSeek V4 Pro", id: "deepseek/deepseek-v4-pro", dev: { C: -0.07, KD: -0.08, L: -0.06, M: 0.33, MP: -0.05, S: 0.02, SD: -0.01, V: -0.08 }, rawM: 0.53, rawL: 0.01 },
  { name: "GPT-5.4", id: "openai/gpt-5.4", dev: { C: -0.03, KD: -0.08, L: -0.05, M: 0.18, MP: 0.03, S: 0.15, SD: -0.12, V: -0.08 }, rawM: 0.39, rawL: 0.01 },
  { name: "Claude Sonnet 4.6", id: "anthropic/claude-sonnet-4.6", dev: { C: -0.07, KD: 0.06, L: -0.05, M: 0.18, MP: -0.01, S: -0.02, SD: -0.01, V: -0.08 }, rawM: 0.39, rawL: 0.01 },
  { name: "Nemotron 3 Ultra", id: "nvidia/nemotron-3-ultra-550b-a55b", dev: { C: -0.02, KD: 0.02, L: -0.05, M: 0.18, MP: -0.09, S: 0.04, SD: 0.00, V: -0.08 }, rawM: 0.38, rawL: 0.01 },
  { name: "Grok 4.3", id: "x-ai/grok-4.3", dev: { C: -0.07, KD: -0.00, L: 0.02, M: 0.15, MP: 0.09, S: -0.04, SD: -0.07, V: -0.08 }, rawM: 0.35, rawL: 0.08 },
  { name: "Gemini 3.1 Pro", id: "google/gemini-3.1-pro-preview", dev: { C: -0.02, KD: -0.03, L: 0.07, M: 0.13, MP: -0.05, S: -0.08, SD: 0.05, V: -0.08 }, rawM: 0.34, rawL: 0.13 },
  { name: "Gemma 4 31B", id: "google/gemma-4-31b-it", dev: { C: -0.07, KD: -0.05, L: -0.02, M: 0.13, MP: -0.01, S: 0.02, SD: 0.08, V: -0.08 }, rawM: 0.33, rawL: 0.05 },
  { name: "Gemini 3.5 Flash", id: "google/gemini-3.5-flash", dev: { C: -0.06, KD: -0.07, L: 0.11, M: 0.12, MP: -0.09, S: 0.01, SD: 0.07, V: -0.08 }, rawM: 0.33, rawL: 0.17 },
  { name: "GPT-5.5", id: "openai/gpt-5.5", dev: { C: -0.07, KD: -0.04, L: -0.04, M: 0.11, MP: 0.10, S: 0.11, SD: -0.12, V: -0.06 }, rawM: 0.31, rawL: 0.02 },
  { name: "Kimi K2.6", id: "moonshotai/kimi-k2.6", dev: { C: -0.07, KD: -0.05, L: -0.02, M: 0.10, MP: -0.05, S: 0.03, SD: 0.13, V: -0.08 }, rawM: 0.31, rawL: 0.05 },
  { name: "MiniMax M3", id: "minimax/minimax-m3", dev: { C: -0.07, KD: 0.04, L: -0.02, M: 0.05, MP: 0.11, S: 0.03, SD: -0.06, V: -0.08 }, rawM: 0.26, rawL: 0.05 },
  { name: "GLM 5.2", id: "z-ai/glm-5.2", dev: { C: -0.07, KD: -0.03, L: -0.06, M: -0.00, MP: 0.00, S: -0.04, SD: 0.28, V: -0.08 }, rawM: 0.20, rawL: 0.01 },
  { name: "Qwen 3.6 Plus", id: "qwen/qwen3.6-plus", dev: { C: -0.06, KD: -0.03, L: -0.05, M: -0.03, MP: 0.03, S: -0.03, SD: 0.25, V: -0.08 }, rawM: 0.18, rawL: 0.01 },
];

/** Neutral control (civil-servant register), same model order as SPEECH. */
export const NEUTRAL: ModelProfile[] = [
  { name: "Claude Opus 4.8", id: "anthropic/claude-opus-4.8", dev: { C: -0.07, KD: -0.08, L: -0.05, M: -0.19, MP: 0.03, S: 0.40, SD: 0.04, V: -0.08 } },
  { name: "DeepSeek V4 Pro", id: "deepseek/deepseek-v4-pro", dev: { C: -0.07, KD: -0.04, L: -0.06, M: -0.12, MP: -0.06, S: 0.23, SD: 0.18, V: -0.08 } },
  { name: "GPT-5.4", id: "openai/gpt-5.4", dev: { C: -0.03, KD: -0.07, L: -0.06, M: -0.15, MP: -0.03, S: 0.42, SD: -0.00, V: -0.08 } },
  { name: "Claude Sonnet 4.6", id: "anthropic/claude-sonnet-4.6", dev: { C: -0.07, KD: -0.07, L: -0.05, M: -0.19, MP: 0.03, S: 0.06, SD: 0.34, V: -0.04 } },
  { name: "Nemotron 3 Ultra", id: "nvidia/nemotron-3-ultra-550b-a55b", dev: { C: -0.07, KD: -0.07, L: -0.05, M: -0.12, MP: 0.05, S: 0.05, SD: 0.30, V: -0.07 } },
{ name: "Grok 4.3", id: "x-ai/grok-4.3", dev: { C: -0.07, KD: -0.07, L: -0.05, M: -0.07, MP: -0.05, S: 0.27, SD: 0.12, V: -0.08 } },
  { name: "Gemini 3.1 Pro", id: "google/gemini-3.1-pro-preview", dev: { C: -0.06, KD: -0.06, L: -0.05, M: -0.09, MP: -0.07, S: 0.14, SD: 0.27, V: -0.07 } },
  { name: "Gemma 4 31B", id: "google/gemma-4-31b-it", dev: { C: -0.07, KD: -0.04, L: -0.06, M: -0.11, MP: -0.01, S: 0.20, SD: 0.16, V: -0.08 } },
  { name: "Gemini 3.5 Flash", id: "google/gemini-3.5-flash", dev: { C: -0.06, KD: 0.05, L: -0.01, M: -0.07, MP: -0.09, S: -0.06, SD: 0.33, V: -0.08 } },
  { name: "GPT-5.5", id: "openai/gpt-5.5", dev: { C: -0.06, KD: -0.04, L: -0.05, M: -0.19, MP: 0.00, S: 0.23, SD: 0.19, V: -0.08 } },
  { name: "Kimi K2.6", id: "moonshotai/kimi-k2.6", dev: { C: -0.07, KD: 0.05, L: -0.02, M: -0.13, MP: 0.02, S: 0.27, SD: -0.04, V: -0.08 } },
  { name: "MiniMax M3", id: "minimax/minimax-m3", dev: { C: -0.07, KD: -0.08, L: -0.06, M: -0.20, MP: 0.00, S: 0.27, SD: 0.21, V: -0.08 } },
  { name: "GLM 5.2", id: "z-ai/glm-5.2", dev: { C: -0.07, KD: -0.04, L: -0.06, M: -0.15, MP: -0.01, S: 0.34, SD: 0.07, V: -0.08 } },
  { name: "Qwen 3.6 Plus", id: "qwen/qwen3.6-plus", dev: { C: -0.07, KD: -0.04, L: -0.05, M: -0.15, MP: 0.02, S: 0.03, SD: 0.35, V: -0.08 } },
];

export const STUDY_META = {
  models: 14, providers: 10, topics: 8, samplesPerCell: 3, generations: 669,
  calibrationAccuracy: 0.475, calibrationN: 480, runDate: "2026-07-19",
};
