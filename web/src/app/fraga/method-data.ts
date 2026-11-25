import methodRaw from "@/data/fraga/method.json";

/**
 * The figures of the method section, written by search/pipeline/15_build_page.py
 * from the reports in data/reports/search/. No figure on the page is typed by
 * hand: each one is read from here, and each source report is recorded with
 * the SHA-256 of its text.
 */

export interface MethodFigures {
  sources: Record<string, string>;
  design: {
    candidates: number;
    max_quotes: number;
    prompt_rounds: number;
    service_vcpus: number;
    interval_level: number;
  };
  corpus: {
    speeches: number;
    first_date: string;
    last_date: string;
    paragraphs: number;
    words: number;
    passages: number;
    passage_target_words: number;
    passage_median_words: number;
    left_out_session: string;
    left_out_hard_wrapped: number;
    left_out_other_parties: number;
    set_aside_paragraphs: number;
    set_aside_rules: number;
    coverage_paired: number;
    coverage_of: number;
    coverage: number;
    coverage_threshold: number;
    links_sampled: number;
    links_passed: number;
  };
  questions: {
    sets: { name: string; questions: number; dev: number; test: number }[];
    model: string;
    review_approved: number;
    review_drawn: number;
    review_threshold: number;
  };
  judge: {
    pairs: number;
    kappa: number;
    threshold: number;
    accepted: boolean;
    exact: number;
    disagreements: number;
    stricter: number;
  };
  retrieval: {
    embedding: string;
    embedding_model: string;
    provider: string;
    precision: string;
    dimensions: number;
    subset_passages: number;
    embedding_scores: { name: string; recall_10: number; licensed: boolean }[];
    passage_words: number;
    rrf_k: number;
    window: number;
    tuned_recall_10: number;
    bm25_k1: number;
    bm25_b: number;
    outcome: string;
    production: string;
    min_difference: number;
    comparisons: {
      question_set: string;
      system: string;
      against: string;
      questions: number;
      system_mean: number;
      against_mean: number;
      difference: number;
      low: number;
      high: number;
    }[];
    sets: { name: string; questions: number; bm25: number; vectors: number; hybrid: number }[];
    natural_mean: number;
    parties: { party: string; questions: number; recall_10: number; deviation: number }[];
    party_margin: number;
  };
  answer_dev: {
    answerable: number;
    unanswerable: number;
    rounds: {
      name: string;
      balanced_accuracy: number;
      abstains_unanswerable: number;
      needless_abstention: number;
    }[];
    chosen_round: number;
  };
  answer_test: {
    model: string;
    effort: string;
    questions: number;
    first_retrieval: string;
    first_cost_usd: number;
    hybrid_cost_usd: number;
    hand_questions: number;
    hand_pairs: number;
    launch_rule: {
      abstains_unanswerable: number;
      needless_abstention: number;
      quote_precision: number;
    };
    report: {
      drawn_questions: number;
      graded_pairs: number;
      selectors: {
        name: string;
        balanced_accuracy: number;
        abstains_unanswerable: number;
        needless_abstention: number;
        quote_precision: number | null;
        graded_quotes: number;
        parties: Record<string, number>;
      }[];
      precision_difference: [number, number, number] | null;
      accuracy_difference: [number, number, number];
      beats_baseline: boolean;
      runs: string;
      launch: boolean;
      launch_rule: Record<string, boolean>;
      candidate_parties: Record<string, number>;
    } | null;
    sealed_balanced_accuracy: Record<string, number> | null;
  };
  operations: {
    vector_questions: number;
    vector_wait_p50_s: number;
    vector_wait_p95_s: number;
    vector_wait_max_s: number;
    vector_wait_short_s: number;
    vector_wait_over_short: number;
    vector_limit_s: number;
    search_p95_ms: number;
    search_p95_budget_ms: number;
    cold_start_s: number;
    cold_start_budget_s: number;
    memory_peak_gib: number;
    memory_budget_gib: number;
    monthly_budget_usd: number;
    cost_per_question_usd: number;
    questions_per_month: number;
    party_views_per_month: number;
  };
  links: {
    speeches: number;
    passed: number;
    passed_with_folded_dashes: number;
    xml_records: number;
    dashless_from: string;
    dashless_to: string;
  };
}

export const figures = methodRaw as MethodFigures;

export function questionSet(name: string) {
  const found = figures.questions.sets.find((set) => set.name === name);
  if (!found) throw new Error(`method.json has no question set ${name}`);
  return found;
}

export function comparison(questionSet: string, system: string) {
  const found = figures.retrieval.comparisons.find(
    (entry) => entry.question_set === questionSet && entry.system === system,
  );
  if (!found) throw new Error(`method.json has no comparison of ${system} on ${questionSet}`);
  return found;
}

export function round(name: string) {
  const found = figures.answer_dev.rounds.find((entry) => entry.name === name);
  if (!found) throw new Error(`method.json has no round ${name}`);
  return found;
}
