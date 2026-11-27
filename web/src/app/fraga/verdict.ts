import type { AnswerStepState } from "../../lib/answer-step.ts";

/**
 * What the page says about the answer step's verdict, for each state of the
 * switch and the test report. The launch rule comes first: when it fails, the
 * page says so and claims no selector, whatever the comparison found.
 */

/** The note above the results when the page does not run the answer step. */
export function stopNote(state: Exclude<AnswerStepState, "on">): string {
  if (state === "failed") {
    return "Citatvalet klarade inte kraven som jag satte i förväg, så sidan visar bara träffarna. Mätningen står under Citaten i metoden nedan.";
  }
  if (state === "passed") {
    return "Citatvalet klarade kraven men är inte påslaget på sidan än.";
  }
  return "Citat ur träffarna visas här när mätningen av dem är klar, om de klarar den.";
}

/**
 * The verdict under the test report's table. The comparison decides which
 * selector the page would run, as the report's `runs` records it, and the
 * launch rule is then judged for that selector.
 *
 * @param beats Whether the language model beat the baseline on both measures.
 * @param state Where the answer step stands.
 */
export function verdict(beats: boolean, state: AnswerStepState): string {
  const comparison = beats
    ? "Språkmodellen slår baslinjen på båda måtten, så det är den som ska välja citaten."
    : "Språkmodellen slår inte baslinjen på båda måtten, så det är baslinjen som ska välja citaten.";
  if (state === "failed") {
    return `${comparison} Den klarar inte lanseringsregeln, så sidan visar bara träffarna.`;
  }
  if (state === "on") return `${comparison} Den klarar lanseringsregeln och väljer citaten på sidan.`;
  return `${comparison} Den klarar lanseringsregeln, men svarssteget är inte påslaget på sidan än.`;
}
