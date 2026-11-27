/**
 * The switch for the answer step of Fråga protokollet.
 *
 * The answer step is everything that chooses quotations or abstains: Claude
 * Opus 5.5 choosing paragraphs from the results, and the baseline choosing one
 * per party in the view with eight parties. With the switch off, the page is
 * the plan's stop mode, a search that returns the hybrid's ranked passages
 * with their speaker, party, date and link, and the proxy refuses `answer` and
 * `parties` outright rather than answering with something else.
 *
 * The switch is off unless the server's environment sets FRAGA_ANSWER_STEP to
 * "on". The page reads it when it is built and the proxy when it is called, so
 * a change in Vercel takes effect with the next deployment, for both at once.
 */

export const ANSWER_STEP_ENV = "FRAGA_ANSWER_STEP";

/** Whether the environment turns the answer step on. */
export function answerStepSwitch(env: Record<string, string | undefined> = process.env): boolean {
  return env[ANSWER_STEP_ENV] === "on";
}

/**
 * Where the answer step stands, which decides what the page says about it.
 * It fails closed: the step runs only once the test report has found that the
 * launch rule holds, as the search service itself requires.
 *
 * - `on`: the switch is on and the launch rule holds, so the page runs it.
 * - `pending`: the test report is not in yet, whatever the switch says.
 * - `failed`: the test report found that the launch rule does not hold, so the
 *   page never runs it, whatever the switch says.
 * - `passed`: the launch rule holds, but the switch is still off.
 */
export type AnswerStepState = "on" | "pending" | "failed" | "passed";

/**
 * Where the answer step stands. `launch` is the test report's verdict, or null
 * before the report exists.
 */
export function answerStepState(switchOn: boolean, launch: boolean | null): AnswerStepState {
  if (launch === null) return "pending";
  if (!launch) return "failed";
  return switchOn ? "on" : "passed";
}

/** Whether the page runs the answer step: the switch is on, and the launch rule holds. */
export function answerStepRuns(switchOn: boolean, launch: boolean | null): boolean {
  return answerStepState(switchOn, launch) === "on";
}
