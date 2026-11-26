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
 * Whether the page runs the answer step: the switch is on, and the launch rule
 * has not failed. `launch` is the test report's verdict, or null before the
 * report exists.
 */
export function answerStepRuns(switchOn: boolean, launch: boolean | null): boolean {
  return switchOn && launch !== false;
}
