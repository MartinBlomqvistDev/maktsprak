import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { answerStepRuns, answerStepSwitch } from "./answer-step.ts";

describe("the answer step's switch", () => {
  it("is off when the variable is missing", () => {
    assert.equal(answerStepSwitch({}), false);
  });

  it("is on only for the exact value on", () => {
    assert.equal(answerStepSwitch({ FRAGA_ANSWER_STEP: "on" }), true);
    for (const value of ["", "true", "1", "ON", "yes", " on"]) {
      assert.equal(answerStepSwitch({ FRAGA_ANSWER_STEP: value }), false, value);
    }
  });
});

describe("whether the page runs the answer step", () => {
  it("never runs it with the switch off", () => {
    for (const launch of [true, false, null]) {
      assert.equal(answerStepRuns(false, launch), false, String(launch));
    }
  });

  it("runs it with the switch on unless the launch rule failed", () => {
    assert.equal(answerStepRuns(true, null), true);
    assert.equal(answerStepRuns(true, true), true);
    assert.equal(answerStepRuns(true, false), false);
  });
});
