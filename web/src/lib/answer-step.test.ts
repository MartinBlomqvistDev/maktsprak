import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { answerStepRuns, answerStepState, answerStepSwitch } from "./answer-step.ts";

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

  it("runs it with the switch on only once the launch rule holds", () => {
    assert.equal(answerStepRuns(true, true), true);
    assert.equal(answerStepRuns(true, null), false);
    assert.equal(answerStepRuns(true, false), false);
  });
});

describe("where the answer step stands", () => {
  it("is pending while there is no report, whatever the switch says", () => {
    assert.equal(answerStepState(false, null), "pending");
    assert.equal(answerStepState(true, null), "pending");
  });

  it("is failed when the launch rule fails, whatever the switch says", () => {
    assert.equal(answerStepState(false, false), "failed");
    assert.equal(answerStepState(true, false), "failed");
  });

  it("is passed when the rule holds and the switch is off", () => {
    assert.equal(answerStepState(false, true), "passed");
  });

  it("is on only when the switch is on and the rule holds", () => {
    assert.equal(answerStepState(true, true), "on");
  });
});
