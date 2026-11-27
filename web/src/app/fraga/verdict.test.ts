import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { stopNote, verdict } from "./verdict.ts";

describe("the verdict under the test report", () => {
  it("claims no selector when the launch rule fails", () => {
    for (const beats of [true, false]) {
      const text = verdict(beats, "failed");
      assert.match(text, /Den klarar inte lanseringsregeln, så sidan visar bara träffarna\.$/);
      assert.doesNotMatch(text, /väljer citaten på sidan/);
    }
  });

  it("names the language model when it runs", () => {
    assert.equal(
      verdict(true, "on"),
      "Språkmodellen slår baslinjen på båda måtten, så det är den som ska välja citaten. Den klarar lanseringsregeln och väljer citaten på sidan.",
    );
  });

  it("names the baseline when the comparison puts it on the page", () => {
    assert.match(verdict(false, "on"), /det är baslinjen som ska välja citaten\. Den klarar lanseringsregeln och väljer citaten på sidan\.$/);
  });

  it("says the step is off when the rule holds and the switch is off", () => {
    assert.match(verdict(true, "passed"), /men svarssteget är inte påslaget på sidan än\.$/);
  });
});

describe("the note above the results in stop mode", () => {
  it("is pending only while the report is not in", () => {
    assert.match(stopNote("pending"), /när mätningen av dem är klar/);
    assert.doesNotMatch(stopNote("failed"), /när mätningen/);
    assert.doesNotMatch(stopNote("passed"), /när mätningen/);
  });

  it("states a failed launch rule plainly", () => {
    assert.match(stopNote("failed"), /klarade inte kraven/);
  });
});
