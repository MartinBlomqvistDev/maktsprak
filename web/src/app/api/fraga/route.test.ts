import { strict as assert } from "node:assert";
import { afterEach, beforeEach, describe, it } from "node:test";
import { POST } from "./route.ts";

/**
 * The proxy in both positions of the answer step's switch. The search service
 * is replaced by a stand-in for fetch that records where it was called and
 * answers with one line, so nothing leaves the machine.
 */

const realFetch = globalThis.fetch;
let called: string[] = [];

function ask(mode: string, question = "Vad har sagts om tandvården?") {
  return POST(
    new Request("http://localhost/api/fraga", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, filters: {}, mode }),
    }),
  );
}

beforeEach(() => {
  process.env.SEARCH_SERVICE_URL = "https://search.example";
  process.env.SEARCH_SHARED_SECRET = "test-secret";
  delete process.env.FRAGA_ANSWER_STEP;
  called = [];
  globalThis.fetch = (async (input: string | URL | Request) => {
    called.push(String(input));
    return new Response('{"type":"search"}\n', {
      status: 200,
      headers: { "Content-Type": "application/x-ndjson" },
    });
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("with the answer step off, as it is unless the environment turns it on", () => {
  it("passes a search through to the service", async () => {
    const response = await ask("search");
    assert.equal(response.status, 200);
    assert.deepEqual(called, ["https://search.example/search"]);
  });

  for (const mode of ["answer", "parties"]) {
    it(`refuses ${mode} without calling the service`, async () => {
      const response = await ask(mode);
      assert.equal(response.status, 403);
      assert.deepEqual(await response.json(), { error: "answer_step_off" });
      assert.deepEqual(called, []);
    });
  }

  for (const value of ["true", "1", "ON", "on "]) {
    it(`reads FRAGA_ANSWER_STEP=${JSON.stringify(value)} as off`, async () => {
      process.env.FRAGA_ANSWER_STEP = value;
      const response = await ask("answer");
      assert.equal(response.status, 403);
      assert.deepEqual(called, []);
    });
  }

  it("still answers an invalid request with 400", async () => {
    const response = await ask("answer", "   ");
    assert.equal(response.status, 400);
    assert.deepEqual(called, []);
  });
});

describe("with the answer step on", () => {
  beforeEach(() => {
    process.env.FRAGA_ANSWER_STEP = "on";
  });

  for (const mode of ["search", "answer", "parties"]) {
    it(`sends ${mode} to the service's /${mode}`, async () => {
      const response = await ask(mode);
      assert.equal(response.status, 200);
      assert.deepEqual(called, [`https://search.example/${mode}`]);
    });
  }
});
