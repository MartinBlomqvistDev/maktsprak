import { NextResponse } from "next/server";

/**
 * Proxies the page's questions to the search service, a FastAPI service on
 * Cloud Run, as `predict` proxies to the inference service. The service's
 * address and the shared secret stay in the server environment and never reach
 * the browser, and the service answers only requests that carry the secret.
 *
 * The service answers in lines of JSON, one per part, and the lines are passed
 * on as they come, so the page shows the list of results while the selector is
 * still choosing. The caller's address goes to the service for its rate limit,
 * which keeps it in memory only. Without the env vars set this returns a typed
 * "not yet connected" response, as `predict` does.
 */

// A cold start of the service, the wait of up to 30 s for the question's
// vector, and the selector, with room for its one retry.
export const maxDuration = 300;

const PATHS = { search: "/search", answer: "/answer", parties: "/parties" } as const;
type Mode = keyof typeof PATHS;

const MAX_QUESTION = 400;

function failure(status: number, error: string, headers?: Record<string, string>) {
  return NextResponse.json({ error }, { status, headers });
}

function isMode(value: unknown): value is Mode {
  return typeof value === "string" && Object.hasOwn(PATHS, value);
}

function callerOf(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  return request.headers.get("x-real-ip") ?? forwarded ?? "unknown";
}

export async function POST(request: Request) {
  const serviceUrl = process.env.SEARCH_SERVICE_URL;
  const secret = process.env.SEARCH_SHARED_SECRET;

  if (!serviceUrl || !secret) {
    return NextResponse.json(
      {
        error: "search_not_connected",
        message:
          "Söktjänsten är inte ansluten ännu. Sätt SEARCH_SERVICE_URL och SEARCH_SHARED_SECRET.",
      },
      { status: 501 }
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return failure(400, "invalid_request");
  }
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    return failure(400, "invalid_request");
  }
  const { question, filters, mode } = body as Record<string, unknown>;
  const trimmed = typeof question === "string" ? question.trim() : "";
  const filtersOk =
    filters === undefined ||
    filters === null ||
    (typeof filters === "object" && !Array.isArray(filters));
  if (trimmed.length < 1 || trimmed.length > MAX_QUESTION || !isMode(mode) || !filtersOk) {
    return failure(400, "invalid_request");
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${serviceUrl}${PATHS[mode]}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Search-Key": secret,
        "X-Caller": callerOf(request),
      },
      body: JSON.stringify({ question: trimmed, filters: filters ?? {} }),
      cache: "no-store",
    });
  } catch {
    return failure(502, "search_failed");
  }

  if (upstream.status === 400) {
    return failure(400, "invalid_request");
  }
  if (upstream.status === 429) {
    return failure(429, "rate_limited", {
      "Retry-After": upstream.headers.get("retry-after") ?? "60",
    });
  }
  if (!upstream.ok || !upstream.body) {
    return failure(502, "search_failed");
  }

  return new Response(upstream.body, {
    status: 200,
    headers: { "Content-Type": "application/x-ndjson", "Cache-Control": "no-store" },
  });
}
