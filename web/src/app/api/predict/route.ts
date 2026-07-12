import { NextResponse } from "next/server";

/**
 * Proxies to the FastAPI inference service (Phase 2 of the rebuild — a
 * `POST /predict` endpoint hosted on Hugging Face Spaces). Until that
 * service exists, this returns a clear, typed "not yet connected" response
 * rather than a fabricated prediction — the UI is designed to show this
 * state gracefully instead of erroring.
 */
export async function POST(request: Request) {
  const inferenceUrl = process.env.INFERENCE_SERVICE_URL;

  if (!inferenceUrl) {
    return NextResponse.json(
      {
        error: "inference_not_connected",
        message:
          "Inferens-tjänsten är inte ansluten ännu. Sätt INFERENCE_SERVICE_URL.",
      },
      { status: 501 }
    );
  }

  const body = await request.json();
  const upstream = await fetch(`${inferenceUrl}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!upstream.ok) {
    return NextResponse.json(
      { error: "inference_failed", message: "Modellen kunde inte svara." },
      { status: 502 }
    );
  }

  const data = await upstream.json();
  return NextResponse.json(data);
}
