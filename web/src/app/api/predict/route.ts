import { NextResponse } from "next/server";

/**
 * Proxies to the FastAPI inference service, a `POST /predict` endpoint on Cloud
 * Run. The upstream URL stays in the server environment and is never sent to the
 * browser, so the service is only reachable through this route. Without the env
 * var set this returns a typed "not yet connected" response rather than a
 * fabricated prediction; the UI shows that state gracefully instead of erroring.
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
