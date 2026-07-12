# Maktspråk / Protokollet

The public site for [MaktspråkAI](https://github.com/MartinBlomqvistDev/MaktsprakAI) — a
fine-tuned KB-BERT classifier that reads Swedish Riksdag speeches and predicts
which party is speaking, trained on a talar-oberoende (speaker-independent)
split so the reported score reflects real generalisation, not memorised
politicians.

This repo is the Next.js frontend only. The ETL pipeline, PDF parser, and
model training live in the [Python repo](https://github.com/MartinBlomqvistDev/MaktsprakAI);
this site reads from the same Supabase database and calls a separate FastAPI
inference service for live predictions.

## Design: "Protokollet"

The identity is built from the project's own subject matter rather than a
generic template: the Riksdag's own transcript grammar (`Anf. 42 · NAMN
(PARTI):`) becomes the site's structural device, and the display face —
[Redaction](https://github.com/mgodefroy/BBB-Redaction-Regular) (SIL OFL,
drawn from scanned/degrading legal documents) — makes "recovering signal from
an official record" literal rather than decorative. The accent colour is
**blue pencil** (`#2451a4` / `#7fa8e0` dark), the actual editorial term for a
redaction/markup mark — deliberately not the cream-and-terracotta look most
AI-generated sites default to. The eight party colours appear only as *data*
(the spectrum bar, live prediction bars), never as UI chrome.

Tokens live in [`src/app/globals.css`](src/app/globals.css); both light and
dark themes are fully specified (`prefers-color-scheme` + a `data-theme`
override for the manual toggle).

## Stack

- Next.js 16 (App Router) + TypeScript + Tailwind v4
- Recharts (analytics pages, in progress)
- Supabase JS — reads the same `speeches` table as the Python pipeline
- A thin `/api/predict` proxy to a FastAPI inference service (not yet
  deployed — see `/metod` on the live site for status)

## Development

```bash
npm install
cp .env.example .env.local   # fill in Supabase credentials
npm run dev
```

`/api/predict` gracefully reports "not connected" (HTTP 501, handled in the
UI) until `INFERENCE_SERVICE_URL` is set — the interactive demo is fully
wired and ready for that service to exist.

## Routes

| Route | Status |
|---|---|
| `/` | Live — hero demo, corpus stats (from Supabase), method summary |
| `/metod` | Live — full methodology: the speaker-independent split, why a row-based split leaks, honest vs. inflated numbers |
| `/partierna` | Planned — rhetoric fingerprints, word clouds per party |
| `/utveckling` | Planned — how a party's language shifts across years |
| `/riktmarke` | Planned — BERT vs. frontier LLMs on the same held-out set |

## Deploy

Vercel, zero config beyond the environment variables in `.env.example`.
