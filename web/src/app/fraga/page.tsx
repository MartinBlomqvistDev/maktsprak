import { readFileSync } from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import examplesRaw from "@/data/fraga/examples.json";
import { FragaApp } from "./FragaApp";
import { Method } from "./Method";
import { figures } from "./method-data";
import type { Example, ExamplesFile } from "./types";

const examples = examplesRaw as ExamplesFile;
const firstYear = Number(figures.corpus.first_date.slice(0, 4));
const lastYear = Number(figures.corpus.last_date.slice(0, 4));

const TITLE = "Fråga protokollet · Maktspråk / Protokollet";
const DESCRIPTION = `Ställ en fråga och få svar med ordagranna citat ur riksdagens anföranden ${firstYear}-${lastYear}, alltid med talare, parti och länk till anförandet.`;

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: {
    title: TITLE,
    description: DESCRIPTION,
    url: "https://maktsprak.se/fraga",
    siteName: "Maktspråk",
    locale: "sv_SE",
    type: "article",
    images: [{ url: "/og.png", width: 2400, height: 1260, alt: "Maktspråk" }],
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    images: ["/og.png"],
  },
};

/**
 * The first example, read from the static files at build time so the page
 * opens on a real answer; the others are fetched when they are chosen.
 */
function firstExample(): Example {
  const first = examples.examples[0];
  const file = path.join(process.cwd(), "public", "fraga", "exempel", `${first.id}.json`);
  return JSON.parse(readFileSync(file, "utf-8")) as Example;
}

export default function FragaPage() {
  const report = figures.answer_test.report;
  const answerStep = report === null || report.launch;
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 sm:py-20">
      <p className="anf-marker mb-4">
        <span className="tabular">Prot. 2026/27:7</span>
        <span className="text-line-2">·</span>
        <span>Sökning i anförandena</span>
      </p>
      <h1 className="font-display text-4xl leading-tight tracking-tight sm:text-5xl">
        Fråga protokollet
      </h1>
      {answerStep ? (
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-ink-2">
          Ställ en fråga på svenska, så svarar sidan med citat ur riksdagens anföranden. Citaten
          står ordagrant som i riksdagens öppna data, alltid med talare, parti, datum och länk. En
          språkmodell väljer vilka stycken som ska citeras, men texten hämtas direkt ur riksdagens
          data, så modellen kan inte ändra ett ord. När sidan inte hittar något svar säger den det.
        </p>
      ) : (
        <p className="mt-5 max-w-2xl text-lg leading-relaxed text-ink-2">
          Ställ en fråga på svenska, så svarar sidan med de passager ur riksdagens anföranden som
          sökningen rankar högst. Texten står ordagrant som i riksdagens öppna data, alltid med
          talare, parti, datum och länk.
        </p>
      )}

      <div className="mt-10">
        <FragaApp
          entries={examples.examples}
          initial={firstExample()}
          generated={examples.generated}
          firstYear={firstYear}
          lastYear={lastYear}
          answerStep={answerStep}
          vectorLimitS={figures.operations.vector_limit_s}
        />
      </div>

      <Method />
    </main>
  );
}
