import { readFileSync } from "node:fs";
import path from "node:path";
import type { Metadata } from "next";
import examplesRaw from "@/data/fraga/examples.json";
import { answerStepState, answerStepSwitch } from "@/lib/answer-step";
import { FragaApp } from "./FragaApp";
import { Method } from "./Method";
import { figures } from "./method-data";
import type { Example, ExamplesFile } from "./types";

const examples = examplesRaw as ExamplesFile;
const firstYear = Number(figures.corpus.first_date.slice(0, 4));
const lastYear = Number(figures.corpus.last_date.slice(0, 4));

// Read when the page is built, as the switch is: a change takes effect with the
// next deployment.
const report = figures.answer_test.report;
const answerState = answerStepState(answerStepSwitch(), report === null ? null : report.launch);
const answerStep = answerState === "on";

const TITLE = "Fråga protokollet · Maktspråk / Protokollet";
const DESCRIPTION = answerStep
  ? `Ställ en fråga och få svar med ordagranna citat ur riksdagens anföranden ${firstYear}-${lastYear}, alltid med talare, parti och länk till anförandet.`
  : `Ställ en fråga och få de passager ur riksdagens anföranden ${firstYear}-${lastYear} som sökningen rankar högst, ordagrant och alltid med talare, parti och länk till anförandet.`;

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
 * opens on a real answer; the others are fetched when they are chosen. Without
 * the answer step the page is built with the results alone, so its source
 * holds no quotation the language model chose.
 */
function firstExample(): Example {
  const first = examples.examples[0];
  const file = path.join(process.cwd(), "public", "fraga", "exempel", `${first.id}.json`);
  const example = JSON.parse(readFileSync(file, "utf-8")) as Example;
  return answerStep ? example : { id: example.id, question: example.question, search: example.search };
}

export default function FragaPage() {
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
          answerState={answerState}
          vectorLimitS={figures.operations.vector_limit_s}
          coldStartS={figures.operations.proxy_cold_first_line_s}
        />
      </div>

      <Method answerState={answerState} />
    </main>
  );
}
