import type { Metadata } from "next";
import { ProtokollMarker } from "@/components/ProtokollMarker";

export const metadata: Metadata = {
  title: "Behövs en egen modell? · Maktspråk / Protokollet",
  description:
    "Den finjusterade 110M-modellen mot sex frontier-LLM på party-klassificering, på modellens egna osedda talare. Träffsäkerhet mot kostnad och latens.",
};

const LLM_URL = "https://maktsprak.se/llm";

// Benchmark 2026-07-20, re-scored 2026-11-27: the DEPLOYED classifier vs six
// frontier LLMs on the 247 of 320 sampled speeches (2015-2026) whose speakers
// appear in training under no name, matched on the Riksdag person id. 73 rows
// came from 20 people trained under another form of their name and are left
// out (research/person_heldout.py). Training speakers score 0.959 in the same
// window, held-out 0.579. Every model reads KB-BERT's own 512-token window
// (_kb_window in llm_classify_benchmark.py), so no LLM sees more of the speech
// than the 110M model can. The 2000-character window is a separate robustness
// run, in benchmark_results_2000char.csv. LLM cost from actual tokens via
// OpenRouter, over all 320 speeches.
const ROWS: { name: string; acc: number; f1: number; cost: number; local?: boolean }[] = [
  { name: "GPT-5.5", acc: 0.745, f1: 0.734, cost: 3.73 },
  { name: "Claude Opus 4.8", acc: 0.717, f1: 0.711, cost: 5.62 },
  { name: "Grok 4.3", acc: 0.668, f1: 0.661, cost: 1.09 },
  { name: "Gemini 3.1 Pro", acc: 0.667, f1: 0.659, cost: 3.76 },
  { name: "DeepSeek V4 Pro", acc: 0.657, f1: 0.649, cost: 0.37 },
  { name: "Qwen 3.6 Plus", acc: 0.648, f1: 0.649, cost: 0.25 },
  { name: "KB-BERT (110M, lokal)", acc: 0.579, f1: 0.592, cost: 0, local: true },
];

const MAX_ACC = 0.8;

function pct(v: number): string {
  return v.toFixed(3);
}

export default function RiktmarkePage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 sm:py-20">
      <p className="anf-marker mb-4">
        <span className="tabular">Prot. 2026/27:5</span>
        <span className="text-line-2">·</span>
        <span>Modelljämförelse</span>
      </p>
      <h1 className="font-display text-4xl leading-tight tracking-tight sm:text-5xl">
        Behövs en egen modell när GPT finns?
      </h1>
      <p className="mt-5 max-w-2xl text-lg leading-relaxed text-ink-2">
        Rimlig fråga om vilken finjusterad modell som helst: varför träna en egen
        när man kan ringa ett API? Så jag lät min 110-miljonersmodell och sex
        frontier-LLM göra exakt samma sak, klassificera parti från riksdagstal,
        och mätte både träffsäkerhet och vad varje klassificering kostar. Min
        modell kom sist. Det är ändå rätt modell för jobbet, och siffrorna
        förklarar varför.
      </p>

      {/* ---------------------------------------------------------- SETUP */}
      <section className="mt-16">
        <ProtokollMarker n={1} name="Upplägget" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            247 anföranden med känt parti, alla hållna av personer som inte
            finns i träningen under något namn. Urvalet drogs som 320, 40 per
            parti, men 73 av dem kom från 20 personer som modellen tränats på
            under en annan namnform, och de räknas inte. Partierna har därför
            mellan 22 och 38 anföranden var. Modellen träffar rätt på{" "}
            <strong>95,9 %</strong> för talare den tränats på, men på{" "}
            <strong>57,9 %</strong> för de här. Alla sju fick exakt samma text, KB-BERT:s eget
            512-token-fönster: ingen LLM fick mer av anförandet än den lilla
            modellen fysiskt kan läsa. KB-BERT kör lokalt, de sex LLM:erna via
            API med faktisk tokenkostnad uppmätt.
          </p>
        </div>
      </section>

      {/* --------------------------------------------------------- RESULT */}
      <section className="mt-14">
        <ProtokollMarker n={2} name="Resultatet" />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-line-2">
                <th className="py-2.5 pr-4 label text-ink-3">
                  Modell
                </th>
                <th className="py-2.5 pr-4 text-right label text-ink-3">
                  Träffsäkerhet
                </th>
                <th className="py-2.5 pr-4 text-right label text-ink-3">
                  Macro-F1
                </th>
                <th className="py-2.5 text-right label text-ink-3">
                  Kostnad / 1000
                </th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => (
                <tr key={r.name} className="border-b border-line">
                  <td className="py-3 pr-4">
                    <span className={r.local ? "font-medium text-ink" : "text-ink-2"}>
                      {r.name}
                    </span>
                  </td>
                  <td className="py-3 pr-4">
                    <div className="flex items-center justify-end gap-2">
                      <span
                        className="h-1.5 rounded-full"
                        style={{
                          width: `${(r.acc / MAX_ACC) * 70}px`,
                          backgroundColor: r.local ? "var(--accent)" : "var(--line-2)",
                        }}
                        aria-hidden
                      />
                      <span className="tabular font-medium text-ink">{pct(r.acc)}</span>
                    </div>
                  </td>
                  <td className="tabular py-3 pr-4 text-right text-ink-2">{pct(r.f1)}</td>
                  <td className="tabular py-3 text-right font-medium">
                    {r.cost === 0 ? (
                      <span className="text-good">$0.00</span>
                    ) : (
                      <span className="text-ink-2">${r.cost.toFixed(2)}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-sm text-ink-3">
          Slumpbaslinje för 8 partier: 0.125. KB-BERT klarade sina 320
          klassificeringar på drygt tre minuter lokalt, utan ett enda API-anrop.
          På sitt fulla testset, 4 335 anföranden från 99 personer, ligger
          modellen på 0.612 i träffsäkerhet och 0.606 i macro-F1. Skillnaden mot
          siffrorna här är mindre än osäkerheten i dem. Två modeller lämnade
          enstaka svar som inte gick att tolka: Gemini fyra och DeepSeek två, av
          sina 247 vardera. De är uteslutna ur modellens nämnare, så de står på
          243 respektive 245 svar. Rättat 27 november 2026: tabellen byggde
          tidigare på alla 320, och KB-BERT stod då på 0.591.
        </p>
      </section>

      {/* ------------------------------------------------------ HONEST READ */}
      <section className="mt-14">
        <ProtokollMarker n={3} name="En ärlig läsning" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Frontier-modellerna vinner på träffsäkerhet, hela vägen ner. GPT-5.5
            ligger 17 procentenheter över min modell, och även de billigaste,
            Qwen och DeepSeek, ligger sju och åtta enheter över.
          </p>
          <p>
            Men träffsäkerhet är inte det enda man betalar för. Den bästa
            modellen, GPT-5.5, kostar nästan fyra dollar per tusen
            klassificeringar och kräver ett API-anrop per anförande. KB-BERT
            kostar noll och svarar på millisekunder, på samma maskin som kör
            resten av pipelinen.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------- TRADEOFF */}
      <section className="mt-14">
        <ProtokollMarker n={4} name="Vad det faktiskt betyder" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Frågan är inte bara vem som är mest träffsäker, utan vad man byter
            bort. Det handlar inte om integritet här; riksdagens protokoll är
            offentliga. Det handlar om kostnad och drift. Den här sajten läser in
            nya anföranden varje vecka och bygger om analysen på hela korpusen.
            Att skicka varje anförande till ett API i det tempot kostar pengar
            som växer med materialet, och lägger till ett externt beroende och en
            nätverksrunda för en uppgift som annars kör på en enda maskin. En
            modell som ligger några procentenheter under men kör gratis är då
            rätt val: skillnaden är inte värd priset.
          </p>
          <p>
            En sak till talar för den lilla modellen, som tabellen inte visar:
            frontier-modellerna är tränade på i princip hela webben, riksdagens
            protokoll inkluderat. En del av deras försprång kan alltså vara
            igenkänning av text de redan sett, inte generalisering. KB-BERT såg
            aldrig de här talarna.
          </p>
          <p>
            Ska du bara klassificera enstaka texter, eller väger träffsäkerheten
            tyngre än kostnaden, då är API-anropet rätt val.
          </p>
          <p className="text-sm text-ink-3">
            Systerstudien vänder på instrumentet: där är LLM:erna författare och
            KB-BERT domare.{" "}
            <a href={LLM_URL} className="text-accent underline underline-offset-2">
              Vilket parti skriver språkmodellerna som?
            </a>
          </p>
        </div>
      </section>
    </main>
  );
}
