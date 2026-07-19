import type { Metadata } from "next";
import { ProtokollMarker } from "@/components/ProtokollMarker";

export const metadata: Metadata = {
  title: "Behövs en egen modell? · Maktspråk / Protokollet",
  description:
    "En finjusterad 110M-modell mot sex frontier-LLM på samma uppgift: klassificera parti från riksdagstal. Träffsäkerhet, och vad varje klassificering kostar.",
};

const LLM_URL = "https://maktsprak.se/llm";

// Benchmark 2026-07-19: 320 osedda anföranden (2002-2014, utanför KB-BERTs
// träningsfönster), balanserat 40 per parti. Samma trunkerade indata (1200
// tecken) för alla. Kostnad från faktiska tokens via OpenRouter.
const ROWS: { name: string; acc: number; f1: number; cost: number; local?: boolean }[] = [
  { name: "Gemini 3.1 Pro", acc: 0.556, f1: 0.552, cost: 3.07 },
  { name: "GPT-5.5", acc: 0.541, f1: 0.536, cost: 2.06 },
  { name: "Claude Opus 4.8", acc: 0.534, f1: 0.523, cost: 2.98 },
  { name: "Qwen 3.6 Plus", acc: 0.447, f1: 0.452, cost: 0.14 },
  { name: "Grok 4.3", acc: 0.419, f1: 0.411, cost: 0.69 },
  { name: "DeepSeek V4 Pro", acc: 0.408, f1: 0.399, cost: 0.19 },
  { name: "KB-BERT (110M, lokal)", acc: 0.328, f1: 0.33, cost: 0, local: true },
];

const MAX_ACC = 0.6;

function pct(v: number): string {
  return v.toFixed(3);
}

export default function RiktmarkePage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 sm:py-20">
      <p className="anf-marker mb-4">
        <span className="tabular">PROT. 2026/27:5</span>
        <span className="text-line-2">·</span>
        <span>MODELLJÄMFÖRELSE</span>
      </p>
      <h1 className="font-display text-4xl leading-tight tracking-tight sm:text-5xl">
        Behövs en egen modell när GPT finns?
      </h1>
      <p className="mt-5 max-w-2xl text-lg leading-relaxed text-ink-2">
        Rimlig fråga om vilken finjusterad modell som helst: varför träna en egen
        när man kan ringa ett API? Så jag lät min 110-miljonersmodell och sex
        frontier-LLM göra exakt samma sak, klassificera parti från riksdagstal,
        och mätte både träffsäkerhet och vad varje klassificering kostar. Mitt
        svar blev inte det jag hoppades på, och det är just därför det är värt
        att visa.
      </p>

      {/* ---------------------------------------------------------- SETUP */}
      <section className="mt-16">
        <ProtokollMarker n={1} name="UPPLÄGGET" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            320 anföranden med känt parti, 40 per parti, alla från 2002-2014.
            Den perioden är vald med flit: den ligger <em>utanför</em>{" "}
            KB-BERTs träningsfönster (2015-2026), så testet är garanterat osett
            för den lilla modellen. Varje modell fick samma trunkerade text och
            samma fråga, svara med partiets förkortning. KB-BERT kör lokalt;
            de sex LLM:erna via API, med faktisk tokenkostnad uppmätt.
          </p>
        </div>
      </section>

      {/* --------------------------------------------------------- RESULT */}
      <section className="mt-14">
        <ProtokollMarker n={2} name="RESULTATET" />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-line-2">
                <th className="py-2.5 pr-4 font-data text-[11px] uppercase tracking-widest text-ink-3">
                  Modell
                </th>
                <th className="py-2.5 pr-4 text-right font-data text-[11px] uppercase tracking-widest text-ink-3">
                  Träffsäkerhet
                </th>
                <th className="py-2.5 pr-4 text-right font-data text-[11px] uppercase tracking-widest text-ink-3">
                  Macro-F1
                </th>
                <th className="py-2.5 text-right font-data text-[11px] uppercase tracking-widest text-ink-3">
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
          klassificeringar på 44 sekunder lokalt, utan ett enda API-anrop.
        </p>
      </section>

      {/* ------------------------------------------------------ HONEST READ */}
      <section className="mt-14">
        <ProtokollMarker n={3} name="EN ÄRLIG LÄSNING" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            De tre största frontier-modellerna vinner. Gemini, GPT-5.5 och Opus
            landar på 0.53-0.56, klart över min modells 0.33 på samma trunkerade
            indata. Att bara rapportera den raden vore dock ohederligt åt andra
            hållet, för två saker drar isär jämförelsen.
          </p>
          <p>
            <strong>Testet är KB-BERTs sämsta tänkbara.</strong> Perioden
            2002-2014 är den enda som är garanterat osedd för modellen, men
            språket där skiljer sig från det den tränats på. Ger jag den sitt
            fulla kontextfönster i stället för den hårda trunkeringen når den{" "}
            <strong>0.447</strong> på samma anföranden, i nivå med Qwen och över
            Grok och DeepSeek. Och på sin egen period, 2015-2026, ligger den på{" "}
            <strong>0.628</strong>, högre än någon av frontier-modellernas
            siffror här.
          </p>
          <p>
            <strong>LLM:erna kan ha sett riksdagstext i förträningen.</strong>{" "}
            Ingen period är garanterat osedd för dem. Deras fördel kan alltså
            delvis vara igenkänning snarare än generalisering. De två
            förbehållen pekar åt varsitt håll, och den ärliga slutsatsen är att
            frontier-skala generaliserar bättre till osett historiskt språk,
            men mindre än prislappen antyder.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------- TRADEOFF */}
      <section className="mt-14">
        <ProtokollMarker n={4} name="VAD DET FAKTISKT BETYDER" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Frågan är inte bara vem som är mest träffsäker, utan vad man byter
            bort. Den bästa modellen här, Gemini, kostar 3 dollar per tusen
            klassificeringar och kräver att varje anförande lämnar din
            infrastruktur. KB-BERT kostar noll, svarar på millisekunder, och
            texten lämnar aldrig maskinen.
          </p>
          <p>
            För den här sajten, som klassificerar färska anföranden inom modellens
            egen period och kör en hel korpus i veckan, är den lilla modellen
            både billigare och minst lika träffsäker. För pseudonymiserad eller
            känslig text är det ofta att data stannar lokalt som avgör, inte de
            sista procentenheterna. Och skulle uppgiften vara enstaka, osedd
            eller bred nog, då är API-anropet rätt val. Det är därför jag mätte,
            i stället för att gissa.
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

      <div className="mt-16 rounded-card border-l-[3px] border-accent bg-accent-soft px-6 py-5">
        <p className="text-[15px] leading-relaxed text-accent-soft-ink">
          Jag byggde den här sajten på min egen modell, och den förlorar mot GPT
          på det här testet. Att visa det ändå, med siffrorna som förklarar när
          den lilla modellen är rätt val och när den inte är det, är mer värt än
          en riggad vinst.
        </p>
      </div>
    </main>
  );
}
