import type { Metadata } from "next";
import { ProtokollMarker } from "@/components/ProtokollMarker";

export const metadata: Metadata = {
  title: "Behövs en egen modell? · Maktspråk / Protokollet",
  description:
    "Den finjusterade 110M-modellen mot sex frontier-LLM på party-klassificering, på modellens egna osedda talare. Träffsäkerhet mot kostnad, latens och om texten måste lämna maskinen.",
};

const LLM_URL = "https://maktsprak.se/llm";

// Benchmark 2026-07-20: the DEPLOYED classifier vs six frontier LLMs on 320
// speeches (2015-2026) by the 145 speakers held out of training. Split verified
// clean: 0.559 on held-out speakers vs 0.995 on training speakers. Same 2000-char
// input for everyone. LLM cost from actual tokens via OpenRouter.
const ROWS: { name: string; acc: number; f1: number; cost: number; local?: boolean }[] = [
  { name: "GPT-5.5", acc: 0.709, f1: 0.704, cost: 2.96 },
  { name: "Claude Opus 4.8", acc: 0.681, f1: 0.678, cost: 4.41 },
  { name: "Gemini 3.1 Pro", acc: 0.654, f1: 0.66, cost: 3.44 },
  { name: "DeepSeek V4 Pro", acc: 0.628, f1: 0.634, cost: 0.29 },
  { name: "Grok 4.3", acc: 0.616, f1: 0.615, cost: 0.91 },
  { name: "Qwen 3.6 Plus", acc: 0.609, f1: 0.618, cost: 0.2 },
  { name: "KB-BERT (110M, lokal)", acc: 0.559, f1: 0.563, cost: 0, local: true },
];

const MAX_ACC = 0.75;

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
        och mätte både träffsäkerhet och vad varje klassificering kostar. Min
        modell kom sist. Det är ändå rätt modell för jobbet, och siffrorna
        förklarar varför.
      </p>

      {/* ---------------------------------------------------------- SETUP */}
      <section className="mt-16">
        <ProtokollMarker n={1} name="UPPLÄGGET" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            320 anföranden med känt parti, 40 per parti, alla hållna av de 145
            talare som hölls utanför träningen. Det är modellens eget osedda
            testset, och att det verkligen är osett är kontrollerat: modellen
            träffar rätt på <strong>99,5 %</strong> för talare den tränats på,
            men <strong>55,9 %</strong> för de här. Den skillnaden är beviset på
            att det inte läcker. Varje modell fick samma text; KB-BERT kör
            lokalt, de sex LLM:erna via API med faktisk tokenkostnad uppmätt.
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
          klassificeringar på 100 sekunder lokalt, utan ett enda API-anrop.
        </p>
      </section>

      {/* ------------------------------------------------------ HONEST READ */}
      <section className="mt-14">
        <ProtokollMarker n={3} name="EN ÄRLIG LÄSNING" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Frontier-modellerna vinner på träffsäkerhet, hela vägen ner. GPT-5.5
            ligger 15 procentenheter över min modell, och även de billigaste,
            Qwen och DeepSeek, ligger några enheter över. Att bara rapportera min
            egen siffra, eller dölja att den kom sist, vore poänglöst: den kom
            sist, och det är en del av svaret.
          </p>
          <p>
            Men träffsäkerhet är inte det enda man betalar för. Den bästa
            modellen, GPT-5.5, kostar 3 dollar per tusen klassificeringar och
            kräver att varje anförande lämnar din infrastruktur. KB-BERT kostar
            noll, svarar på millisekunder, och texten lämnar aldrig maskinen.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------- TRADEOFF */}
      <section className="mt-14">
        <ProtokollMarker n={4} name="VAD DET FAKTISKT BETYDER" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Frågan är inte bara vem som är mest träffsäker, utan vad man byter
            bort. Den här sajten klassificerar en hel korpus i veckan, år efter
            år, på pseudonymiserad text. Att skicka varenda anförande till ett
            API vecka efter vecka kostar pengar som växer med korpusen, lägger
            till en nätverksrunda, och skickar ut text som inte borde lämna
            maskinen. En finjusterad modell som ligger några procentenheter under
            men kör gratis och lokalt är då rätt val, inte trots att den är
            mindre träffsäker utan för att skillnaden inte är värd priset.
          </p>
          <p>
            Skulle uppgiften i stället vara enstaka, eller kräva högsta möjliga
            träffsäkerhet, då är API-anropet rätt val. Det är därför jag mätte, i
            stället för att gissa.
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
          Jag byggde den här sajten på min egen modell, och den kom sist i det
          här testet. Att visa det ändå, med siffrorna som förklarar när den
          lilla modellen är rätt val och när den inte är det, är mer värt än en
          riggad vinst.
        </p>
      </div>
    </main>
  );
}
