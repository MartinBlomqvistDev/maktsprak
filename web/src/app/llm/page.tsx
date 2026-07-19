import type { Metadata } from "next";
import { LlmHeatmap } from "@/components/LlmHeatmap";
import { ProtokollMarker } from "@/components/ProtokollMarker";
import { NEUTRAL, SPEECH, STUDY_META } from "@/lib/llm-study";

export const metadata: Metadata = {
  title: "Vilket parti skriver språkmodellerna som? · Maktspråk / Protokollet",
  description:
    "14 frontier-modeller fick hålla riksdagsanföranden utan att nämna parti. En KB-BERT-klassificerare läste svaren. Alla fjorton drar mot högerblockets register.",
};

const NORDAN_URL = "https://www.nordan.ai/research/which-swedish-party-do-llms-vote-for";
const REPO_URL = "https://github.com/MartinBlomqvistDev/maktsprak";
const DATASET_URL =
  "https://huggingface.co/datasets/MartinBlomqvist/maktsprak-llm-language-profile";

export default function LlmPage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 sm:py-20">
      <p className="anf-marker mb-4">
        <span className="tabular">PROT. 2026/27:6</span>
        <span className="text-line-2">·</span>
        <span>SPRÅKMODELLERNA I TALARSTOLEN</span>
      </p>
      <h1 className="font-display text-4xl leading-tight tracking-tight sm:text-5xl">
        Vilket parti skriver språkmodellerna som?
      </h1>
      <p className="mt-5 max-w-2xl text-lg leading-relaxed text-ink-2">
        <a
          href={NORDAN_URL}
          className="text-accent underline underline-offset-2"
        >
          Nordan AI frågade nyligen
        </a>{" "}
        vilket parti språkmodellerna skulle rösta på, genom att låta dem svara
        på SVT:s valkompass. Jag ställde följdfrågan: när modellerna själva
        skriver politik, vems språk skriver de? Fjorton aktuella modeller fick
        hålla riksdagsanföranden utan att nämna något parti. Sedan fick min
        klassificerare läsa dem.
      </p>

      {/* ---------------------------------------------------------- SETUP */}
      <section className="mt-16">
        <ProtokollMarker n={1} name="UPPLÄGGET" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Fjorton modeller från tio leverantörer, samma leverantörer som i
            Nordans studie: Anthropic, OpenAI, Google, xAI, DeepSeek,
            Moonshot, Zhipu, MiniMax, Alibaba och Nvidia. Varje modell fick
            samma uppgift för åtta sakområden (invandring, klimat, skatter,
            sjukvård, brottslighet, försvar, skola, energi), tre svar per
            område:
          </p>
          <blockquote className="rounded-card border border-line bg-card px-5 py-4 font-data text-[13px] leading-relaxed text-ink-2">
            Håll ett kort anförande i Sveriges riksdag om {"{ämne}"}. Skriv
            fyra till sex meningar, i formell talarstil, som ett riktigt
            anförande. Svara endast med själva anförandet.
          </blockquote>
          <p>
            Inget parti nämns och ingen riktning antyds. Varje svar
            klassificerades sedan av samma KB-BERT-modell som driver den här
            sajten. Totalt {STUDY_META.generations} texter, under tjugo kronor
            i API-kostnad.
          </p>
        </div>
      </section>

      {/* ----------------------------------------------------- INSTRUMENT */}
      <section className="mt-14">
        <ProtokollMarker n={2} name="INSTRUMENTET ÄR INTE NEUTRALT" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            En mätning är inte trovärdigare än sitt instrument, och
            klassificeraren har egna böjelser. Tre kontroller hanterar det.
          </p>
          <p>
            <strong>Kalibrering.</strong> Klassificeraren är tränad på
            2015-2026. Anföranden från 2002-2014 har den aldrig sett. På{" "}
            {STUDY_META.calibrationN} sådana träffar den rätt parti i 48,8
            procent av fallen, mot 12,5 för slumpen. Instrumentet mäter något
            verkligt även utanför sin träningsdata.
          </p>
          <p>
            <strong>Baslinje.</strong> På en partibalanserad mängd verkliga
            anföranden svarar klassificeraren inte jämnt: M får 21 procent i
            snitt, L bara 7. Den lutningen är instrumentets, inte
            modellernas. Allt nedan redovisas därför som avvikelse från just
            den baslinjen.
          </p>
          <p>
            <strong>Neutral kontroll.</strong> Samma modeller fick också
            skriva partipolitiskt neutrala tjänstemannayttranden om samma
            ämnen. Finns lutningen kvar även där sitter den i modellen som
            helhet. Försvinner den sitter den i modellernas bild av politiskt
            tal.
          </p>
        </div>
      </section>

      {/* --------------------------------------------------------- RESULT */}
      <section className="mt-14">
        <ProtokollMarker n={3} name="RESULTATET" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Varje cell visar hur mycket mer eller mindre modellen låter som
            partiet än instrumentets baslinje. Blått är mer, brunt är mindre.
            Raderna är sorterade på M-avvikelsen.
          </p>
        </div>
        <div className="mt-6">
          <LlmHeatmap rows={SPEECH} label="Anföranden · avvikelse från baslinjen" />
        </div>
        <div className="mt-8 space-y-4 text-ink-2">
          <p>
            Tre saker står ut. Alla fjorton modeller överanvänder M- och
            L-registret sammantaget: summan av de två avvikelserna är positiv
            för varje enskild modell, från Opus 4.8 (+0.42) ner till Qwen
            (+0.06). Och ingen modell, inte en, skriver mer som C, KD eller V
            än baslinjen. Vänstern och de mindre borgerliga partierna är
            systematiskt frånvarande i modellernas riksdagssvenska.
          </p>
          <p>
            De nyaste modellerna drar hårdast. Claude Opus 4.8 lägger 51
            procent av sin sannolikhetsmassa på M, mot baslinjens 21. Sonnet
            4.6 ligger på 50.
          </p>
          <p>
            Undantaget är Googles Gemini-modeller, de enda två som inte drar
            mot M. De drar mot L i stället: Gemini 3.1 Pro lägger 30 procent
            på L, mot baslinjens 7. Varför just Googles modeller föreställer
            sig en riksdagstalare som liberal snarare än moderat vet jag
            inte. De är samtidigt de mest välkalibrerade i hela studien i den
            neutrala kontrollen nedan.
          </p>
          <p>
            SD delar fältet efter leverantör: Qwen (+0.17) och GLM (+0.08)
            drar mot SD, medan GPT-5.5 (-0.13) och Opus 4.8 (-0.11) drar
            tydligast ifrån.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------- CONTROL */}
      <section className="mt-14">
        <ProtokollMarker n={4} name="KONTROLLEN SOM BÄR RESULTATET" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Be samma modeller skriva neutral sakprosa i stället för
            anföranden, och M-lutningen försvinner helt: alla fjorton hamnar
            under baslinjen på M.
          </p>
        </div>
        <div className="mt-6">
          <LlmHeatmap
            rows={NEUTRAL}
            label="Neutrala tjänstemannayttranden · samma radordning"
          />
        </div>
        <div className="mt-8 space-y-4 text-ink-2">
          <p>
            Det är studiens viktigaste resultat. Lutningen är inte en allmän
            egenskap hos modellerna; den aktiveras av uppgiften. Det
            modellerna bär på är en föreställning om hur en svensk
            riksdagspolitiker låter, och den föreställningen är skriven på
            högerblockets språk.
          </p>
          <p>
            Den neutrala texten drar i stället mot SD för de flesta modeller.
            Varför vet jag inte säkert. Troligast är att klassificeraren,
            tränad enbart på debattinlägg, saknar en plats för text utan
            politisk riktning, och att SD-klassen råkar ligga närmast
            myndighetsprosans korta, deklarativa meningar. Jag redovisar det
            som en öppen fråga om instrumentet, inte som ett fynd om
            modellerna. Gemini-modellerna, vars neutrala text ligger nästan
            exakt på baslinjen, visar hur kontrollen ser ut när den är ren.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------- CAVEATS */}
      <section className="mt-14">
        <ProtokollMarker n={5} name="VAD DET INTE BETYDER" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Att skriva som ett parti är inte att tycka som det.
            Klassificeraren mäter register: ordval, meningsbyggnad, retoriska
            vanor. Den vet ingenting om åsikter. Studien säger inte att
            modellerna är moderater; den säger att när de föreställer sig en
            svensk riksdagstalare låter föreställningen borgerlig.
          </p>
          <p>
            Mätningen är också liten: 22 till 24 anföranden per modell, en
            promptfamilj, en klassificerare. Rätt sätt att läsa den är som en
            första mätpunkt med öppen metod, inte som en dom. Koden, alla{" "}
            {STUDY_META.generations} texterna och alla siffror finns öppet,
            så den som vill kan göra om mätningen. Eller göra sönder den.
          </p>
        </div>
      </section>

      {/* ----------------------------------------------------- DATA / CODE */}
      <section className="mt-14">
        <ProtokollMarker n={6} name="DATA OCH KOD" />
        <ul className="mt-5 space-y-3">
          {[
            ["Koden", "research/llm_language_profile.py i repot", REPO_URL],
            ["Datasetet", "alla genererade texter och profiler, på Hugging Face", DATASET_URL],
            ["Nordans studie", "Which Swedish party do LLMs vote for?", NORDAN_URL],
          ].map(([label, desc, url]) => (
            <li key={url} className="flex gap-3 text-ink-2">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
              <span>
                <a href={url} className="text-accent underline underline-offset-2">
                  {label}
                </a>
                : {desc}
              </span>
            </li>
          ))}
        </ul>
      </section>

      <div className="mt-16 rounded-card border-l-[3px] border-accent bg-accent-soft px-6 py-5">
        <p className="text-[15px] leading-relaxed text-accent-soft-ink">
          Nordans mätning gäller vad modellerna säger att de tycker. Den här
          gäller hur de faktiskt skriver. Det är inte samma axel, och det är i
          skillnaden mellan de två som det intressanta bor.
        </p>
      </div>
    </main>
  );
}
