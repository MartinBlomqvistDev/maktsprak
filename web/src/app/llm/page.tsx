import type { Metadata } from "next";
import { LlmHeatmap } from "@/components/LlmHeatmap";
import { ProtokollMarker } from "@/components/ProtokollMarker";
import { NEUTRAL, SPEECH, STUDY_META } from "@/lib/llm-study";

const LLM_TITLE = "Vilket parti skriver språkmodellerna som? · Maktspråk / Protokollet";
const LLM_DESCRIPTION =
  "14 frontier-modeller fick hålla riksdagsanföranden utan att nämna parti. En KB-BERT-klassificerare läste svaren. Elva av fjorton skriver mer som Moderaterna än baslinjen, ingen mer som Vänsterpartiet.";

export const metadata: Metadata = {
  title: LLM_TITLE,
  description: LLM_DESCRIPTION,
  // Set explicitly: openGraph does not inherit a page's own title/description from
  // the root layout, so without this the shared card falls back to the site-level
  // copy and loses the finding.
  openGraph: {
    title: LLM_TITLE,
    description: LLM_DESCRIPTION,
    url: "https://maktsprak.se/llm",
    siteName: "Maktspråk",
    locale: "sv_SE",
    type: "article",
    images: [{ url: "/og.png", width: 2400, height: 1260, alt: "Maktspråk" }],
  },
  twitter: {
    card: "summary_large_image",
    title: LLM_TITLE,
    description: LLM_DESCRIPTION,
    images: ["/og.png"],
  },
};

const NORDAN_URL = "https://www.nordan.ai/research/which-swedish-party-do-llms-vote-for";
const REPO_URL = "https://github.com/MartinBlomqvistDev/maktsprak";
const DATASET_URL =
  "https://huggingface.co/datasets/MartinBlomqvist/maktsprak-llm-language-profile";

export default function LlmPage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 sm:py-20">
      <p className="anf-marker mb-4">
        <span className="tabular">Prot. 2026/27:6</span>
        <span className="text-line-2">·</span>
        <span>Språkmodellerna i talarstolen</span>
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
        <ProtokollMarker n={1} name="Upplägget" />
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
            sajten. Totalt {STUDY_META.generations} texter, ett trettiotal kronor
            i API-kostnad.
          </p>
        </div>
      </section>

      {/* ----------------------------------------------------- INSTRUMENT */}
      <section className="mt-14">
        <ProtokollMarker n={2} name="Instrumentet är inte neutralt" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            En mätning är inte trovärdigare än sitt instrument, och
            klassificeraren har egna böjelser. Tre kontroller hanterar det.
          </p>
          <p>
            <strong>Kalibrering.</strong> På {STUDY_META.calibrationN}{" "}
            partibalanserade anföranden från 84 politiker som inte finns i
            träningen under något namn träffar klassificeraren rätt parti i 50,0
            procent av fallen, mot 12,5 för slumpen. Instrumentet mäter alltså
            något verkligt på text det aldrig sett.
          </p>
          <p className="text-sm text-ink-3">
            <strong>Rättelse 27 november 2026.</strong> Här stod tidigare 59,2
            procent. Urvalet jämförde namn utan titel med listan över
            undanhållna talare. Därför kom 30 av de 480 anförandena ur själva
            träningsdatan, och 77 till från personer som modellen tränats på
            under en annan namnform. Det nya urvalet är draget på samma sätt
            bland 84 politiker som inte finns i träningen under något namn.
            Slutsatserna står sig, eftersom de bygger på avvikelsen från
            baslinjen. Med en baslinje dragen från samma personer låg
            fortfarande tolv av fjorton modeller över den på M, och ingen på V
            eller C, mätt med baslinjen från före rättelsen nedan.
          </p>
          <p>
            <strong>Baslinje.</strong> På en partibalanserad mängd verkliga
            anföranden, kortade till samma längd som modellernas tal, svarar
            klassificeraren inte jämnt: S och M får 29 procent var i snitt,
            medan L bara får 4. Den lutningen är instrumentets, inte
            modellernas. Allt nedan redovisas därför som avvikelse från just
            den baslinjen.
          </p>
          <p className="text-sm text-ink-3">
            <strong>Rättelse 27 november 2026.</strong> Baslinjen läste tidigare
            bara de första 2 000 tecknen av varje verkligt anförande, tre gånger
            så mycket text som modellernas tal. Klassificeraren ger korta texter
            mer M, så samma anföranden fick M 21 procent i stället för 29. Därför
            blev varje modells M-avvikelse åtta punkter för stor, och tolv av
            fjorton låg över baslinjen på M i stället för elva.
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
        <ProtokollMarker n={3} name="Resultatet" />
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
            Två saker står ut. <strong>Elva av fjorton modeller skriver mer som
            Moderaterna</strong> än baslinjen, från Claude Opus 4.8 (+0.34) och
            DeepSeek V4 Pro (+0.26) och nedåt. Och <strong>inte en enda av de
            fjorton skriver mer som Vänsterpartiet eller Centerpartiet</strong>
            än baslinjen: avståndet till vänstern och Centern är entydigt, hela
            kolumnerna är bruna.
          </p>
          <p>
            De största modellerna drar hårdast. Claude Opus 4.8 lägger 63
            procent av sin sannolikhetsmassa på Moderaterna, mot baslinjens 29.
            DeepSeek ligger på 55.
          </p>
          <p>
            Tre modeller bryter M-mönstret. Qwen och GLM skriver inte mer som
            Moderaterna än baslinjen; de drar i stället mot SD (GLM +0.27, Qwen
            +0.24). MiniMax drar mot MP (+0.13). Räknat över alla åtta partierna
            drar elva modeller mot M, nio mot MP, sex mot SD, fem mot KD, fyra
            mot L och två mot S. Tre av de fyra L-avvikelserna ligger under
            +0.01, alltså brus. Mot <strong>Vänsterpartiet och
            Centerpartiet drar ingen alls</strong>: de är de enda partier som
            inte en enda modell hamnar över baslinjen på.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------- CONTROL */}
      <section className="mt-14">
        <ProtokollMarker n={4} name="Kontrollen som bär resultatet" />
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
            Moderaternas språk.
          </p>
          <p>
            Den neutrala texten drar i stället mot S och SD för de flesta
            modeller. Varför vet jag inte. Klassificeraren är tränad enbart på
            debattinlägg och har ingen klass för text utan politisk riktning, så
            vad den gör med sådan text är svårt att tolka. Jag redovisar det som
            en öppen fråga om instrumentet, inte som ett fynd om modellerna. Det
            som bär beviset är att M-lutningen är borta i varenda cell.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------- CAVEATS */}
      <section className="mt-14">
        <ProtokollMarker n={5} name="Vad det inte betyder" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Att skriva som ett parti är inte att tycka som det.
            Klassificeraren mäter register: ordval, meningsbyggnad, retoriska
            vanor. Den vet ingenting om åsikter. Studien säger inte att
            modellerna är moderater; den säger att när de föreställer sig en
            svensk riksdagstalare låter föreställningen som Moderaterna.
          </p>
          <p>
            Mätningen är också liten: 24 anföranden per modell, en promptfamilj,
            en klassificerare. Rätt sätt att läsa den är som en första mätpunkt
            med öppen metod, inte som en dom. Koden, alla{" "}
            {STUDY_META.generations} texterna och alla siffror finns öppet,
            så den som vill kan göra om mätningen. Eller motbevisa den.
          </p>
        </div>
      </section>

      {/* ----------------------------------------------------- DATA / CODE */}
      <section className="mt-14">
        <ProtokollMarker n={6} name="Data och kod" />
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
          gäller hur de faktiskt skriver. Två olika axlar, och de pekar åt olika
          håll.
        </p>
      </div>
    </main>
  );
}
