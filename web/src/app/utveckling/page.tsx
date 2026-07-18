import type { Metadata } from "next";
import { DivergenceChart } from "@/components/DivergenceChart";
import { PartyFrameChart } from "@/components/PartyFrameChart";
import { ProtokollMarker } from "@/components/ProtokollMarker";
import { TrajectoryGrid } from "@/components/TrajectoryGrid";
import { YearSlider } from "@/components/YearSlider";
import { meta } from "@/lib/site-data";

export const metadata: Metadata = {
  title: "Utveckling · Maktspråk / Protokollet",
  description:
    "Riksdagens språk från 2002 till idag: vad debatten rörde sig mot och ifrån, och hur partiernas retorik har förändrats.",
};

const RISING_TERMS = ["ukraina", "pandemin", "inflationen", "gaza", "kärnkraft"];
const FALLING_TERMS = ["alliansen", "landsting", "jobb", "arbetsmarknaden", "rut"];

const RISING_NOTES: Record<string, string> = {
  ukraina: "Rysslands invasion, 2022",
  pandemin: "Covid-19, 2020–21",
  inflationen: "Prischocken, 2023",
  gaza: "Kriget bryter ut, 2023",
  kärnkraft: "Energidebatten återvänder",
};
const FALLING_NOTES: Record<string, string> = {
  alliansen: "Alliansen upplöses, 2019",
  landsting: "Blev regioner, 2019",
  jobb: "Alliansens paradord tappar mark",
  arbetsmarknaden: "Mindre centralt i debatten",
  rut: "RUT-avdraget lämnar debatten",
};

export default function UtvecklingPage() {
  return (
    <main className="relative z-[1]">
      {/* ------------------------------------------------------------ HERO */}
      <section className="border-b border-line">
        <div className="mx-auto max-w-6xl px-6 pb-14 pt-16 sm:pt-20">
          <p className="anf-marker mb-6 !text-ink-3">
            <span className="tabular">PROT. 2026/27:4</span>
            <span className="text-line-2">·</span>
            <span>SPRÅKET ÖVER TID</span>
          </p>
          <h1 className="font-display max-w-3xl text-[2.2rem] leading-[1.1] tracking-tight sm:text-[3rem]">
            Samma kammare, ett nytt språk vartannat år
          </h1>
          <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-2">
            Jag har låtit en språkmodell analysera samtliga anföranden i riksdagen
            sedan {meta.first_year}. Skiftena är stora: ett år präglas debatten av
            flyktingkrisen, nästa av pandemin, därefter av kriget i Ukraina. Samma
            kammare, men språket byts ut.
          </p>
        </div>
      </section>

      {/* --------------------------------------------------------- SLIDER */}
      <section className="border-b border-line bg-paper-2">
        <div className="mx-auto max-w-6xl px-6 py-14">
          <ProtokollMarker n={1} name="ETT ÅR I TAGET" />
          <p className="mt-4 mb-8 max-w-2xl text-ink-2">
            Dra i reglaget, eller tryck play. För varje år: de ord som är mest
            utmärkande jämfört med alla andra år (viktad log-odds, Monroe m.fl.
            2008). Politikernamn och procedurord är till stor del bortfiltrerade.
          </p>
          <YearSlider />
        </div>
      </section>

      {/* ---------------------------------------------------- TRAJECTORIES */}
      <section className="border-b border-line">
        <div className="mx-auto max-w-6xl px-6 py-14">
          <ProtokollMarker n={2} name="FÖLJ ETT ORD GENOM ÅREN" />
          <p className="mt-4 mb-8 max-w-2xl text-ink-2">
            Reglaget ovan visar vad som präglade varje år. Här är den andra
            rörelsen: enskilda ord över hela perioden. Vissa slår igenom och
            stannar, andra tonar bort. Andelen räknas per 10 000 ord, så att år
            med mer debatt inte väger tyngre.
          </p>

          <p className="font-data mb-3 flex items-center gap-2 text-[11px] uppercase tracking-widest text-ink-3">
            <span className="h-[3px] w-6 rounded-full bg-accent" /> På väg upp
          </p>
          <TrajectoryGrid terms={RISING_TERMS} direction="up" notes={RISING_NOTES} />

          <p className="font-data mb-3 mt-8 flex items-center gap-2 text-[11px] uppercase tracking-widest text-ink-3">
            <span className="h-[3px] w-6 rounded-full bg-crit" /> På väg ut
          </p>
          <TrajectoryGrid terms={FALLING_TERMS} direction="down" notes={FALLING_NOTES} />
        </div>
      </section>

      {/* ------------------------------------------------- PARTY MOVEMENT */}
      <section className="border-b border-line bg-paper-2">
        <div className="mx-auto max-w-6xl px-6 py-14">
          <ProtokollMarker n={3} name="PARTIERNAS RÖRELSE" />
          <p className="mt-4 mb-8 max-w-2xl text-ink-2">
            Vem äger en fråga, och har det förändrats? Här är varje partis andel
            av sitt eget tal som handlar om en given fråga, år för år. Välj en
            fråga och se linjerna röra sig, mot varandra eller ifrån varandra.
            Krympande avstånd mellan två partiers linjer är ett mätbart tecken på
            att de närmar sig varandra retoriskt, oavsett vad de säger om saken.
          </p>
          <PartyFrameChart />
        </div>
      </section>

      {/* ----------------------------------------------------- DIVERGENCE */}
      <section>
        <div className="mx-auto max-w-6xl px-6 py-14">
          <ProtokollMarker n={4} name="TALAR PARTIERNA MER LIKA?" />
          <div className="mt-4 grid gap-8 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <DivergenceChart />
            <p className="text-ink-2">
              Ett mått, inte en värdering: hur olika partierna talar från år
              till år, mätt som skillnaden mellan deras ordförråd
              (Jensen–Shannon-divergens). Högre värde betyder att de talar om
              mer olika saker. {meta.first_year} var skillnaden som störst;
              sedan dess har ordförrådet legat förhållandevis stabilt. Ingen
              dramatisk konvergens och ingen tydlig splittring; partierna
              talar i hög grad om samma kriser, om än med olika ord och,
              som föregående avsnitt visar, med olika tyngdpunkt.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
