import { ProtokollMarker } from "@/components/ProtokollMarker";
import { PartySpectrum } from "@/components/PartySpectrum";
import { PredictDemo } from "@/components/PredictDemo";
import { RedactReveal } from "@/components/RedactReveal";
import { figures } from "@/app/fraga/method-data";

export default function Home() {
  // The corpus's size in speeches, as the search's reconciliation counted them
  // in data/reports/search/reconciliation.json. The Parquet archive holds one
  // record per speaker and protocol, so its row count, meta.json's count, is a
  // count of records and not of speeches. The count used to be fetched from
  // Supabase, which is only a trimmed ETL landing zone of recent years.
  const speechCount = figures.corpus.legacy_speeches.toLocaleString("sv-SE");

  return (
    <main className="relative z-[1]">
      {/* ---------------------------------------------------------------- HERO */}
      <section className="border-b border-line">
        <div className="mx-auto max-w-6xl px-6 pb-20 pt-16 sm:pt-24">
          <p className="anf-marker mb-6 !text-ink-3">
            <span className="tabular">Prot. 2026/27:1</span>
            <span className="text-line-2">·</span>
            <span>Språkanalys av riksdagens protokoll</span>
          </p>

          <h1 className="font-display max-w-4xl text-[2.4rem] leading-[1.08] tracking-tight sm:text-[3.4rem]">
            <RedactReveal
              segments={[
                { text: "Hur något sägs avslöjar " },
                { text: "partiet", redact: true },
                { text: "." },
              ]}
            />
          </h1>

          <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-2">
            Jag har finjusterat en språkmodell på riksdagens debatter tills den
            lärt sig partiernas retoriska fingeravtryck: hur en politiker
            formulerar sig. Klistra in valfri text nedan, utan källa, så
            ser du vilket parti modellen tror att den kommer från.
          </p>

          <div className="mt-10 max-w-xl">
            <PredictDemo />
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------- SPECTRUM / STATS */}
      <section className="border-b border-line bg-paper-2">
        <div className="mx-auto max-w-6xl px-6 py-10">
          <div className="grid gap-10 sm:grid-cols-[1fr_auto] sm:items-end">
            <div>
              <p className="label mb-3 text-ink-3">
                Åtta partier, ett protokoll
              </p>
              <PartySpectrum />
            </div>
            <dl className="flex gap-8">
              <div>
                <dt className="label text-ink-3">
                  Anföranden
                </dt>
                <dd className="tabular font-display text-2xl">{speechCount}</dd>
              </div>
              <div>
                <dt className="label text-ink-3">
                  Macro-F1, ohörda talare
                </dt>
                <dd className="tabular font-display text-2xl">0.619</dd>
              </div>
            </dl>
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------------- METHOD */}
      <section className="border-b border-line">
        <div className="mx-auto max-w-6xl px-6 py-20">
          <div className="max-w-2xl">
            <ProtokollMarker n={1} name="Extraherar" />
            <p className="mt-4 mb-10 text-ink-2">
              Varje vecka hämtas nya protokoll från Riksdagens öppna API.
              Tvåspaltiga PDF:er läses kolumnvis, sidhuvuden rensas bort och
              varje anförande knyts till rätt talare, även repliker och
              talare utan partibeteckning.
            </p>

            <ProtokollMarker n={2} name="Tränades" />
            <p className="mt-4 mb-10 text-ink-2">
              Modellen finjusterades en gång, på anföranden från 2015 och framåt.
              Korpusen växer vecka för vecka med nya protokoll, medan modellen är
              densamma. Valideringen är <strong>talar-oberoende</strong>: 15 % av
              politikerna hölls helt utanför träningen, så siffran mäter om
              modellen fångar ett partis sätt att argumentera hos talare den
              aldrig har sett.
            </p>

            <ProtokollMarker n={3} name="Avslöjar" />
            <p className="mt-4 text-ink-2">
              Resultatet är ett verktyg som läser politisk text och visar
              vilket parti den mest liknar retoriskt. Till det kommer en öppen
              redovisning av var modellen är säker och var den gissar.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
