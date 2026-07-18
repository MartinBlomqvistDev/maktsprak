import { ProtokollMarker } from "@/components/ProtokollMarker";
import { PartySpectrum } from "@/components/PartySpectrum";
import { PredictDemo } from "@/components/PredictDemo";
import { RedactReveal } from "@/components/RedactReveal";
import { meta } from "@/lib/site-data";

export default function Home() {
  // The corpus count comes from meta.json, which build_site_data.py writes from
  // the Parquet archive. It used to be fetched from Supabase, which is the wrong
  // source and was quietly giving the wrong answer: Supabase is a trimmed ETL
  // landing zone holding only recent years (~37k rows), not the 2002-2026 corpus.
  // It also meant a database round-trip, and egress, to render the front page.
  const speechCount = meta.count.toLocaleString("sv-SE");

  return (
    <main className="relative z-[1]">
      {/* ---------------------------------------------------------------- HERO */}
      <section className="border-b border-line">
        <div className="mx-auto max-w-6xl px-6 pb-20 pt-16 sm:pt-24">
          <p className="anf-marker mb-6 !text-ink-3">
            <span className="tabular">PROT. 2026/27:1</span>
            <span className="text-line-2">·</span>
            <span>SPRÅKANALYS AV RIKSDAGENS PROTOKOLL</span>
          </p>

          <h1 className="font-display max-w-4xl text-[2.4rem] leading-[1.08] tracking-tight sm:text-[3.4rem]">
            <RedactReveal
              segments={[
                { text: "Hur något sägs avslöjar " },
                { text: "partiet", redact: true },
                { text: ", inte bara " },
                { text: "vad som sägs", redact: true },
                { text: "." },
              ]}
            />
          </h1>

          <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-2">
            Jag har låtit en finjusterad språkmodell läsa {speechCount}{" "}
            riksdagsanföranden tills den lärt sig partiernas retoriska
            fingeravtryck: inte vad en politiker talar om, utan hur. Klistra in
            valfri text nedan, utan att ange källa, och se vilket parti modellen
            tror att den kommer från.
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
              <p className="font-data mb-3 text-[11px] uppercase tracking-widest text-ink-3">
                Åtta partier, ett protokoll
              </p>
              <PartySpectrum />
            </div>
            <dl className="flex gap-8">
              <div>
                <dt className="font-data text-[10px] uppercase tracking-widest text-ink-3">
                  Anföranden
                </dt>
                <dd className="tabular font-display text-2xl">{speechCount}</dd>
              </div>
              <div>
                <dt className="font-data text-[10px] uppercase tracking-widest text-ink-3">
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
            <ProtokollMarker n={1} name="EXTRAHERAR" />
            <p className="mt-4 mb-10 text-ink-2">
              Varje vecka hämtas nya protokoll från Riksdagens öppna API.
              Tvåspaltiga PDF:er läses kolumnvis, sidhuvuden rensas bort och
              varje anförande knyts till rätt talare, även repliker och
              talare utan partibeteckning.
            </p>

            <ProtokollMarker n={2} name="TRÄNAR" />
            <p className="mt-4 mb-10 text-ink-2">
              En KB-BERT-modell finjusteras på den rensade korpusen.
              Valideringen är <strong>talar-oberoende</strong>: 15 % av
              politikerna hålls helt utanför träningen, så resultatet mäter
              generalisering till personer modellen aldrig sett, inte
              igenkänning av bekanta röster.
            </p>

            <ProtokollMarker n={3} name="AVSLÖJAR" />
            <p className="mt-4 text-ink-2">
              Resultatet är ett verktyg som läser politisk text och visar
              vilket parti den mest liknar retoriskt, och en öppen
              redovisning av var modellen är säker och var den gissar.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
