import type { Metadata } from "next";
import { PartyFingerprintCard } from "@/components/PartyFingerprintCard";
import { ProtokollMarker } from "@/components/ProtokollMarker";
import { PARTY_ORDER } from "@/lib/parties";
import { distinctive } from "@/lib/site-data";

export const metadata: Metadata = {
  title: "Partierna — Maktspråk / Protokollet",
  description:
    "Varje partis retoriska fingeravtryck: de ord som är mest utmärkande för just det partiet, jämfört med de andra sju.",
};

export default function PartiernaPage() {
  return (
    <main className="relative z-[1]">
      {/* ------------------------------------------------------------ HERO */}
      <section className="border-b border-line">
        <div className="mx-auto max-w-6xl px-6 pb-14 pt-16 sm:pt-20">
          <p className="anf-marker mb-6 !text-ink-3">
            <span className="tabular">PROT. 2026/27:3</span>
            <span className="text-line-2">·</span>
            <span>RETORISKA FINGERAVTRYCK</span>
          </p>
          <h1 className="font-display max-w-3xl text-[2.2rem] leading-[1.1] tracking-tight sm:text-[3rem]">
            Varje parti har ett sätt att argumentera
          </h1>
          <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-2">
            Nedan är varje partis mest utmärkande ord: inte de vanligaste
            orden i partiets tal, utan de som skiljer partiet från de andra
            sju (viktad log-odds med en informativ Dirichlet-prior, Monroe
            m.fl. 2008). Delade politiska slitord filtreras bort, liksom
            partiernas egna namn.
          </p>
        </div>
      </section>

      {/* --------------------------------------------------------- CARDS */}
      <section>
        <div className="mx-auto max-w-6xl px-6 py-14">
          <ProtokollMarker n={1} name="FINGERAVTRYCK PER PARTI" />
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {PARTY_ORDER.map((p) => (
              <PartyFingerprintCard
                key={p.code}
                code={p.code}
                words={distinctive[p.code] ?? []}
              />
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
