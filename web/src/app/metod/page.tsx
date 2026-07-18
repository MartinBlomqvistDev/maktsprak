import { ProtokollMarker } from "@/components/ProtokollMarker";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Metod · Maktspråk / Protokollet",
  description:
    "Hur modellen valideras: talar-oberoende split, varför radbaserad split läcker, och vad skillnaden kostar i siffror.",
};

export default function MetodPage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16 sm:py-20">
      <p className="anf-marker mb-4">
        <span className="tabular">PROT. 2026/27:2</span>
        <span className="text-line-2">·</span>
        <span>METOD OCH VALIDERING</span>
      </p>
      <h1 className="font-display text-4xl leading-tight tracking-tight sm:text-5xl">
        Varför siffran sjönk från 0,78 till 0,62.
      </h1>
      <p className="mt-5 max-w-2xl text-lg leading-relaxed text-ink-2">
        Den högre siffran kom från en vanlig valideringsfälla. Den lägre är
        talar-oberoende: ingen politiker i testet har en enda mening i
        träningsdatan. Här är skillnaden, och vad den kostar i siffror.
      </p>

      {/* ---------------------------------------------------------- THE LEAK */}
      <section className="mt-16">
        <ProtokollMarker n={1} name="PROBLEMET" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Att gissa parti från text är lätt att överskatta. Om samma
            politiker förekommer i både tränings- och testdata lär sig
            modellen delvis att känna igen{" "}
            <em>enskilda personers</em> formuleringar, inte partiets
            retorik. En vanlig radbaserad train/test-split (90/10 på
            anföranden) gör exakt det: samma talare hamnar på båda sidor.
          </p>
          <p>
            Resultatet är en siffra som ser imponerande ut men mäter fel
            sak. Den säger egentligen: &ldquo;känner modellen igen
            Magdalena Andersson när hon talar?&rdquo;, inte &ldquo;känner
            modellen igen Socialdemokraternas sätt att argumentera?&rdquo;
          </p>
        </div>
      </section>

      {/* ------------------------------------------------------------- FIX */}
      <section className="mt-14">
        <ProtokollMarker n={2} name="LÖSNINGEN" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Nuvarande validering är <strong>talar-oberoende</strong>: alla
            unika talare listas, blandas med ett fast frö, och 15 % läggs
            åt sidan i sin helhet innan träningen ens börjar. Ingen mening
            som en testad politiker någonsin sagt förekommer i
            träningsdatan.
          </p>
          <dl className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-card border border-line bg-line sm:grid-cols-4">
            {[
              ["Träningstalare", "821"],
              ["Testtalare", "146"],
              ["Träningsrader", "31 572"],
              ["Testrader", "5 713"],
            ].map(([label, value]) => (
              <div key={label} className="bg-card px-4 py-3">
                <dt className="font-data text-[10px] uppercase tracking-widest text-ink-3">
                  {label}
                </dt>
                <dd className="tabular font-display mt-1 text-xl">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* ---------------------------------------------------------- NUMBERS */}
      <section className="mt-14">
        <ProtokollMarker n={3} name="SIFFRORNA" />
        <div className="mt-5 overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-line-2">
                <th className="py-2.5 pr-4 font-data text-[11px] uppercase tracking-widest text-ink-3">
                  Modell
                </th>
                <th className="tabular py-2.5 pr-4 text-right font-data text-[11px] uppercase tracking-widest text-ink-3">
                  Träffsäkerhet
                </th>
                <th className="tabular py-2.5 text-right font-data text-[11px] uppercase tracking-widest text-ink-3">
                  Macro-F1
                </th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-line">
                <td className="py-3 pr-4">
                  <span className="font-medium text-ink">Nuvarande</span>{" "}
                  <span className="text-ink-3">· talar-oberoende split</span>
                </td>
                <td className="tabular py-3 pr-4 text-right text-good font-medium">
                  0.628
                </td>
                <td className="tabular py-3 text-right text-good font-medium">
                  0.619
                </td>
              </tr>
              <tr className="border-b border-line">
                <td className="py-3 pr-4">
                  <span className="text-ink-3">
                    Föregående · dubbelriktad klassvikt
                  </span>
                </td>
                <td className="tabular py-3 pr-4 text-right text-ink-3">0.588</td>
                <td className="tabular py-3 text-right text-ink-3">0.595</td>
              </tr>
              <tr>
                <td className="py-3 pr-4">
                  <span className="text-ink-3">
                    Radbaserad split (läckt)
                  </span>
                </td>
                <td className="tabular py-3 pr-4 text-right text-ink-3 line-through decoration-crit">
                  0.776
                </td>
                <td className="tabular py-3 text-right text-ink-3 line-through decoration-crit">
                  0.775
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-5 text-sm text-ink-3">
          Slumpbaslinje för 8-vägsklassificering: 0.125. Random Forest på
          samma korpus utan läckage landar i samma härad; 0.619 är alltså
          en verklig, försvarbar signal, inte brus.
        </p>
        <p className="mt-3 text-sm text-ink-3">
          Mellansteget är värt en rad: den föregående modellen korrigerade
          partiobalansen <em>två gånger</em>: både genom att översampla
          ovanliga partier och genom att vikta förlustfunktionen. Aggregerat
          syntes det inte. Det syntes på korta meningar, där modellen föll
          tillbaka på de minsta partierna oavsett innehåll: »Stoppa
          invandringen!« lästes som V. Med en mekanism i stället för två läses
          samma mening som SD, med 95,9 %.
        </p>
      </section>

      {/* ----------------------------------------------------------- PARSER */}
      <section className="mt-14">
        <ProtokollMarker n={4} name="DATAN" />
        <div className="mt-5 space-y-4 text-ink-2">
          <p>
            Riksdagens protokoll är tvåspaltiga PDF:er. En naiv textextraktion
            läser rakt över kolumnerna och blandar ihop dem, och sidhuvuden
            (protokollnummer, datum) hamnar mitt i meningar. Parsern läser nu
            kolumnvis, känner igen och rensar bort återkommande sidhuvuden,
            och hanterar repliker (<code>Anf. N X (Y) replik:</code>) och
            talare utan partibeteckning korrekt, istället för att låta
            nästa anförande sluka det föregående.
          </p>
          <p>
            Hela korpusen byggs om från källan med den fixade koden:{" "}
            <strong>75 148</strong> anföranden ur 2 970 protokoll,
            2002-2026. Varje protokoll Riksdagen publicerat finns sparat
            lokalt, så korpusen är inte ett arv från tidigare körningar utan
            något som går att återskapa: ett kommando, utan nätverk, med
            samma resultat varje gång.
          </p>
          <p>
            Det är inte en akademisk poäng. Radens id var länge{" "}
            <code>protokoll_N</code> där N var en räknare över det{" "}
            <em>parsern råkade hitta</em>, inte något som stod i dokumentet.
            När parsern lagades ändrades vilka anföranden som hittades, alla
            följande index sköts ett steg, och samma id kom att peka på två
            olika tal. Id:t är nu{" "}
            <code>protokoll_talare_parti</code>: det står i dokumentet och
            kan inte glida.
          </p>
        </div>
      </section>

      <div className="mt-16 rounded-card border-l-[3px] border-accent bg-accent-soft px-6 py-5">
        <p className="text-[15px] leading-relaxed text-accent-soft-ink">
          <strong>Varför jag visar det här öppet:</strong> en modell som bara
          redovisar sin bästa siffra går inte att kontrollera. Den här sidan
          visar hur modellen valideras, var den kan gå fel och vad felen
          kostar, så att du kan bedöma den själv istället för att ta mitt ord
          för det.
        </p>
      </div>
    </main>
  );
}
