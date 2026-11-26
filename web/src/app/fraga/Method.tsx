import { ProtokollMarker } from "@/components/ProtokollMarker";
import { SPEECH, STUDY_META } from "@/lib/llm-study";
import { PARTY_ORDER, partyByCode } from "@/lib/parties";
import { dec, int, interval, longDate, pct, signed, words } from "./format";
import { comparison, figures, questionSet, round } from "./method-data";

/**
 * The method section of Fråga protokollet. Every figure is read from
 * method.json, which search/pipeline/15_build_page.py writes from the reports
 * in data/reports/search/; none is typed here.
 */

const REPO = "https://github.com/MartinBlomqvistDev/maktsprak";
const PREREGISTRATION = `${REPO}/blob/main/docs/search/PREREGISTRATION.md`;
const REPORTS = `${REPO}/tree/main/data/reports/search`;
const SEARCH_CODE = `${REPO}/tree/main/search`;

const SET_LABELS: Record<string, string> = {
  natural: "Frågor i vanligt språk",
  keyword: "Samma frågor som sökord",
  interpellation: "Interpellationernas rubriker",
  filtered: "Frågor med parti eller period",
  stance: "Argument för eller emot",
};

function Section({ n, name, children }: { n: number; name: string; children: React.ReactNode }) {
  return (
    <section className="mt-14">
      <ProtokollMarker n={n} name={name} />
      <div className="mt-5 space-y-4 text-ink-2">{children}</div>
    </section>
  );
}

function Th({ children, right }: { children: React.ReactNode; right?: boolean }) {
  return (
    <th className={`py-2.5 pr-4 label text-ink-3 ${right ? "text-right" : ""}`}>{children}</th>
  );
}

function Link({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} className="text-accent underline underline-offset-2">
      {children}
    </a>
  );
}

function RetrievalTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-line-2">
            <Th>Frågor på testdelen</Th>
            <Th right>BM25</Th>
            <Th right>Vektorer</Th>
            <Th right>Hybrid</Th>
          </tr>
        </thead>
        <tbody>
          {figures.retrieval.sets.map((set) => {
            const best = Math.max(set.bm25, set.vectors, set.hybrid);
            const cell = (value: number) => (
              <td
                className={`tabular py-3 pr-4 text-right ${value === best ? "font-medium text-ink" : "text-ink-2"}`}
              >
                {dec(value, 3)}
              </td>
            );
            return (
              <tr key={set.name} className="border-b border-line">
                <td className="py-3 pr-4 text-ink-2">
                  {SET_LABELS[set.name] ?? set.name}{" "}
                  <span className="tabular text-ink-3">({int(set.questions)})</span>
                </td>
                {cell(set.bm25)}
                {cell(set.vectors)}
                {cell(set.hybrid)}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PartyTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-line-2">
            <Th>Parti</Th>
            <Th right>Frågor</Th>
            <Th right>Hybrid</Th>
            <Th right>Mot medel</Th>
          </tr>
        </thead>
        <tbody>
          {figures.retrieval.parties.map((row) => (
            <tr key={row.party} className="border-b border-line">
              <td className="py-2.5 pr-4 text-ink-2">{partyByCode(row.party)?.name ?? row.party}</td>
              <td className="tabular py-2.5 pr-4 text-right text-ink-3">{int(row.questions)}</td>
              <td className="tabular py-2.5 pr-4 text-right text-ink-2">{dec(row.recall_10, 3)}</td>
              <td className="tabular py-2.5 pr-4 text-right text-ink-2">
                {signed(row.deviation, 3)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AnswerResults() {
  const answer = figures.answer_test;
  const report = answer.report;
  const rule = answer.launch_rule;
  if (!report) {
    return (
      <>
        <p>
          Nu bedömer jag själv de {int(answer.hand_pairs)} citat som de två sätten gav{" "}
          {int(answer.hand_questions)} slumpade frågor ur testdelen, utan att se vilket sätt som
          valt vilket citat. Resultatet hamnar här när bedömningen är klar.
        </p>
        <p>
          Lanseringsregeln avgör om svarssteget körs på sidan. Det ska avstå på minst{" "}
          {pct(rule.abstains_unanswerable)} av frågorna som protokollen inte kan svara på och i
          onödan på högst {pct(rule.needless_abstention)} av de andra, och minst{" "}
          {pct(rule.quote_precision)} av citaten ska besvara frågan. Klarar det inte regeln visar
          sidan träffarna utan citatval.
        </p>
      </>
    );
  }
  const label = (name: string) => (name === "selector" ? "Claude Opus 5.5" : "Baslinjen");
  const level = pct(figures.design.interval_level);
  const accuracy = report.accuracy_difference;
  const precision = report.precision_difference;
  return (
    <>
      <p>
        Jag bedömde själv de {int(report.graded_pairs)} citat som de två sätten gav{" "}
        {int(report.drawn_questions)} slumpade frågor ur testdelen, utan att se vilket sätt som
        valt vilket citat.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-line-2">
              <Th>Valt av</Th>
              <Th right>Avstår rätt</Th>
              <Th right>Avstår i onödan</Th>
              <Th right>Citat som svarar</Th>
            </tr>
          </thead>
          <tbody>
            {report.selectors.map((row) => (
              <tr key={row.name} className="border-b border-line">
                <td className="py-3 pr-4 text-ink-2">{label(row.name)}</td>
                <td className="tabular py-3 pr-4 text-right">{pct(row.abstains_unanswerable)}</td>
                <td className="tabular py-3 pr-4 text-right">{pct(row.needless_abstention)}</td>
                <td className="tabular py-3 pr-4 text-right">
                  {row.quote_precision === null ? "inga citat" : pct(row.quote_precision)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        {precision && (
          <>
            Andelen citat som svarar på frågan skiljer sig med {signed(precision[0], 3)}, med ett
            bootstrapintervall på {level} från {interval(precision[1], precision[2], 3)}.{" "}
          </>
        )}
        Den balanserade träffsäkerheten, medelvärdet av de två andelarna avstår rätt och svarar
        när det går, skiljer sig med {signed(accuracy[0], 3)}, med ett intervall från{" "}
        {interval(accuracy[1], accuracy[2], 3)}.
      </p>
      <p>
        {report.beats_baseline
          ? "Språkmodellen slår baslinjen på båda måtten, så den väljer citaten på sidan."
          : "Språkmodellen slår inte baslinjen på båda måtten, så baslinjen väljer citaten på sidan."}{" "}
        {report.launch
          ? "Lanseringsregeln håller."
          : "Lanseringsregeln håller inte, så sidan visar träffarna utan citatval."}
      </p>
    </>
  );
}

export function Method() {
  const design = figures.design;
  const corpus = figures.corpus;
  const retrieval = figures.retrieval;
  const ops = figures.operations;
  const answer = figures.answer_test;
  const links = figures.links;
  const judge = figures.judge;
  const natural = questionSet("natural");
  const hybrid = comparison("natural", "hybrid");
  const vectors = comparison("natural", "vectors");
  const keyword = comparison("keyword", "vectors");
  const hybridKeyword = retrieval.sets.find((set) => set.name === "keyword");
  const baseline = round("baseline");
  const chosen = round(`round-${figures.answer_dev.chosen_round}-opus-5-5`);
  const local = retrieval.embedding_scores.filter((score) => score.name !== retrieval.embedding);
  const api = retrieval.embedding_scores.find((score) => score.name === retrieval.embedding);
  const partyQuestions = retrieval.parties.map((row) => row.questions);
  const leaningM = SPEECH.filter((model) => model.dev.M > 0).length;
  const level = pct(design.interval_level);
  const candidates = words(design.candidates);
  const unexplained = links.speeches - links.passed_with_folded_dashes;

  return (
    <div>
      <Section n={1} name="Hur det fungerar">
        <p>
          Frågan söks på två sätt samtidigt. Nyckelordssökning (BM25) letar efter frågans ord, och
          vektorsökning jämför vad frågan handlar om med varje passage i korpusen. De två listorna
          slås ihop, och de {candidates} bästa passagerna går vidare.
        </p>
        <p>
          Claude Opus 5.5 läser de {candidates} passagerna med numrerade stycken. Den svarar bara
          med vilka stycken den väljer, eller med att inget av dem besvarar frågan, och den får
          välja högst {words(design.max_quotes)} citat. Koden kontrollerar att varje val pekar på
          en av frågans egna passager och hämtar sedan styckena ur korpusen. Ingen annan kod skriver
          ut ett citat.
        </p>
        <p>
          Ett citat är därför alltid hela stycken ur ett och samma anförande, under det
          anförandets talare, parti och länk. Ordagrant betyder samma text som i riksdagens öppna
          data, med taggar borttagna, tecken avkodade och blanksteg ihopslagna.
        </p>
      </Section>

      <Section n={2} name="Korpusen">
        <p>
          Korpusen är riksdagens egna öppna data, där varje anförande publiceras för sig. Den
          rymmer <strong>{int(corpus.speeches)}</strong> anföranden från{" "}
          {longDate(corpus.first_date)} till {longDate(corpus.last_date)}, av ledamöter och
          statsråd från de {words(PARTY_ORDER.length)} partierna. Talmannen och ledamöter utan
          parti är inte med.
        </p>
        <p>
          Riksmötet {corpus.left_out_session} lämnade jag utanför. Riksdagen publicerar det som
          text som bryts vid den tryckta spalten, med avstavade ord och utan markerade stycken.
          Sådan text går inte att citera ordagrant, och det gäller{" "}
          {int(corpus.left_out_hard_wrapped)} anföranden.
        </p>
        <p>
          Riksdagens poster innehåller också protokollets egen text, som applåder, voteringar och
          beskedet att en debatt är avslutad. Jag lägger {int(corpus.set_aside_paragraphs)} sådana
          stycken åt sidan med {int(corpus.set_aside_rules)} namngivna regler, så att de aldrig
          citeras som någons ord.
        </p>
        <p>
          Jag stämde av korpusen mot den äldre, som klassificeraren på resten av sajten bygger på.
          Av dess {int(corpus.coverage_of)} anföranden har {int(corpus.coverage_paired)} en
          motsvarighet här, alltså {pct(corpus.coverage, 2)}. Kravet var{" "}
          {pct(corpus.coverage_threshold)}, och det satte jag innan jag räknade.
        </p>
        <p>
          Varje anförande delas i passager av hela stycken, på runt{" "}
          {int(corpus.passage_target_words)} ord. Det blir {int(corpus.passages)} passager.
        </p>
        <p>
          Länken på ett citat går till anförandets egen sida hos riksdagen. Av{" "}
          {int(corpus.links_sampled)} slumpade länkar ledde {int(corpus.links_passed)} till rätt
          anförande. Den som återstod öppnade anförandets post ur riksdagens öppna data, med texten
          men utan rubrik. En länk kan alltså öppna rådata.
        </p>
        <p>
          Länkarna i exemplen på den här sidan prövade jag på samma sätt, och {int(links.passed)} av{" "}
          {int(links.speeches)} klarade kontrollen. Riksdagens datamängder för riksmötena{" "}
          {links.dashless_from} till {links.dashless_to} skriver tankstreck som bindestreck,
          medan sidan för samma anförande har kvar tankstrecket. Räknas de två tecknen som lika
          klarar {int(links.passed_with_folded_dashes)} av {int(links.speeches)} kontrollen
          {unexplained === links.xml_records && unexplained > 0
            ? unexplained === 1
              ? ", och den som återstod öppnade posten ur öppna data"
              : `, och de ${words(unexplained)} som återstod öppnade posten ur öppna data`
            : ""}
          . Ett citat ur de åren kan alltså skilja sig från riksdagens sida i just det tecknet.
        </p>
      </Section>

      <Section n={3} name="Bestämt innan jag mätte">
        <p>
          Frågeseten, måtten, jämförelserna och reglerna för vad som skulle köras på sidan skrev jag
          ner innan något var mätt. <Link href={PREREGISTRATION}>Förregistreringen</Link> ligger i
          repot, och git visar i vilken ordning saker hände. Det som ändrades senare står i
          daterade tillägg, med skälen.
        </p>
        <p>
          Frågorna delades i en utvecklingsdel, där allt trimmades, och en testdel som öppnades
          först efter det. Claude Opus 5.5 skrev {int(natural.questions)} frågor i vanligt språk
          utifrån slumpade passager, och passagens ovanliga ord fick inte återkomma i frågan. Samma
          passager fick också frågor i form av några sökord. Jag granskade{" "}
          {int(figures.questions.review_drawn)} slumpade frågor och godkände{" "}
          {int(figures.questions.review_approved)}.
        </p>
      </Section>

      <Section n={4} name="Sökningen">
        <p>
          Jag ville veta om vektorsökning slår nyckelordssökning på svenskt riksdagsspråk.
          Svenskan sätter ihop ord, så den som frågar om att avveckla kärnkraften hittar inte ordet
          kärnkraftsavvecklingen med nyckelord, inte ens när orden kortas till sina stammar. Senast
          jag mätte samma sak, på en engelsk korpus, vann ren nyckelordssökning.
        </p>
        {api && local.length > 0 && (
          <p>
            Jag prövade {words(retrieval.embedding_scores.length)} embeddingmodeller på
            utvecklingsdelen, mot en delmängd av korpusen med {int(retrieval.subset_passages)}{" "}
            passager. Qwen3-Embedding-8B hittade rätt anförande bland de {candidates} bästa för{" "}
            {dec(api.recall_10, 3)} av frågorna i vanligt språk. De {words(local.length)} lokala
            modellerna låg mellan {dec(Math.min(...local.map((s) => s.recall_10)), 3)} och{" "}
            {dec(Math.max(...local.map((s) => s.recall_10)), 3)}, så jag valde Qwen3.
          </p>
        )}
        <p>
          Tabellen visar resultatet på testdelen: andelen frågor där rätt anförande finns bland de{" "}
          {candidates} bästa träffarna. Hybriden är BM25 och vektorerna ihopslagna, och det är den
          som körs på sidan.
        </p>
        <RetrievalTable />
        <p>
          På frågor i vanligt språk hittar hybriden rätt anförande för{" "}
          <strong>{dec(hybrid.system_mean, 3)}</strong> av frågorna och BM25 för{" "}
          <strong>{dec(hybrid.against_mean, 3)}</strong>. Skillnaden är {dec(hybrid.difference, 3)},
          med ett bootstrapintervall på {level} från {interval(hybrid.low, hybrid.high, 3)}. Regeln
          krävde minst {dec(retrieval.min_difference, 2)} och ett intervall helt över noll, så
          hybriden körs på sidan.
        </p>
        <p>
          Vektorerna ensamma slår BM25 ännu tydligare på samma frågor:{" "}
          {dec(vectors.system_mean, 3)} mot {dec(vectors.against_mean, 3)}, med ett intervall från{" "}
          {interval(vectors.low, vectors.high, 3)}. Ställda som sökord vänder det. Där når BM25{" "}
          {dec(keyword.against_mean, 3)} och vektorerna {dec(keyword.system_mean, 3)}, och
          intervallet för skillnaden går från {interval(keyword.low, keyword.high, 3)}. Båda
          intervallen ligger helt på ena sidan om noll, så båda resultaten håller.
        </p>
        {hybridKeyword && (
          <p>
            Hybriden tappar lite mot vektorerna på vanligt språk, {dec(hybrid.system_mean, 3)} mot{" "}
            {dec(vectors.system_mean, 3)}, och nästan inget mot BM25 på sökord,{" "}
            {dec(hybridKeyword.hybrid, 3)} mot {dec(hybridKeyword.bm25, 3)}. Förregistreringen
            prövade bara hybriden mot BM25 för sidan.
          </p>
        )}
        <p>
          Måttet är en undre gräns. Varje fråga i vanligt språk har ett enda rätt anförande, och
          andra anföranden kan svara på den också.
        </p>
        <p>
          Förregistreringen säger att ett parti ska nämnas här om andelen för frågor ur partiets
          anföranden avviker mer än {dec(retrieval.party_margin, 2)} från medelvärdet, som är{" "}
          {dec(retrieval.natural_mean, 3)}. Det gör alla {words(retrieval.parties.length)}. Med
          mellan {int(Math.min(...partyQuestions))} och {int(Math.max(...partyQuestions))} frågor
          per parti är osäkerheten stor, och uppdelningen avgör ingenting.
        </p>
        <PartyTable />
      </Section>

      <Section n={5} name="Citaten">
        <p>
          Jag mätte två sätt att välja citat mot varandra. Baslinjen använder ingen språkmodell.
          Den citerar den bästa träffens stycken med frågans ord och avstår när träffens poäng
          ligger under en tröskel. Det andra sättet är Claude Opus 5.5.
        </p>
        <p>
          Prompten fick {words(design.prompt_rounds)} omgångar på utvecklingsdelen, med BM25:s{" "}
          {candidates} träffar per fråga, och den sista valdes. Den avstod på{" "}
          {pct(chosen.abstains_unanswerable)} av frågorna som protokollen inte kan svara på och i
          onödan på {pct(chosen.needless_abstention)} av de andra, mot baslinjens{" "}
          {pct(baseline.abstains_unanswerable)} och {pct(baseline.needless_abstention)}.
        </p>
        <p>
          Citatens kvalitet skulle bedömas av en språkmodell, men bara om den först stämde med mina
          egna bedömningar av {int(judge.pairs)} par av fråga och text. Den gav samma betyg som jag
          i {int(Math.round(judge.exact * judge.pairs))} av dem, och kappa blev{" "}
          <strong>{dec(judge.kappa, 3)}</strong> mot kravet {dec(judge.threshold, 1)}, så bara mina
          egna bedömningar räknas. I {int(judge.stricter)} av de {int(judge.disagreements)} par där
          vi skilde oss åt var den strängare än jag.
        </p>
        <p>
          Svarssteget kördes först på testdelen med BM25:s träffar, innan sökningen var avgjord.
          Jag trodde att API-krediten gick ut tidigare än den gjorde. Sedan valde sökningen
          hybriden, och då skulle sidan ha valt citat ur andra träffar än dem den visar. Jag lade
          därför den första körningen åt sidan oläst, skrev ett tillägg till förregistreringen och
          körde testdelen igen på hybridens träffar, med samma prompt. De två körningarna kostade{" "}
          {dec(answer.first_cost_usd, 2)} och {dec(answer.hybrid_cost_usd, 2)} dollar. Testdelen
          körs därmed en gång fler än förregistreringen från början tillät. Den första körningen
          avgör ingenting och öppnas först när den andra är bedömd.
        </p>
        <AnswerResults />
      </Section>

      <Section n={6} name="Åtta partier">
        <p>
          I vyn med alla {words(PARTY_ORDER.length)} partier ställs frågan en gång per parti, bara
          bland partiets egna anföranden. I exemplen har Claude Opus 5.5 valt citaten för varje
          parti. När du ställer en egen fråga väljer baslinjen i den vyn. Med Opus vore det ett
          anrop per parti, och månadens budget skulle räcka till ungefär{" "}
          {words(ops.party_views_per_month)} sådana frågor.
        </p>
      </Section>

      <Section n={7} name="Driften">
        <p>
          Sökningen körs i en egen tjänst på Google Cloud Run med indexet inbakat, och tjänsten
          stängs av när ingen använder den. Jag mätte den i en container med{" "}
          {words(design.service_vcpus)} processor och {dec(ops.memory_budget_gib, 0)} GiB minne. Den
          svarade på {int(ops.search_p95_ms)} ms vid 95:e percentilen, startade på{" "}
          {dec(ops.cold_start_s, 1)} s som längst och använde som mest{" "}
          {dec(ops.memory_peak_gib, 1)} GiB. Gränserna var {int(ops.search_p95_budget_ms)} ms och{" "}
          {dec(ops.cold_start_budget_s, 0)} s.
        </p>
        <p>
          Jag mätte också genom sajten, från att frågan skickas. I{" "}
          {int(ops.proxy_search_requests)} sökningar kom träfflistan efter{" "}
          {dec(ops.proxy_search_p50_s, 1)} s i median och {dec(ops.proxy_search_p95_s, 1)} s vid
          95:e percentilen. För {words(ops.proxy_answer_requests)} frågor med svarssteget kom
          citaten efter {dec(ops.proxy_answer_p50_s, 1)} s i median. När tjänsten hade stått stilla
          fick den första frågan vänta {dec(ops.proxy_cold_first_line_s, 1)} s på träfflistan
          medan den startade.
        </p>
        <p>
          Frågans vektor räknas ut av Qwen3-Embedding-8B, som jag når via OpenRouter och som körs
          hos Nebius. Sökningen hänger alltså på två tjänster utanför sajten. När jag mätte{" "}
          {int(ops.vector_questions)} frågor väntade en fråga i median{" "}
          {dec(ops.vector_wait_p50_s, 1)} s på sin vektor, {dec(ops.vector_wait_p95_s, 1)} s vid
          95:e percentilen och som längst {dec(ops.vector_wait_max_s, 1)} s. Tjänsten väntar i upp
          till {int(ops.vector_limit_s)} s och söker sedan med BM25 ensamt, och sidan säger till
          när det händer. Gränsen satte jag efter att ha sett väntetiderna och innan testdelen
          öppnades. Med en gräns på {int(ops.vector_wait_short_s)} s hade{" "}
          {pct(ops.vector_wait_over_short, 1)} av frågorna fått BM25 ensamt.
        </p>
        <p>
          Ett svar med Opus 5.5 kostar ungefär {dec(ops.cost_per_question_usd, 2)} dollar. Sidans
          budget är {dec(ops.monthly_budget_usd, 0)} dollar i månaden, så den räcker till ungefär{" "}
          {int(ops.questions_per_month)} frågor. Resten av månaden väljer baslinjen citaten, och
          sidan säger det.
        </p>
        <p>
          En schemalagd körning väcker sökningen med jämna mellanrum, utan svarssteget. En annan
          ställer en fast fråga varje dag och larmar om det väntade anförandet inte kommer tillbaka.
        </p>
      </Section>

      <Section n={8} name="Vad sidan inte gör">
        <p>
          Sidan sammanfattar aldrig vad ett parti tycker. Skälet kommer ur sajtens egen studie, där{" "}
          {leaningM} av {STUDY_META.models} språkmodeller skrev mer som Moderaterna än baslinjen
          gör. <Link href="/llm">Vilket parti skriver språkmodellerna som?</Link>
        </p>
        <p>
          Ett citat är några stycken ur ett längre anförande, ibland ur en replik. Läs hela
          anförandet via länken innan du använder ett citat.
        </p>
      </Section>

      <Section n={9} name="Data och kod">
        <ul className="space-y-3">
          {[
            ["Koden", "sökningen, svarssteget och mätningen i search/", SEARCH_CODE],
            ["Förregistreringen", "med alla tillägg", PREREGISTRATION],
            ["Rapporterna", "varje tal på den här sidan, som JSON", REPORTS],
          ].map(([label, desc, url]) => (
            <li key={url} className="flex gap-3 text-ink-2">
              <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
              <span>
                <Link href={url}>{label}</Link>: {desc}
              </span>
            </li>
          ))}
        </ul>
      </Section>
    </div>
  );
}
