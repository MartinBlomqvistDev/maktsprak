# Pre-registration: search over the Riksdag's speeches

Written 2026-11-20, before any question set, index or evaluation result exists.
The commit that adds this file fixes that order.

Later decisions go in dated addenda at the end. The text above them is not
edited once it is committed.

## What is being built

A page on maktsprak.se, working name *Fråga protokollet*, where a question in
Swedish is answered with verbatim quotations from the Riksdag's chamber debates.
Each quotation carries the speaker, the party, the date and a link to that
speech at riksdagen.se. When the record does not answer the question, the page
says so.

**Scope.** The eight parties of the existing corpus, from the 2002/03 session
to the latest one. The Speaker and members without a party are outside.

**Corpus.** The Riksdag's open data, where every speech (anförande) is
published on its own. Before anything here is measured, that data is
reconciled against the existing corpus, which is parsed from the protocol PDFs.
If the datasets cover less than 99% of the speeches, or the texts differ
systematically, the work stops, and what happens next is recorded in an
addendum.

**Units.** A speech is one numbered contribution to a debate, as the Riksdag
publishes it. A passage is a run of whole paragraphs from one speech, and
passages are what the indexes hold. The unit of quotation is the paragraph.
Hybrid retrieval means the BM25 ranking and the vector ranking merged with
reciprocal rank fusion, with any filter applied to both.

Four rules shape what is measured:

- **Verbatim by construction.** The language model returns passage ids and
  paragraph numbers and nothing else. Code fetches the text from the index, so
  the model cannot change a word.
- **No characterisation.** Nothing generates a summary of what a party thinks.
  [This project's own study](https://maktsprak.se/llm) found that 12 of 14
  frontier models write more like Moderaterna than the baseline does.
- **Baseline first.** BM25 is measured before vectors, and a selector without a
  language model before one with.
- **Abstain rather than guess.** Abstention is measured as a metric of its own.

Language models do three jobs here: they write the generated question sets,
they judge relevance once that judgement has been checked against mine, and
they select paragraphs in the answer step. The model's id, the hash of its
prompt and the date are stored with everything a model produces.

## The question

Does semantic search beat keyword search in Swedish parliamentary language, and
for which kinds of question?

I expect hybrid retrieval to win on questions in natural language. Swedish
builds compounds: a question about "avveckling av kärnkraften" does not match
"kärnkraftsavvecklingen" lexically, even after stemming. The last time I
measured this, on an English corpus with questions built from a citation graph,
plain BM25 won. So both outcomes are written out in advance, further down.

## Question sets

| Set | Size | How it is built | Gold |
|---|---|---|---|
| Natural questions | 300 | A language model writes a question that a voter could ask and that a sampled passage answers. Rare words from the passage may not be reused, which code checks after stemming | The passage's speech |
| Keyword questions | 300 | The same passages, phrased as two to four search terms | The same |
| Interpellations | Every one in a sample of sessions | The title of the interpellation is the question. No language model | The speeches in that debate |
| Filtered | 100 | Natural questions with a party or a period added | The passage, under the filter |
| Unanswerable | 150 | Off topic; a false premise (a speaker who never sat in the Riksdag, Sverigedemokraterna before 2010); events after the index ends | Abstain |
| Stance | 50 | "Which arguments have been made against X?", where the passage states its stance outright | The passage's speech |

- **Sampling.** Passages are drawn stratified on year, party and length, with a
  fixed seed.
- **Split.** 40% development and 60% test, split by passage and stratified on
  year and party. Everything is tuned on the development split. The test split
  is run three times and no more: once for retrieval, once for the answer step
  and once for the reranker.
- **Provenance.** The question sets are tracked in git. An addendum records
  their hashes, and the sessions sampled for the interpellation set, before any
  system is scored on them.
- **Review.** I review 50 randomly drawn generated questions: is the question
  reasonable, and does the passage answer it? Below 85% approved, the questions
  are generated again. There are two rounds at most. If the second also fails,
  the interpellation set becomes the primary measure and the addendum says so.
- **Known weakness.** A natural question has one gold speech, but other speeches
  may answer it too, so recall is a lower bound. Two things offset that: pooled
  judgement, and the interpellation set, where a question has several gold
  speeches.

## Metrics

**Retrieval**

- recall@5, recall@10, recall@50 and MRR@10, at speech level: a retrieved
  passage counts as its speech, and a speech counts once, at its best rank.
  Where a question has several gold speeches, recall is the share of them found.
- nDCG@10 on a pooled subset: 100 questions from the test split, the top five
  results from each system, graded 0, 1 or 2. A system that is added later
  puts its top five in the pool before it is scored.
- Each is broken down by question set, by period (2002-2009, 2010-2017,
  2018-2026) and by party.

**Answers**

- Quote precision: the share of selected quotations that answer the question.
- Abstention: the share of unanswerable questions where the system abstains,
  and the share of answerable ones where it abstains needlessly.
- Selection errors: selections that the assembler rejects.
- Party balance: the party distribution among the selected quotations, against
  the distribution among the candidates the selector was given. It is the
  question the LLM study asked, put to selection instead of text.

**Operations**

- Response time at p50 and p95, cold and warm, measured from outside through
  the site's proxy. The whole chain counts, and the clock starts when the
  question is submitted.
- Cost per question, cold start and memory.

**The judge.** Quote precision and pooled relevance are judged by a language
model. Its verdicts are used only if they agree with mine on 100
question-passage pairs, with Cohen's kappa at least 0.6. Otherwise only manual
judgements are used.

**Statistics.** Paired bootstrap over questions, 10 000 resamples, 95%
percentile intervals. Four comparisons are pre-registered and no others are
tested. The intervals are not adjusted for their number, which is one reason
the list is short.

## The comparisons

The four comparisons are read from the three runs on the test split.

Retrieval, in one run on the full corpus:

1. Hybrid against BM25, on recall@10 for the natural questions. This decides
   what retrieval runs in production.
2. Vectors alone against BM25, on recall@10, for the natural questions and for
   the keyword questions, which are the same passages asked two ways. This
   answers the question above and decides nothing in production. A win is
   stated only where the interval of the difference excludes 0.

The answer step, in one run:

3. Selection by a language model against the baseline selector, on quote
   precision and abstention. This decides which selector runs.

After launch, in one run:

4. A reranker against none, on nDCG@10. This decides whether one is added.

Everything else that is reported, such as the breakdowns by question set,
period and party, is description and decides nothing.

## Decision rules

| Decision | Rule |
|---|---|
| Hybrid in production | recall@10 on the natural questions, test split: hybrid minus BM25 is at least 0.03 and the lower bound of the interval is above 0. Otherwise BM25 runs alone (outcome B). The decision is made on the full corpus only |
| A fair chance for BM25 | Stemming, stop words, `k1` and `b` are tuned on the development split before the comparison |
| Embedding model | At most four candidates: a small multilingual local model, a Swedish local model, a large multilingual local model and an API model. They are compared on a subset: every gold speech plus a stratified sample, up to about 60 000 passages. The best recall@10 on the natural questions of the development split wins. If several are within 0.01, the one that is lightest to operate is chosen. An API model can win only if embedding the whole corpus fits a budget of 10 USD |
| Which candidates | The list is locked in an addendum on the day the vector work begins, after checking the Scandinavian Embedding Benchmark and current prices |
| Passage size | 200 words to begin with. 120 and 320 are tried on the development split, with the chosen model |
| ANN index | Accepted if its recall@10 against exact search is at least 0.97 |
| Storage | The simplest option that stays within a search p95 of 500 ms warm, a cold start of 8 s and 2 GiB of memory, measured in a container with one vCPU |
| Reranker | Only if nDCG@10 rises by at least 0.02 and p95 grows by at most 300 ms. The model is fixed in an addendum before the comparison |
| Selection by language model | Must beat the baseline selector on both quote precision and abstention. Otherwise the baseline runs. How "beat" is computed is fixed in an addendum before the answer step is built |
| Launch | The system abstains on at least 90% of the unanswerable questions, abstains needlessly on at most 15% of the answerable ones, and quote precision is at least 0.85 |
| Parties | recall@10 is reported per party. A party that deviates more than 0.05 from the mean is named on the method page |

The subset used to compare embedding models has fewer distractors than the full
corpus, which flatters every system. That is why the hybrid decision waits for
the full corpus.

## Outcomes, written in advance

One of these two statements will describe the result. Both are written now, so
that neither is worded after the fact.

- **Outcome A, hybrid wins.** RAG over the Riksdag's speeches 2002-2026 in
  production: hybrid retrieval with BM25 and vectors, answers as verbatim
  quotations with a deep link, evaluated on N questions.
- **Outcome B, BM25 wins.** RAG over the Riksdag's speeches in production, with
  verbatim quotations and deep links. I measured vector search and hybrid
  against BM25 on N questions. BM25 won and is what runs.

Under outcome B no vector index runs in production, and the project claims
none.

## What runs in CI

- **In CI:** unit tests on a synthetic corpus, with a deterministic embedder and
  the baseline selector, and the attack tests on the assembler. No keys and no
  network.
- **Locally:** the whole evaluation, with one command. Every measured figure is
  saved as JSON under `data/reports/search/`, and published text reads its
  figures from those files or cites them.

The evaluation itself does not run in CI.

## Stop rules

- No system is changed after its run on the test split.
- If a defect forces a run to be repeated, an addendum says what was wrong and
  keeps both results.
- If BM25 wins the first comparison, outcome B stands.
- The answer step gets three rounds of prompt work on the development split and
  no more, then one run on the test split. If the launch rule is not met, the
  page launches without the answer step, as a search that returns quotations
  and links.

## Addenda

### 2026-11-20: the corpus starts with the 2003/04 session

Reading the source ahead of the reconciliation turned up a systematic
difference in one session. The Riksdag publishes 2002/03 as text broken at the
width of the printed column: words are hyphenated across lines, and nothing
marks where a paragraph ends. The JSON, HTML and text datasets are alike in
this. Of the 12 479 records in that session, 12 345 are speeches by the eight
parties, and 109 073 of their lines end in a hyphen.

Quoting that text would mean deciding by rule how each of those words is put
back together, with no source to check the result against. I decided the same
day to leave the session out. Three things above change:

- **Scope.** The corpus runs from the 2003/04 session to the latest one. Its
  first speech is dated 2003-09-16.
- **The coverage rule.** The 99% applies to the sessions in scope.
- **Outcome A.** It reads "2003-2026" where it says "2002-2026".

The same reading showed that a speech record also carries paragraphs that are
the record's own: applause, the note that a debate was closed, an interjection
by somebody else. They are set aside when the corpus is built and are never
part of a passage or a quotation.

The figures are in `data/reports/search/anforanden_build.json`.

### 2026-11-21: the reconciliation, and what the corpus stores

The coverage rule was applied. The sessions in scope run from 2003/04 to
2026/27. In them, 286 323 of the 287 608 speeches in the existing corpus have
a counterpart in the new one: a coverage of 0.9955, against the 0.99 required.
The reconciliation was run four times as the corpus build was corrected. Every
run met the rule. Out of 287 608, the runs paired 286 325 on 2026-11-20, then
286 326, 286 323 and 286 323 on 2026-11-21. The report holds the last run.

The other way round, 286 323 of the 300 210 new speeches in protocols dated
within the span of the older corpus have a counterpart. Pairs are made only
within one protocol, and 12 413 of the 13 887 new speeches without one lie in
94 protocols that the older corpus lacks.

The pre-registration gave "differ systematically" no measure. It was judged
after the runs, and no bound was set afterwards. The judgement rests on two
figures over the paired speeches:

- **The speakers' words.** 46 359 of the 101 830 999 tokens in the new
  speeches are missing from the older text, so the older text holds almost
  every word of the new speeches.
- **What the older text holds beyond the new.** 4 283 956 of the 106 068 596
  tokens in the older speeches have no counterpart in the new speech's
  paragraphs. With the paragraphs set aside as the record's own counted too,
  4 115 331 have none. Beyond common words such as "av", "för" and "och", the
  commonest of these tokens include party letters, digits, "res", "utskottet"
  and "frånvarande", words found in vote lists and the order of business. I
  take this surplus to be protocol text that the PDF parser kept inside a
  speech, a defect of the older corpus.

I judged that the texts do not differ systematically, and the work goes on.

Coverage cannot show whether a new speech is complete. A new speech is paired
when at least 0.6 of its pairs of adjacent tokens are found in the older text,
so a speech that had lost its end would still pair. A speech would lose its end
if one of the ten rules that close a speech fired on the speaker's own
sentence. In `anforanden_build.json` those rules are `closing_note`,
`concurrence`, `leave_of_absence`, `chair_narration`, `chamber_narration`,
`committee_narration`, `demand_narration`, `document_read`, `election` and
`vote`. Between them they close 15 220 speeches. After the closing text, the
rule `after_closing` set aside 4 909 paragraphs.

When the corpus was built, a language model read every distinct closing text
and every distinct paragraph that `after_closing` set aside, apart from lists
of names. Those, such as the members who concurred in a speech or the name
lines after a closing, were checked by pattern. By that reading, none of the
closing texts or paragraphs holds words the speaker spoke. Reading them is a
job beyond the three named above, and I have not checked its readings against
my own.

Each quotation links to `https://data.riksdagen.se/anforande/<speech id>/html`,
the Riksdag's page for that one speech. The Riksdag's own data gives a speech
the address of its protocol page with `#anf<number>`. That page was fetched for
one speech in each of the 24 sessions, and 23 of the 24 held neither the anchor
nor the speech's text. Of 50 sampled speeches, 49 had a page at the chosen
address that passed every judgement in the report. Two of the judgements are
that the page names the speech and that it holds the opening words of each of
the speech's paragraphs in order. The address of the other, H109118-59,
answered with the speech's open-data XML record, which holds the text but not
the heading that names the speech. How many addresses answer this way is not
known.

The corpus starts in 2003, so the first period of the breakdown is 2003-2009;
the second stays 2010-2017. The last runs from 2018 to the latest year among
the speeches and takes its label from that year: 2018-2026 today, and
2018-2027 once the corpus holds a speech from 2027. The last session in scope,
2026/27, runs until September 2027.

The figures are in `data/reports/search/`: `reconciliation.json` for the
reconciliation, `anforanden_build.json` for what each rule set aside,
`deep_links.json` for the links and `passages.json` for the periods.
