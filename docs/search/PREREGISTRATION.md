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

### 2026-11-21: the question sets, and how BM25 and the judge are set up

The question sets are written. Their hashes are recorded here before any system
is scored on them, together with the sessions of the interpellation set and the
choices the sections above left open.

| Set | Questions | Development | Test | SHA-256 |
|---|---|---|---|---|
| Natural | 300 | 120 | 180 | `a2dedbf93034594e1a24b733dd6f88d6352cad8814590e55127bb492530e2418` |
| Keyword | 300 | 120 | 180 | `90cbb69a0efc64e7dd893ac2448d90f42e67ffcda7a795f21c3deab4477b9d41` |
| Interpellations | 1 387 | 555 | 832 | `7241d388ac2123287b305b1ae758ed783b5f2c3a937e7d097b66a865ea18f34a` |
| Filtered | 100 | 40 | 60 | `2a65b555cac6c9f40ff40daecb03655f68d533f6577fd063b37a8b6b4e33fb92` |
| Unanswerable | 150 | 60 | 90 | `39d2ca9b16d0e5b22cc0c0f003f05f90cf2d06b0ada814a76b7bdc74c676ece1` |
| Stance | 50 | 20 | 30 | `75b1be3bc981b8527502f6408790fab2121eabf108326ce2146f3ef3ea0be934` |

The sets are in `data/reports/search/questions/`, one JSON Lines file each.
`questions_manifest.json` beside them records the seed, the prompts' hashes,
the cost and how the requests ended.

**How the generated sets were written.** Claude Opus 5.5
(`claude-opus-5-5`) wrote them at low effort, from the prompts in
`search/eval/prompts/`. The requests went through Anthropic's message batches
and cost 2.52 USD. The model, the prompt's hash and the date are stored with
each question. The prompts were
tried first on 30 passages of a draw of their own, with its own seed, in two
runs costing 0.64 USD together. No passage of that pilot could be drawn for the
sets.

- **The draw.** Every passage is sorted by the year of its speech, its party,
  its length band and a random key; the bands are the terciles of passage
  length, cut at 173 and 211 words. A systematic sample from that order, with
  seed 20261121, gave 450 places: 300 for natural questions, 100 for filtered
  and 50 for stance questions, spread over the order in proportion. In each set,
  two of every five places in that order belong to the development split. That
  was fixed before any question was written.
- **Replacement.** When the model found no question a voter would reasonably
  ask of a passage, the place moved to the next passage of the same year, party
  and length band. A place could try up to twelve passages. An answer that
  broke a rule got one more request about the same passage, and a second
  failure also moved the place on. Of the 490 requests that count, after the
  correction described below, 450 filled a place, 29 found no question in the
  passage, 7 broke a rule of form and 4 reused a rare word. They asked about
  481 passages.
- **Rare words.** A word is rare when fewer than one passage in a thousand
  holds its stem: fewer than 563.9 of the 563 909, by Snowball's Swedish
  stemmer. The request listed the passage's rare words, and code checked the
  question for them after stemming. The rule above names the natural questions;
  it also applied to the filtered and the stance questions, which are natural
  questions of a set form.
- **Keyword questions** are two to four single words, written in the same
  request as the natural question from the same passage.
- **Filtered questions** were written like the natural ones, from 100 passages
  of their own. Every other one is searched under its party, the rest under its
  period.
- **Stance questions** ask for the arguments against X, "Vilka argument har
  förts fram mot X?", or for X, "Vilka argument har förts fram för X?",
  following the side the passage takes. The table above gives only the first,
  as an example.
- **Unanswerable questions:** 50 off topic, 25 naming a member who never sat in
  the Riksdag, 25 about Sverigedemokraterna in a year from 2003 to 2009, and 50
  about a year from 2028 to 2035. Code checked what can be checked: each name
  against every speaker in the corpus, each year named in a question.
- **One model.** A request the model declined would have gone to no other
  model, so one model wrote every generated question. None was declined.

**A defect in the keyword check, corrected before the sets were assembled.**
The check first counted a hyphenated word such as "a-kassan" or
"EU-medborgare" as two words. It turned away 47 answers that broke no rule, and
some of the places involved moved on to other passages because of it. The check
now counts a word the way the corpus does, as a run of characters between
whitespace. The run sent 538 requests for the questions written from a passage.
Every logged answer was judged again with the corrected check, and the waves
were run again against the log, each request taking the logged answer to the
same request. No new request was needed. 490 of the 538 count, and 21 places
now hold a passage they had moved on from. In one place, natural-0086, the
answer that counts came from a second request that did not name the rare word
"skattefria" the first answer had reused, because that first answer had been
turned away for its keywords. It reuses no rare word and passes every check.

**The interpellation set.** The sessions are 2007/08, 2011/12 and 2022/23,
one drawn with seed 20261121 from each period's sessions whose two calendar
years both lie in that period. A debate is known by its heading, which names
the interpellation by session and number. The Riksdag's `rel_dok_id` cannot
serve: in 2014/15 and 2015/16 several debates of one day share one value, and
some debates have none. The question is the heading's text after its first
" om ". In each session two of every five debates, in a random order, belong to
the development split.

**The review.** The 50 questions are drawn with seed 20261122 from the four
sets written from a passage, in proportion to their size: 20 natural, 20
keyword, 7 filtered and 3 stance questions. A question is approved when I
answer yes to both questions. For keyword questions, a reasonable question
means reasonable search terms, and an answer means that the passage is a good
hit for them. 85% is 43 of 50. The outcome will be in
`data/reports/search/review_generation_1.json`.

**BM25 on the development split.** Four analyzers stem or do not, and drop
Snowball's 114 Swedish stop words or keep them. Each is tried with `k1` at 0.6,
0.9, 1.2, 1.5 and 1.8 and `b` at 0.3, 0.45, 0.6, 0.75 and 0.9. The
configuration with the highest recall@10 on the natural questions of the
development split is chosen. A tie goes to the higher MRR@10, then to the
configuration nearest Lucene's defaults (`k1` 1.2, `b` 0.75), then to the
analyzer listed first in `search/index/analyzer.py`. That configuration is the
BM25 of every comparison.

**The judge.** Claude Opus 5.5 at low effort, with the prompt
`search/eval/prompts/judge.md` (SHA-256
`c7361c709ba78f606ad8d70ebe460b2b238dfa4a5c600ee11141889f80803168`), grades
how well a passage answers a question: 2 when it answers it, 1 when it is about
the question's topic but does not answer it, 0 when it is about something else.
The 100 pairs for my own grades come from the pilot's questions, which belong
to no set, so the calibration touches neither split. 25 questions are drawn
with seed 20261123, and each is paired with the passage it was written from and
with the three other passages BM25 ranks highest, with stemming, with the stop
words dropped and at Lucene's defaults. The
judge grades them without seeing my grades. The kappa that decides is Cohen's,
unweighted, over the three grades: at 0.6 or more the judge's grades are used.
The linear-weighted kappa, and the kappa of grade 2 against the other two, are
reported and decide nothing.
