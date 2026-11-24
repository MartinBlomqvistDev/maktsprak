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

### 2026-11-21: the embedding candidates, the subset they are compared on, and the hybrid

The vector work began on 2026-11-21, and the candidates were fixed the same day,
before any passage was embedded. They were checked against the Scandinavian
Embedding Benchmark and against the prices listed that day.

The addendum was committed on 2026-11-22. Before that, bekko-embedding-v1-a25m
was the only candidate that had run. Three trial runs of the code embedded 512,
512 and 300 passages with it on 2026-11-21, after the candidates were fixed.
They also embedded 22 questions, 20 of them from the development split. The best
passage was read for two of the questions. The measurements of the storage
options have embedded 100 questions of the development split with it, to compare
its runtimes and precisions. No run scored a search against the questions'
answers. After the trial runs, the text gained the figures for memory, the check
before the test split and the precision of each model. The candidates, the
way the subset is drawn and the rule that compares them stayed as they were.

**The benchmark.** The Scandinavian Embedding Benchmark is now part of MTEB, as
`MTEB(Scandinavian, v1)`. Its results were read from MTEB's results dataset,
`mteb/results`, at revision `dfe159b89b95b84787cd2a896e578991820545f4` (Parquet
export `d9ec6baf92aaa634a81d7ac0d916bad2864e42a0`). Two of the benchmark's
seven retrieval tasks are Swedish, SwednRetrieval and SweFaqRetrieval. The
models were ranked by the mean of those two, with the mean of all seven beside
it. `search/pipeline/check_embedding_candidates.py` reproduces the table, and
its report is `data/reports/search/embedding_candidates.json`.

**Size classes.** A small model has under 150 million parameters. A large one
is the largest that can still embed a question inside the search service. The
storage rule gives the service 2 GiB of memory and one vCPU, and the model
shares that memory with the index. A model of 0.6 billion parameters takes
1.12 GiB in 16-bit floats, which leaves room for the index only if its vectors
are compressed. For the corpus's 563 909 passages, 1 024 dimensions take
2.15 GiB as float32, 1.08 GiB as float16 and 0.54 GiB as int8. That puts the
ceiling at about 0.6 billion parameters. Apart from saga-embed_v1, every open
model ahead of snowflake-arctic-embed-l-v2.0 on Swedish retrieval has from 1.7
to 14 billion parameters, so none of them can run that way. One of them,
Qwen3-Embedding-8B, is the API candidate.

**The candidates.** Retrieval scores are the benchmark's own, on the Swedish
tasks and on all seven.

| Class | Model | Revision | Parameters | Dimensions | Licence | Swedish | All |
|---|---|---|---|---|---|---|---|
| Small multilingual | `hotchpotch/bekko-embedding-v1-a25m` | `44f0b8af0f487acd0ccf1a7cb7ae7a29a6dfc09c` | 123 M | 384 | MIT | 0.770 | 0.673 |
| Swedish | `SagaEmbed/saga-embed_v1` | `5f5975288b821c190ab9e51134cea45516199816` | 395 M | 1 024 | none stated | 0.848 | 0.723 |
| Large multilingual | `Snowflake/snowflake-arctic-embed-l-v2.0` | `ac6544c8a46e00af67e330e85a9028c66b8cfd9a` | 568 M | 1 024 | Apache-2.0 | 0.796 | 0.654 |
| API | `qwen/qwen3-embedding-8b` through OpenRouter, served by Nebius in bf16 | weights `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` | 8 B | 1 024 of 4 096 | Apache-2.0 | 0.810 | 0.712 |

MTEB evaluated earlier commits of all four models. By the hashes Hugging Face
lists, each pinned revision holds the same weight and tokenizer files as the one
MTEB evaluated. The later commits changed the model cards, added a list of
Matryoshka sizes to arctic-embed's configuration, and added files the
measurement does not load. Which commit Nebius serves Qwen3-Embedding-8B from
cannot be checked from outside.

- **Small.** bekko-embedding-v1-a25m has the best Swedish retrieval of the
  models under 150 million parameters. Its smaller sibling,
  bekko-embedding-v1-a8m, has 0.720, and multilingual-e5-small 0.692.
- **Swedish.** saga-embed_v1 was trained for embedding in the Scandinavian
  languages, and its model card names AI Sweden's ModernBERT as its base. Of
  the 186 models with results on both Swedish tasks it is third. The next model
  trained for these languages, EmbeddingGemma-Scandi-300m, has 0.549. KBLab's
  sentence-bert-swedish-cased has no result on the two tasks. The model card of
  saga-embed_v1 names no licence, and the licence rule below covers that.
- **Large.** snowflake-arctic-embed-l-v2.0 leads the multilingual models from
  150 million to 0.6 billion parameters. Next come jina-embeddings-v3 at 0.790,
  whose licence allows only non-commercial use, and bge-m3 at 0.788.
- **API.** Anthropic offers no embedding model, so the API model comes from
  another provider and is paid with my own key. The rule lets it win only if
  embedding the whole corpus fits 10 USD, and decision D6 caps all embedding at
  the same 10 USD. Six API models score higher than Qwen3-Embedding-8B on
  Swedish retrieval, and none of them fits:
  - gemini-embedding-001 (0.853) costs 0.15 USD per million tokens. Even at one
    token per word, the corpus's 107 234 571 words would cost 16.09 USD. Google's
    batch price of 0.075 USD fits only if the corpus comes to under 133.3 million
    tokens. Google publishes no tokenizer for the model, so that cannot be
    counted before paying, and the project holds no key with Google.
  - voyage-finance-2 (0.837) and voyage-multilingual-2 (0.827) cost 0.12 USD.
    Their published tokenizer counts the corpus at 230 054 800 tokens, which is
    27.61 USD.
  - voyage-3.5 (0.821) and voyage-3 (0.812) cost 0.06 USD. Their published
    tokenizer counts 212 071 882 tokens, which is 12.72 USD, or 8.53 USD with
    Voyage's batch discount of 33%, on an account the project does not have.
    The procedure below embeds the subset and then the whole corpus at three
    passage sizes, which would come to about 26.5 USD even at the batch price.
  - text-embedding-3-large (0.819) costs 0.13 USD, so even one token per word
    would cost 13.94 USD.

  Qwen3-Embedding-8B's own tokenizer counts the same 212 071 882 tokens. At
  0.01 USD per million through OpenRouter, the whole corpus costs 2.12 USD and
  the whole procedure about 6.6 USD. Its weights are open, so the same
  model can be run elsewhere if the service changes.

The prices are those listed on 2026-11-21 by OpenRouter, Google and Voyage. The
report gives the source of each.

**How each candidate is run.**

- Each uses the prompts its model card gives, and none is tuned. bekko uses no
  prompt. saga-embed_v1 puts `task: retrieval | query: ` before a question and
  `title: none | text: ` before a passage. arctic-embed puts `query: ` before a
  question. Qwen3 puts `Instruct: Given a web search query, retrieve relevant
  passages that answer the query`, a line feed and `Query:` before a question.
- A passage is embedded as its paragraphs joined by line feeds, the same text
  that BM25 indexes.
- The local models compute in float32. Qwen3 runs in bf16, the precision its
  provider serves.
- Vectors are normalised and compared by cosine, and the search is exact.
- A local model reads at most 1 024 tokens of a passage, or its own configured
  limit where that is lower: saga-embed_v1 reads 512. Qwen3 reads the whole
  passage; the longest has 1 836 of its tokens. Each run's manifest counts the
  passages that were cut.
- Qwen3 is measured at 1 024 of its 4 096 dimensions: the first 1 024,
  normalised again, which its Matryoshka training allows. At 4 096 dimensions
  the corpus's vectors would take 8.6 GiB as float32, and even as int8 they
  would exceed the 2 GiB of the storage rule. The local models keep their own
  sizes.

**The subset.** It holds every speech that answers a question of the
development split in the natural, keyword, interpellation, filtered or stance
set: 5 095 speeches with 10 231 passages. Speeches drawn at random fill it up
to about 60 000 passages, and the draw gave 60 189. It uses seed 20261125 and
takes whole speeches within each pair of year and party, so that each pair gets
its share of the corpus's passages. The test split is not read. The subset's
passage ids are hashed into `data/reports/search/vector_subset.json` before any
vector is scored.

**The comparison.** Each candidate's vectors search the subset on their own,
and the highest recall@10 on the natural questions of the development split
wins. Among the candidates within 0.01 of it, the lightest to operate is
chosen, in this order: bekko-embedding-v1-a25m, saga-embed_v1,
snowflake-arctic-embed-l-v2.0, Qwen3-Embedding-8B. The local models are ordered
by size. The API model comes last, because with it every search would depend
on an outside service and a paid key. BM25 is run on the subset beside them and
decides nothing.

**The licence rule.** A model is chosen only if its licence allows it to run on
the site. saga-embed_v1 is measured all the same. If it comes first and its
author has granted no licence by the time the choice is made, the choice passes
to the other candidates under the same rule, and the report says so.

**After the choice.** The chosen model embeds the whole corpus in passages of
200 words, then of 120 and of 320, packed by the same rule. On the development
split, the hybrid is tuned over the three passage sizes and the RRF constant
`k` at 10, 20, 30, 60 and 100. A passage scores the sum, over the two rankings,
of `1 / (k + rank)`, where rank counts from 1 and each ranking holds the 1 000
best passages under the question's filter. Inside the hybrid, BM25 is built at
the passage size being tried, with the configuration tuned on 2026-11-21:
stemming without stop words, `k1` 1.2 and `b` 0.9. The highest recall@10 on the
natural questions wins. A tie goes to the higher MRR@10, then to 200 words,
then to the `k` nearest 60, and between two equally near, to the smaller. In
the comparisons, BM25 stays the one tuned at 200 words, and the vectors alone
are the hybrid's own vector ranking.

**What is measured and what runs.** Every comparison in this phase searches
float32 vectors exactly, so it measures what each model can reach. The service
may embed and search differently. It may embed the question in lower precision
or through another runtime, such as ONNX, and its question vectors then differ
from the ones measured here. Vectors of 1 024 dimensions fit the storage rule
only if the service holds them compressed, as float16 or int8, or in an
approximate index. A local model's weights have to fit beside them.

**The check before the test split.** The tuned configuration is therefore
checked in a build of the service before the test split is opened. The build
embeds the questions exactly as the service will, with the same runtime,
precision and processor architecture. It searches the passages' vectors in the
format and index the service will use. The rule for an approximate index
applies to the whole build. Its recall@10 against the exact float32 search must
be at least 0.97, over the questions of the development split, each searched
with any filter it carries. The build must also meet the storage rule. More
than one build may be tried. If every build fails, the passage size the tuning
ranked next is checked the same way, with its best `k`. If every passage size
fails, the choice passes to the next candidate under the comparison's rule, and
that candidate is tuned and checked in turn. An addendum records each such
change before the test split is opened, and decides what runs if every
candidate fails. Passages added later are embedded the way the checked
ones were.

**The run on the test split.** It is made once, on the whole corpus, after the
check above has passed and an addendum has recorded the tuned configuration
with the hash of its report. The script refuses to run without that record and
refuses to run a second time. The paired bootstrap uses the seed of the BM25
baseline, 20261124.

### 2026-11-22: the answer step, before it is built

This addendum fixes what the sections above leave open about the answer step:
the model that selects, the questions it answers, how its quotations are
graded, and what it takes for selection by a language model to beat the
baseline selector. It is committed before any code of the answer step is
written.

**The selector.** Claude Opus 5.5 (`claude-opus-5-5`) selects the paragraphs,
at low effort in the first round. The selector that is measured is the one the
page runs. When the page's monthly budget for the model is spent, the page
answers with the baseline selector and says so. The model answers in structured
output: whether the candidates answer the question, and at most four
selections, each a candidate's passage id with the first and last of its
paragraphs, numbered from 1 within the passage. It returns nothing else, no
reason and no words of its own.

**The questions.** The answer step is run on the natural, filtered and stance
questions, each written from a passage that answers it, and on the unanswerable
ones. That is 180 answerable and 60 unanswerable questions on the development
split, and 270 and 90 on the test split. The keyword questions are left out:
they are search terms, and the judge's calibration holds none of them. So are
the interpellations, whose question is the heading of a debate.

**The candidates.** Each question gets the ten passages that BM25 ranks
highest, under its filter when it has one. Both selectors get the same ten. The
rounds on the development split run on BM25 as the tuning set it: with
stemming, no stop words dropped, `k1` 1.2 and `b` 0.9 (`stem-nostop` in
`data/reports/search/bm25_tuning_dev.json`). They run before comparison 1 has
decided the retrieval, since the API credit set aside for them must be used by
2026-11-25. The run on the test split uses the retrieval that comparison 1 puts
in production, with the prompt unchanged. The test split stays closed until
that run.

**The baseline selector** quotes the first candidate, from the first to the last
of its paragraphs that hold a word of the question outside Snowball's stop
words, compared after stemming. When no paragraph holds one, it quotes the whole
passage. It abstains when that passage scores below a share of the most the
question's terms can score, which is `k1` + 1 times the sum of their idf. A
term the index lacks counts with the idf of a term no passage holds. The share
is the value with the highest balanced accuracy on the development split, and a
tie goes to the lowest. Before the run on the test split it is chosen again the
same way, with the candidates of the retrieval in production. The prompt is not
tuned again, so this favours the baseline.

**Quote precision.** A quotation is a selection that the assembler accepts: its
paragraphs as the corpus holds them, separated by a blank line. The judge
grades every quotation given to an answerable question with the model, effort
and prompt it had for the calibration pairs: Claude Opus 5.5 at low effort, and
`search/eval/prompts/judge.md` unchanged (SHA-256
`c7361c709ba78f606ad8d70ebe460b2b238dfa4a5c600ee11141889f80803168`). Quote
precision is the share of a selector's quotations that the judge grades 2. A
quotation given to an unanswerable question is not graded and counts as a
failure to abstain. Each pair of question and text is graded once, and the
grade serves every selector and round that gives that quotation.

The judge's grades count only if its kappa reaches 0.6 against my grades of the
100 calibration pairs, which I have not yet given. If it falls short, the
grades of the development rounds decide nothing. The round that goes to the
test run is then chosen on balanced accuracy alone, and an addendum fixes how
the quotations of the test run are graded by hand before that run.

On the development split the judge also grades the ten candidates of every
answerable question, so that an abstention can be told apart from a miss of the
retrieval. Those grades are description.

**Abstention.** A selector abstains when its answer holds no quotation. It
abstains needlessly when it abstains on an answerable question, whatever its
candidates hold, so a miss of the retrieval counts against both selectors
alike. Balanced accuracy is the mean of two shares: the unanswerable questions
on which a selector abstains, and the answerable ones on which it does not.

**Beating the baseline.** Selection by a language model beats the baseline
selector when both of these hold on the test split, each under the paired
bootstrap with 10 000 resamples:

1. The difference in quote precision, language model minus baseline, has a 95%
   interval whose lower bound is above 0. Each resample draws from the
   answerable questions, and each selector's precision is computed over its
   quotations on the questions drawn.
2. The difference in balanced accuracy has a 95% interval whose lower bound is
   above 0. Each resample draws from the answerable and the unanswerable
   questions separately, so that it keeps the size of each.

The two shares behind balanced accuracy are reported beside it and decide
nothing alone. The launch rule is read on the point estimates of the test run.

**Selection errors.** The assembler accepts a selection only if its passage is
among the question's candidates, its range lies within the passage with the
first paragraph no later than the last, it overlaps no selection accepted
before it, and it is among the first four. An output that does not have the
form above, or carries any other field, is rejected whole. So are the
selections of an answer marked unanswerable. Every rejected selection is a
selection error. A request that fails, is refused or is cut off is sent once
more. If it fails again, the baseline selector answers the question, as the
page would, and the report counts it.

**Rounds.** The answer step gets at most three rounds on the development split.
A round may change the prompt and the effort. The model, the candidates and the
cap of four quotations stay as they are. Each round is recorded with its
prompt's hash. The prompt that goes to the test run is the one whose round has
the highest mean of quote precision and balanced accuracy on the development
split, and a tie goes to the later round.

**Party balance** is counted per selector: the parties of the quotations given
to answerable questions, against the parties of all the candidates of those
questions.

The requests of the development rounds and of the judge go through Anthropic's
message batches. On the page, each question's request is sent on its own, with
the same parameters.

### 2026-11-22: the search service's memory, before the embedding comparison

The storage rule gives the search service 2 GiB, the memory of the inference
service's instance on Cloud Run. That figure was set before anything had
counted the query model, which shares the memory with the index. The storage
spike measured both (`docs/search/adr-001-index-store.md`). With vectors of
1 024 dimensions, the only exact store within 2 GiB holds them as int8, at a
p95 of 470 ms. No local model of that size could be measured beside it. Kept at
2 GiB, the rule would choose the embedding model by its memory before the
comparison has measured how well it retrieves.

From this addendum on, the service runs with one vCPU and 4 GiB, the most Cloud
Run allows for one vCPU, and the storage rule's bound on memory is 4 GiB. The
rest of the rule stands: a search p95 of 500 ms warm and a cold start of 8 s,
measured in a container with one vCPU. The storage decision holds under both
bounds.

The memory costs nothing at this page's traffic. A service billed by request
gets 360 000 GiB-seconds a month free, by Cloud Run's price page on 2026-11-22,
which at 4 GiB is 25 hours of instance time. A ping every five minutes, as the
inference service gets, uses well under one of them.

I decided this before the embedding comparison ran, so the bound cannot
follow from its result.

### 2026-11-23: three models in the first round

This addendum changes two things in the addendum of 2026-11-22: the model that
selects, and what a round may change. It is committed before any paid request
of the answer step. Until now only the baseline selector has run, on the
development split. It calls no model.

**Three models in round 1.** Claude Opus 5.5 (`claude-opus-5-5`), Claude
Sonnet 5.5 (`claude-sonnet-5-5`) and Claude Haiku 5.5 (`claude-haiku-5-5`) each
answer the 240 questions of the development split. They get the same prompt,
the same candidates and the same parameters: low effort, at most 8 000 output
tokens, and thinking left at each model's default. That default is adaptive
thinking, and Opus 5.5 cannot turn it off, so thinking is on for all three. The
same effort need not mean the same depth in two models, and what is measured is
each model at that setting.

**The choice of model.** A model's score is the mean of its quote precision and
its balanced accuracy. A model qualifies if its score is at most 0.01 below the
highest score and its quote precision at most 0.01 below the highest quote
precision. The cheapest model that qualifies is chosen. By list price, Haiku
5.5 is the cheapest and Opus 5.5 the most expensive. The condition on precision
keeps a cheaper model from winning on fewer needless abstentions while it
quotes worse. The judge grades every model's quotations as the addendum of
2026-11-22 sets out. It is shown the question and the quotation only, so it
cannot tell which model chose a quotation.

The chosen model runs on the test split and on the page, since the selector
that is measured must be the one the page runs. Each model's score, the two
shares behind its balanced accuracy, its party balance and its cost per
question are reported. Beyond the choice, they are description.

**Rounds 2 and 3** run only the chosen model and change only the prompt. The
effort stays low. Among that model's rounds, round 1 included, the prompt that
goes to the test run is picked by the rule of 2026-11-22.

**If the judge falls short** of a kappa of 0.6, quote precision is not measured
on the development split, and nothing then shows that a cheaper model quotes as
well as Opus 5.5. Round 1 then runs Opus 5.5 alone, as the addendum of
2026-11-22 has it.

**Why.** The page's budget for the model is 2 USD a month. At list price, a
question costs about 0.04 USD with Opus 5.5, 0.02 with Sonnet 5.5 and 0.001
with Haiku 5.5. The budget therefore answers roughly 50, 100 and 2 000
questions a month before the baseline takes over.

### 2026-11-24: the test run of the answer step, graded by hand

The judge's kappa against my grades of the 100 calibration pairs is 0.321,
below 0.6. Under the addendum of 2026-11-22, the judge's grades of the
development rounds therefore decide nothing, the round that goes to the test
run is chosen on balanced accuracy alone, and I grade the quotations of the
test run by hand. Under the addendum of 2026-11-23, round 1 ran Claude Opus 5.5
alone. Rounds 1 and 3 share the highest balanced accuracy on the development
split, 0.922, and the tie goes to the later round, so the test run uses the
prompt of round 3 (SHA-256
`2d6a0dc174ba03ddbf19c86b1537a351ae2d8c366cc0ab04fff194d40d82af27`).

This addendum fixes how the test run is graded, and it moves the run ahead of
comparison 1, so that it can be made on the API credit before 2026-11-25. It
is committed before the answer step reads the test split.

**The candidates.** On the test split and on the page, the answer step's
candidates are the ten passages BM25 ranks highest under the question's
filter, as tuned on 2026-11-21: stemming, no stop words dropped, `k1` 1.2 and
`b` 0.9, at 200 words. This holds whatever comparison 1 decides for the list
of results, so the page runs the system that was measured. If comparison 1
puts the hybrid in production, the answer step stays on BM25, since the test
split allows the answer step one run and no system is changed after its run.
The baseline's threshold stays the one chosen on the development split, whose
candidates are the same.

**The run.** Claude Opus 5.5 answers each test question at low effort with the
prompt of round 3, unchanged, through a message batch. The baseline selector
answers the same questions from the same candidates.

**The seal.** The addendum on the embedding candidates checks the hybrid's
configuration before the test split is opened. The answer step's run comes
before that check, so its results stay sealed until the retrieval run on the
test split has been made. Until then nothing is computed from them and I grade
nothing. No figure of retrieval, such as recall@10 or the rank of a gold
passage, is ever computed from the answer step's files; the retrieval run
ranks the test questions itself.

**Grading by hand.** I grade the quotations given to 100 answerable questions
of the test split, drawn at random with the seed 20261124. Every quotation
that either selector gives to a drawn question is graded, each pair of
question and text once, on the scale and with the instructions of the 100
calibration pairs. The pairs come in an order drawn with the same seed, and
each shows only the question and the quotation, so I cannot tell which
selector chose it. Quote precision is the share of a selector's quotations on
the drawn questions that I grade 2, and the paired bootstrap for it resamples
the drawn questions. Balanced accuracy needs no grade and is computed on every
question of the test split. The comparisons and the launch rule are otherwise
computed as the addendum of 2026-11-22 sets out.

### 2026-11-24: the tuned configuration, and the check before the test split

The tuning on the development split chose passages of 200 words and an RRF
constant `k` of 10, with qwen3-8b. It reached recall@10 0.600 and MRR@10 0.347
on the 120 natural questions, alone at the top of the grid. Its report is
`data/reports/search/hybrid_tuning_dev.json`, and the SHA-256 of its text is
2c0f2962dbd8388f52a631a77c76ac0abd1bb73d7e1bb01fda908c1a790251f1. Ten is the
smallest `k` in the grid, and at 200 and 320 words every larger `k` scored
lower. The grid stays as registered.

The check ran in a build of the service. It holds the corpus's vectors from
qwen3-8b as a float32 matrix and searches them exactly beside the tuned BM25,
as ADR-001 decided, in a container with one vCPU and 4 GiB. The build got each
question's vector the way the service will: one question per request to
OpenRouter, routed to Nebius in bf16 without fallback. All 855 questions of
the development split that the evaluation searched went to the build, each with
the filter it carries. The build's ten best held on average 0.994 of the exact
float32 search's ten best, against a floor of 0.97. When the exact search uses
the vectors the build asked for, it agrees with the build to 0.9995, and the two
vector legs return the same ten passages for every question. The difference
lies in the API's answers. Asked for one question at a time, it returns vectors
that differ slightly from those it returned to the evaluation's requests of 64;
the lowest cosine between the two was 0.999668. Under the storage spike's
workload the build had a p95 of 168 ms, a slowest cold start of 5.0 s and a
peak of 3.15 GiB. The check passed, and its figures are in
`data/reports/search/service_check_dev.json`.

A question waits for its vector before it is searched. Over the 855 requests on
the afternoon of 2026-11-24, the median wait was 1.25 s, the 95th percentile
5.6 s and the longest 13.8 s. The service waits for the vector. If none has
come within 30 s, it answers from BM25 alone and says so. A shorter limit would
send so many questions to BM25 alone that the service would no longer be the
build that was checked. At 3 s, 22.5 percent of these questions would have gone
there. I set the limit after seeing these waits, before the test split was
opened.

The test run embeds the test questions in requests of 64, as the evaluation
embedded those of the development split.

### 2026-11-24: the answer step on the hybrid's candidates, run again on the test split

Retrieval was run on the test split on 2026-11-24, and comparison 1 put the
hybrid in production: outcome A (`data/reports/search/retrieval_test.json`).
The addendum of 2026-11-24 on the answer step's test run had settled that case
in advance: the answer step would stay on BM25's candidates, since the test
split allows it one run and no system is changed after its run. This addendum
reverses that, and the answer step is run on the test split a second time, on
the hybrid's candidates. That departs from three rules: the test split is run
three times and no more, no system is changed after its run on the test split,
and the answer step was to stay on BM25 whatever comparison 1 decided. The test
split is now run four times.

**Why.** That addendum moved the answer step's run ahead of comparison 1 so
that it could be made on the API credit before 2026-11-25. The credit runs
until 2026-11-28. Had the date been right, the run would have followed the
addendum of 2026-11-22: it would have waited for comparison 1 and used the
retrieval that comparison puts in production. Kept on BM25, the page would
choose its quotations from BM25's ten passages while its list of results shows
the hybrid's.

**What I knew when I decided.** The first run is known only by its record,
`data/reports/search/answer_test_run.json`, which holds its settings, the
message batch, the SHA-256 of its answers and its cost, 5.30 USD. No figure has
been computed from its answers, and I have graded none of their quotations. In
the new run nothing is left to choose. The model, the prompt, the effort and
the questions stay as they were. The candidates follow from comparison 1, and
the baseline's threshold is chosen again by the rule of 2026-11-22.

**The first run** decides nothing. It is kept, in line with the stop rule that
keeps both results when a run is repeated, and its files are left as they are.
The SHA-256 of its answers is
`91c2cb55014f5bb306b228e1fa03e5cebb71ce6b04b86fd54fe74396f799320d` for the
model and `c99feac638426381278ea727b972696dd97d38881708f43b3ea3ef86a33e4f41`
for the baseline. I open it only after the comparisons and the launch rule of
the new run have been computed with my grades. Its balanced accuracy and the
two shares behind it are then reported as description, and its quotations are
not graded.

**The candidates.** On both splits and on the page, a question gets the ten
passages the hybrid ranks highest under its filter. The hybrid is the one the
tuning chose, as the addendum of 2026-11-24 on the tuned configuration records
it: qwen3-8b at 200 words, merged with BM25 by reciprocal rank fusion with
`k` 10. The addendum of 2026-11-21 on the embedding candidates fixed the rest:
BM25 as tuned on 2026-11-21, and each ranking holding the 1 000 best passages
under the filter. The vectors are searched exactly, as float32. Each question
gets its vector the way the service will: one question per request to
OpenRouter, routed to Nebius in bf16 without fallback. The retrieval run did
not store the test questions' vectors, and the 150 unanswerable questions were
never embedded. All 600 questions of the answer step are therefore embedded for
it. If no vector has come within 30 s, the page answers from BM25 alone and
says so, and the answer step then gets BM25's ten.

**The baseline** quotes the hybrid's first candidate. As before, it abstains
when that passage's BM25 score falls below a share of the most the question's
terms can score. A passage that holds none of the question's terms scores 0.
The share is chosen again on the development split, with the hybrid's
candidates, before the new run.

**The run.** Claude Opus 5.5 answers the 360 questions of the test split at low
effort, with at most 8 000 output tokens and the prompt of round 3 unchanged
(SHA-256 `2d6a0dc174ba03ddbf19c86b1537a351ae2d8c366cc0ab04fff194d40d82af27`),
through a message batch. The baseline answers the same questions from the same
candidates. The rounds on the development split are not run again, so the
prompt is the one tuned on BM25's candidates. The run is made once. It is not
sealed, since the retrieval run on the test split has been made. As before, no
figure of retrieval is computed from the answer step's files.

**Grading by hand** follows the addendum of 2026-11-24 on the answer step's
test run and applies to the new run: the 100 answerable questions drawn with
the seed 20261124, the pairs in an order drawn with the same seed, each showing
only the question and the quotation. The comparisons and the launch rule are
computed as the addendum of 2026-11-22 sets out.
