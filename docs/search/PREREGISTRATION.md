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

None yet.
