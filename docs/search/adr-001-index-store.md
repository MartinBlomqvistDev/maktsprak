# ADR-001: where the search service keeps its index

**Status:** decided on 2026-11-22 by the storage rule of the pre-registration.
The decision holds under the 4 GiB its addendum of the same day gives the
service.

**Figures:** `data/reports/search/storage_spike.json`,
`data/reports/search/storage_spike_memory_4g.json` and
`data/reports/search/lancedb_fts_probe.json`, written by the scripts in
`search_service/spike/`. Every figure below is read from those files.

## The question

The search service answers a question with a filtered hybrid search over the
corpus's 563 909 passages. BM25 and the vectors each rank the 1 000 best
passages under the question's filter, as deep as the hybrid is tuned, and
reciprocal rank fusion merges the two rankings. The ten best come back with
their text. The service runs on Cloud Run with one vCPU and scales to zero, so
a cold start loads the whole index again.

The pre-registration decides where the index lives: "The simplest option that
stays within a search p95 of 500 ms warm, a cold start of 8 s and 2 GiB of
memory, measured in a container with one vCPU". Two options were measured:

- **A.** LanceDB embedded in the service, with the index baked into the image.
  Vectors, full-text index, text and filter columns sit in one store.
- **B.** No database. The vectors are a numpy matrix beside the BM25 index of
  `search/index/lexical.py`, the configuration tuned on the development split,
  and the filters are numpy columns.

B is the simpler of the two: it adds no database, and its lexical leg is the
BM25 the evaluation measures. B is therefore chosen if it holds the budget, and
A only if it does not.

## How it was measured

**The container.** Every measured container ran under `docker run --cpus=1
--memory=2g --memory-swap=2g`: one CPU's time, 2 GiB and no swap. The stores of
1 024 dimensions were measured once more with 4 GiB (see "With 4 GiB"). The
thread pools of numpy's BLAS, of torch and of ONNX Runtime, and LanceDB's
compute threads, were set to one thread, as the service's own image would set
them.
LanceDB's IO threads kept their default. Set to one as well, they raised
lance-384-float32's p95 from 670 ms to 1 211 ms, and the report keeps that run
among its superseded runs. The engine was Docker Engine 29.1.3 in Ubuntu 24.04
under WSL2, with cgroup v2, on a laptop with an Intel Core Ultra 7 155H. The
script that drives the measurement ran inside WSL too, and timed each request
from outside the container. Cloud Run runs only x86_64 ("Cloud Run
specifically supports the Linux x86_64 ABI format", in its container contract),
and so does this machine.

**What a store holds.** The text and the BM25 index are the corpus's own: all
563 909 passages, and the BM25 configuration tuned on the development split
(Snowball's Swedish stemmer, stop words kept, `k1` 1.2, `b` 0.9). The vectors
are synthetic, unit vectors drawn from a seeded generator, since no embedding
model has been chosen. They come in 384 dimensions, bekko-embedding-v1-a25m's,
and in 1 024, which the other three candidates use. Synthetic vectors cost what
real ones of the same shape cost to store, load and scan, so the times and the
memory hold. They have none of the structure an approximate index relies on,
so what an approximate index finds among them says nothing about its recall on
real vectors.

Option B's stores hold the vectors as float32, as float16, and as int8 with
one scale per dimension, and search them exactly. Option A's stores hold them
as float32 and float16 and search them exactly, and as float32 under two
approximate indexes, IVF_PQ and IVF_HNSW_SQ, at LanceDB's default search
parameters and at heavier ones. Each format is built in both dimensions.
LanceDB's full-text index ranks option A's lexical leg, set as close to the
tuned analyzer as it allows. A's ranking reads only each candidate's id, and
the text of the ten hits is read afterwards, as B reads it.

**The searches.** 300 questions of the development split, from the natural
(120), keyword (120), filtered (40) and stance (20) sets; nothing of the test
split was read. A question with a filter of its own kept it. Every other
question got one, drawn as a party, a period, one speaker, or a party and a
period together; the 300 searches carried 97 party filters, 84 period filters,
39 speaker filters and 40 of party and period, besides the 40 questions' own.
Each question's synthetic vector is drawn from a seed derived from its id, so
every store got the same searches. 30 questions of the interpellation set went
first and were not counted. The p95 is over the 300 searches, each timed from
sending the request to the whole answer.

**Cold starts.** Each container got three. Before each, the page cache of
WSL's virtual machine was dropped and a new container was created. A cold
start is the time from just before `docker start` to the first search
answered, and the budget is held against the slowest of the three.

**Memory.** The cgroup's peak, page cache included, is reported beside the
processes' own memory. A store is within the budget's memory when the peak
stayed within 2 GiB and the kernel killed nothing. Where the kernel had to take
back page cache, the peak touched the limit, and whether it stayed within came
down to a few pages; the tables mark those runs, and the pages the store had
to read again show what the limit cost.

**The query model.** A service with a local model embeds every question
itself. Its weights share the memory with the store, loading it adds to every
cold start, and embedding adds to every search. bekko-embedding-v1-a25m was
measured in four variants: with sentence-transformers and torch, in float32
and in bfloat16; and exported to ONNX and run by ONNX Runtime without torch, in
float32 and with its weights quantized to int8. Each variant was checked
against sentence-transformers in float32 on 100 questions of the workload and
one text long enough to be cut. saga-embed_v1 and
snowflake-arctic-embed-l-v2.0 are not in this machine's Hugging Face cache,
and the spike downloads nothing, so the stores of 1 024 dimensions were
measured without a local model. The API candidate's round trip was not
measured either. A store and a model of the same dimension were measured
together where each was within the budget alone.

**A reference for Cloud Run.** The repository records one cold start on Cloud
Run, the inference service's: 26 s, against 0.2 s warm
(`.github/workflows/keep-warm.yml`). Its image was rebuilt from the same base,
torch, requirements, app and command. Its weights were saved with the same
call from an earlier revision of the classifier in this machine's Hugging Face
cache, of the same architecture and size, instead of downloaded. It was started
cold here like the stores and sent the request the keep-warm workflow sends.
The image sets no thread settings, so torch sizes its pools by the CPUs it
sees, and under the limits alone it sees all 22 of WSL's virtual machine. It
was therefore also started pinned to one CPU.

## Results

### The stores alone

| Store | On disk (MiB) | Slowest cold start (s) | Hybrid p50 (ms) | Hybrid p95 (ms) | Peak (GiB) | Own memory (GiB) | Within the budget |
|---|---|---|---|---|---|---|---|
| numpy-384-float32 | 1 877 | 3.10 | 60 | 114 | 1.75 | 0.16 | yes |
| numpy-384-float16 | 1 464 | 1.94 | 86 | 324 | 1.35 | 0.16 | yes |
| numpy-384-int8 | 1 258 | 1.68 | 72 | 160 | 1.13 | 0.15 | yes |
| numpy-1024-float32 | 3 254 | 4.34 | 1 931 | 3 576 | at the limit | 0.14 | no: p95 and memory |
| numpy-1024-float16 | 2 153 | 3.01 | 146 | 924 | at the limit | 0.19 | no: p95 |
| numpy-1024-int8 | 1 602 | 2.39 | 99 | 470 | 1.54 | 0.16 | yes |
| lance-384-float32 | 1 284 | 3.20 | 303 | 691 | at the limit | 0.93 | no: p95 |
| lance-384-float16 | 871 | 2.71 | 260 | 511 | 1.49 | 0.74 | no: p95 |
| lance-1024-float32 | 2 660 | 3.91 | 1 317 | 2 640 | at the limit | 1.39 | no: p95 |
| lance-1024-float16 | 1 560 | 4.48 | | | killed | | no: killed for memory |
| lance-384-float32-ivfpq | 1 299 | 3.91 | 50 | 110 | 1.52 | 0.34 | yes |
| lance-1024-float32-ivfpq | 2 698 | 3.70 | 63 | 134 | at the limit | 0.42 | yes |
| lance-384-float32-hnswsq | 1 607 | 3.92 | 85 | 144 | 1.16 | 0.58 | yes |
| lance-1024-float32-hnswsq | 3 326 | 4.93 | 119 | 188 | at the limit | 0.92 | yes |

Every cold start was within 8 s. The lexical leg alone, the same in every
store of an option, had a p95 of 50 to 67 ms in B and 71 to 78 ms in A.

The vector leg decides. numpy-1024-float32 holds 2.15 GiB of vectors, more than
the container's 2 GiB, so the searches kept reading pages again (78 354 780
over the warm phases), and the p95 rose to 3.6 s. float16 halves the vectors,
but numpy has no matrix product in half precision, so each block is turned into
float32 first. At 1 024 dimensions the vector leg then took 416 ms on average,
and the p95 reached 924 ms. int8 is turned into float32 the same way, faster,
and held at 470 ms.

LanceDB's exact search was over the p95 in both dimensions and both formats.
Its own memory was 0.74 to 1.39 GiB, against 0.14 to 0.19 GiB in B, and
lance-1024-float16 was killed for memory after 69 of the 300 searches. Its IO
threads also ran beside the one CPU: the quota held lance-384-float32 back in
3 309 CPU periods of the warm phases, against 79 for numpy-384-float32.

### The approximate indexes

At LanceDB's defaults, IVF_PQ and IVF_HNSW_SQ held the budget in both
dimensions. The heavier settings:

| Store | Setting | Hybrid p95 (ms) | Within the budget |
|---|---|---|---|
| lance-384-float32-ivfpq | nprobes 20 | 98 | yes |
| lance-384-float32-ivfpq | nprobes 20, refine factor 10 | 595 | no |
| lance-384-float32-ivfpq | nprobes 50, refine factor 10 | 364 | yes |
| lance-1024-float32-ivfpq | nprobes 20 | 104 | yes |
| lance-1024-float32-ivfpq | nprobes 20, refine factor 10 | 969 | no |
| lance-1024-float32-ivfpq | nprobes 50, refine factor 10 | 845 | no |
| lance-384-float32-hnswsq | ef 1 500 | 117 | yes |
| lance-384-float32-hnswsq | ef 3 000 | 158 | yes |
| lance-1024-float32-hnswsq | ef 1 500 | 192 | yes |
| lance-1024-float32-hnswsq | ef 3 000 | 264 | yes |

IVF_PQ ranks by compressed vectors. Re-ranking its candidates by their full
vectors, the refine factor, took the p95 to 364 ms at 384 dimensions and over
the budget at 1 024. Of the exact search's ten nearest passages, IVF_PQ at its
defaults returned 0.096 on average at 384 dimensions and 0.087 at 1 024, and
IVF_HNSW_SQ 0.854 and 0.765. Random vectors give an index no structure to use,
so these shares say nothing about recall on real vectors. The pre-registration
accepts an index only if its recall@10 against exact search reaches 0.97, and
only the real vectors can show that.

### The query model

| Variant | On disk (MiB) | Slowest cold start (s) | Import, load (s) | Embedding p50, p95 (ms) | Own memory (GiB) | Lowest cosine to float32 | Within the budget |
|---|---|---|---|---|---|---|---|
| torch, float32 | 503 | 10.25 | 6.74, 2.11 | 27, 36 | 0.65 | 1.0 | no: cold start |
| torch, bfloat16 | 268 | 10.38 | 6.98, 2.12 | 85, 132 | 0.64 | 0.999775 | no: cold start |
| ONNX, float32 | 503 | 3.10 | 0.09, 1.74 | 18, 25 | 0.83 | 1.0 | yes |
| ONNX, int8 | 151 | 3.12 | 0.16, 1.44 | 9, 11 | 0.49 | 0.98938 | yes |

The import and load columns are those of each variant's slowest cold start.
Importing torch alone takes most of the 8 s. ONNX Runtime with the
tokenizers library imports in a tenth of a second. Both ONNX variants gave the
same tokens as sentence-transformers on all 101 texts. The int8 variant's
vectors differ from float32's, with a cosine of at least 0.98938 and 0.99507 on
average, so a service that embeds in int8 has to be evaluated with the int8
encoder.

### The stores with the query model

| Store and model | Slowest cold start (s) | Hybrid p50 (ms) | Hybrid p95 (ms) | Peak (GiB) | Own memory (GiB) | Pages read again | Within the budget |
|---|---|---|---|---|---|---|---|
| numpy-384-float32, ONNX float32 | 4.73 | 81 | 133 | at the limit | 0.92 | 123 277 | yes |
| numpy-384-float32, ONNX int8 | 4.20 | 66 | 113 | at the limit | 0.58 | 2 495 | yes |
| numpy-384-float16, ONNX float32 | 5.19 | 98 | 321 | at the limit | 0.92 | 276 | no: memory |
| numpy-384-float16, ONNX int8 | 3.54 | 92 | 328 | 1.89 | 0.58 | 0 | yes |
| numpy-384-int8, ONNX float32 | 3.95 | 91 | 173 | at the limit | 0.91 | 0 | no: memory |
| numpy-384-int8, ONNX int8 | 3.48 | 82 | 168 | 1.68 | 0.57 | 0 | yes |
| lance-384-float32-ivfpq, ONNX float32 | 5.04 | 76 | 150 | 1.93 | 1.07 | 0 | yes |
| lance-384-float32-ivfpq, ONNX int8 | 4.64 | 61 | 120 | 1.24 | 0.74 | 0 | yes |
| lance-384-float32-hnswsq, ONNX float32 | 5.59 | 94 | 153 | at the limit | 1.55 | 240 | yes |
| lance-384-float32-hnswsq, ONNX int8 | 5.05 | 82 | 129 | 1.96 | 1.21 | 0 | yes |

The variants run by torch were over the budget alone and were not paired.
Embedding a question took 7 to 9 ms on average in int8 and 16 to 19 ms in
float32, on top of the store's search.

### The reference for Cloud Run

| Inference service | CPUs seen | Threads | Cold starts (s) | Median (s) | Factor | Warm p50, p95 (ms) |
|---|---|---|---|---|---|---|
| Under the limits alone | 22 | 38 | 9.78, 12.50, 10.08 | 10.08 | 2.58 | 2 471, 4 026 |
| Pinned to one CPU | 1 | 7 | 4.56, 4.81, 4.65 | 4.65 | 5.59 | 46, 50 |

The factor is the 26 s recorded on Cloud Run divided by the median here.
Under the limits alone, torch ran 38 threads on one CPU's time, the quota held
them back in 530 periods during 20 warm predictions, and a prediction took
2.5 s at the median. The same record gives 0.2 s warm on Cloud Run, so its
instance does not run that way; pinned, the service answered in 46 ms. The
pinned factor is therefore the nearer one. By it, the slowest cold start of
every store and of every pair here would take 9 to 31 s on Cloud Run.

The factor is rough and net. It rests on one recorded figure, and when that
figure was measured is not recorded. Cloud Run gives this service startup CPU
boost, two CPUs while an instance starts and for 10 seconds after by Google's
page on CPU limits, which a container here never gets. Cloud Run also places
the instance and fetches the image, which a container here never waits for.
The service's configuration, read on 2026-11-22, names no execution
environment, so Cloud Run chose the first generation or the second by the
features the service uses. By Google's page on the two, the first has fast
cold starts and the second longer ones for some services. The factor holds for
the search service only if it runs in the same generation. The budget holds the
cold start measured here, as the pre-registration words it, so the factor only
marks a risk.

## LanceDB's full-text search against its documentation

LanceDB's documentation of full-text search says that it "ranks them with
BM25", lists Swedish among the languages it stems and the languages whose stop
words it removes, and gives the defaults: stemming, stop-word removal and
ASCII folding on, and a `max_token_length` of 40. It does not say which `k1`
and `b` its BM25 uses. `search_service/spike/probe_fts.py` checked each claim
on the corpus with LanceDB 0.41.0, inside the spike's image:

- **Stemming holds.** 19 829 words of the corpus, its 5 000 commonest and a
  random draw from the rest, were indexed one per row and searched. LanceDB's
  Swedish stemmer grouped every one exactly as Snowball's Swedish stemmer
  (PyStemmer 3.1.0) does, which the tuned BM25 uses.
- **The default folding merges Swedish words.** With `ascii_folding` at its
  default, "får" and "far", "här" and "har", "ön" and "on", and "säl" and "sal"
  each become one word. Option A's index turns folding off, and they stay apart.
- **The stop words differ.** LanceDB's Swedish list drops 109 of Snowball's 114
  stop words and keeps "ej", "icke", "ingen", "inte" and "utan". The tuned BM25
  removes no stop words, so option A's index removes none either.
- **Long words are dropped.** 3 words of the corpus are longer than 40 bytes.
  The longest, "försvarsunderrättelseinhämtningssammanhangen", is not found at
  the default limit and is found without one. Option A's index has no limit.
- **BM25 is Lucene's, with fixed parameters.** Every 28th passage, 20 140 in
  all, was indexed both ways. For five one-word queries, LanceDB's scores fit
  Lucene's BM25 with `k1` 1.2 and `b` 0.75: the idf each score implies varies
  by at most 1.6 parts in ten million across the hits, and equals Lucene's.
  LanceDB lets neither parameter be set. On 40 natural questions of the
  development split, its score for every hit matched that BM25 on all 40 when
  a repeated word counts each time it occurs. Counted once, it matched on 17.
  23 of the questions repeat a word.
- **int8 vectors cannot be searched.** A column of int8 vectors is refused for a
  float query, so option A has no int8 store.

So option A's lexical leg would not be the BM25 the evaluation measures. The
tuned BM25 has `b` 0.9 and counts a repeated word once, and LanceDB's can be
set to neither. On the spike's lexical searches, which use the real text and
the real questions, LanceDB returned 0.926 of the tuned BM25's ten passages on
average. Under A, the lexical leg's recall would have to be measured again, or
the tuned BM25 run beside LanceDB, which leaves LanceDB holding only the
vectors and the text.

## With 4 GiB

The pre-registration's addendum of 2026-11-22 gives the service 4 GiB, the
most Cloud Run allows for one vCPU. With it, the memory of a query model of
1 024 dimensions no longer chooses the embedding model before the comparison
has run. The stores of 1 024 dimensions were measured again under
`--memory=4g`, with the budget's memory at 4 GiB and everything else as above:

| Store | Slowest cold start (s) | Hybrid p50 (ms) | Hybrid p95 (ms) | Peak (GiB) | Own memory (GiB) | Within the budget |
|---|---|---|---|---|---|---|
| numpy-1024-float32 | 6.66 | 99 | 158 | 3.19 | 0.14 | yes |
| numpy-1024-float16 | 4.35 | 160 | 995 | 2.13 | 0.19 | no: p95 |
| numpy-1024-int8 | 2.51 | 105 | 493 | 1.54 | 0.16 | yes |
| lance-1024-float32 | 4.40 | 1 345 | 2 858 | at the limit | 3.27 | no: p95 and memory |
| lance-1024-float16 | 3.78 | | | killed | | no: killed for memory |
| lance-1024-float32-ivfpq | 2.86 | 62 | 116 | 2.99 | 0.39 | yes |
| lance-1024-float32-hnswsq | 4.26 | 114 | 182 | 2.05 | 0.88 | yes |

With 4 GiB, B holds the vectors of 1 024 dimensions in float32 and searches
them exactly at a p95 of 158 ms, against 3 576 ms with 2 GiB: the vectors stay
in the page cache, and no page had to be read again. Its cold start, 6.66 s at
the slowest, is the longest of any store here, and the risk on Cloud Run grows
with it. About 0.8 GiB is left beside the store's peak for a query model of
that size, and no such model was measured there. LanceDB's exact search stayed
over the budget, and lance-1024-float16 was killed for memory again. IVF_PQ
with 50 probes and a refine factor of 10 came within the budget this time, at
362 ms.

## Decision

**Option B.** The vectors are kept as a numpy matrix beside the tuned BM25
index, and a local query model runs through ONNX Runtime.

B is the simpler option, and it holds the budget. At 384 dimensions it holds in
all three formats alone, and with the query model in four of six pairs, among
them the fastest: float32 vectors with the int8 model, at a p95 of 113 ms and a
slowest cold start of 4.20 s. At 1 024 dimensions it holds with int8 vectors
under 2 GiB, at 470 ms, and with float32 vectors under the 4 GiB of the
addendum, at 158 ms; neither was measured with a model of that size beside it.
Option A holds only with an approximate index; searched exactly, its stores
were over the p95 in both dimensions and under both limits. The rule therefore
chooses B, under 2 GiB and under 4 GiB alike.

With an index, A searched as fast as B at 384 dimensions and faster at 1 024.
Nothing here shows that its index finds what exact search finds, though, and
its lexical leg is not the BM25 the evaluation measures.

## Consequences

- **Nothing in the service is quantized.** With the service's 4 GiB, B
  searches float32 vectors exactly in both dimensions, and the query model runs
  through ONNX Runtime in float32, whose vectors are sentence-transformers' own.
  The service then searches with the vectors the embedding comparison
  measures. The check of its build before the test split, which the addendum
  on the embedding candidates sets, has no quantization to account for. Under
  2 GiB, the vectors of 1 024 dimensions fitted only as int8, which returned
  0.976 of float32's ten nearest passages on the synthetic vectors. The pairs
  at 384 dimensions sat at the memory limit.
- **The query model runs through ONNX Runtime.** With torch, importing the
  library alone took 5.1 to 7.0 s, and every torch variant was over the cold
  start. ONNX in int8 would save about 9 ms a question, but its vectors move by
  a cosine of up to 0.011 and would need a check of their own.
- **A model of 1 024 dimensions is still to be measured beside its store.**
  About 0.8 GiB is left beside numpy-1024-float32's peak. The build checked
  before the test split shows whether the chosen model fits there, and the
  API candidate's round trip is measured in it as well.
- **The cold start on Cloud Run is a risk.** By the nearer factor, every store
  and every pair measured with 2 GiB would start in 9 to 31 s on Cloud Run, and
  numpy-1024-float32 with 4 GiB later still. The service is
  therefore pinged every five minutes, as the inference service is, and its
  cold start is measured from outside once it is deployed. Its deployment names
  the second generation, which runs Linux with cgroups, the conditions the
  memory here was measured under.
- **Under outcome B, only the vectors go.** If BM25 runs alone, the store is
  the BM25 index with the text, and its lexical leg had a p95 of 50 to 67 ms.
- **If LanceDB is revisited,** its settings here apply: ASCII folding and
  stop-word removal off, no limit on token length, IO threads at their default,
  and an `ef` no lower than the depth. LanceDB refuses an HNSW search for more
  rows than `ef`.

## What the measurement does not cover

- **Real vectors.** The times and the memory hold for any vectors of these
  shapes; what a search finds does not. Recall is for the embedding comparison
  and the check of the service's build.
- **The models of 1 024 dimensions.** saga-embed_v1 and
  snowflake-arctic-embed-l-v2.0 were not in the cache, and the API candidate's
  round trip was not measured. Once the two are in the cache, the spike's
  `--build` prepares them, and `--resume` measures only what a report lacks.
- **The machine.** A laptop's processor, under WSL2. Dropping the page cache of
  WSL's virtual machine does not drop Windows' cache of the virtual disk, so a
  cold read here can come from memory. Cloud Run fetches and streams the image
  instead.
- **One request at a time.** Each search was sent after the last had answered.
  Requests that arrive together share the one vCPU, and that was not measured.
- **The memory verdict at the limit.** Where the peak touched the limit,
  whether a run stayed within it came down to a few pages. The tables mark
  those runs.

## Addendum, 2026-11-27: the query model was superseded

The storage decision stands. The vectors are a numpy matrix beside the tuned
BM25 index, searched exactly, and nothing in the store is quantized.

The part of the decision about the query model does not describe the service.
The embedding comparison that the pre-registration's addenda of 2026-11-22 to
2026-11-24 set out chose `qwen/qwen3-embedding-8b`, and the service embeds each
question with it through OpenRouter, routed to Nebius in bf16 with fallbacks
switched off, at 1 024 dimensions. No query model runs in the service, and the
service holds no ONNX code. The decision's local ONNX model, its float32 query
model and the cold start and memory reasoning for it are therefore out of date.

That choice adds what the Consequences above leave out: the search depends on
two services outside the project. A key that runs out of credit or is revoked,
or a provider that is down, turns every search into BM25 alone. The service
waits up to 30 s for a question's vector and then answers from BM25, and it does
so sooner when the embedding API refuses the request or keeps failing after its
retries. Over 855 questions of
the development split, a question waited a median of 1.25 s for its vector,
5.61 s at the 95th percentile and 13.79 s at the longest. In a container with
one vCPU and 4 GiB, the service peaked at 3.154 GiB
(`data/reports/search/service_check_dev.json`).
