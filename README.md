# Maktspråk

![CI](https://github.com/MartinBlomqvistDev/MaktsprakAI/actions/workflows/ci.yml/badge.svg)
![Web CI](https://github.com/MartinBlomqvistDev/MaktsprakAI/actions/workflows/web-ci.yml/badge.svg)

**Can a machine tell which Swedish party wrote a sentence, and has the way parties argue changed since 2002?**

An end-to-end NLP system over the complete Riksdag debate record: a weekly ETL that parses the chamber's two-column PDFs, a fine-tuned KB-BERT classifier served from its own inference API, and a precomputed analytics site built on the Fightin' Words statistic.

Every headline number here is one that could have been inflated and wasn't. That is the point of the project.

| | |
|---|---|
| **Corpus** | 75 148 speeches · 2 970 protocols · 2002-2026 · rebuilt from source, verified against it |
| **Model** | KB-BERT, 8-way party classification, **0.628 acc / 0.619 macro-F1** speaker-independent |
| **Serving** | FastAPI on Cloud Run, scale-to-zero |
| **Site** | Next.js on Vercel, static precomputed JSON |
| **Tests** | 211, green, on every push |

---

## Why the numbers are lower than you'd expect

A party classifier is trivial to over-report, and this project did over-report, publicly, before catching it.

The first version scored **0.776 macro-F1** with a plain row-based train/test split. That split leaves the *same politicians* on both sides, so the model partly learns to recognise **individual phrasing** rather than **party rhetoric**. It answers "does this sound like Magdalena Andersson?", not "does this sound like Socialdemokraterna?".

Re-scored with a **speaker-independent split** (15% of politicians held out entirely, so every evaluated speech comes from someone the model has never seen), the honest figure is **0.619**. Against a 0.125 random baseline for 8 classes, and with the same corpus reaching a similar ceiling under a Random Forest, that is real signal, not noise.

The leaked number is kept in the repo, struck through, next to the honest one:
[`data/models/README.md`](data/models/README.md).

**This kept happening, and each time the fix is in the code:**

- **A class-imbalance correction applied twice.** `WeightedRandomSampler` *and* `class_weight="balanced"` were stacked, double-correcting toward the smallest classes. Aggregate metrics looked fine; it was only visible in live behaviour, where `"Stoppa invandringen!"` scored **V** highest. One mechanism now, not two, and the short-slogan collapse is gone (that input reads SD at 95.9%).
- **A benchmark that silently drifted.** `evaluate_model.py` re-derived the held-out speaker set by re-running the same seeded shuffle against a *live, growing* database. Same seed, different input: the "held-out" set stopped matching the one training actually used (821/146 speakers at eval time vs. 827/146 at training). Training now persists `val_speakers.json` and evaluation loads it.
- **An identifier that was not a key.** See [Data integrity](#data-integrity) below.

---

## Data integrity

The corpus is **derived data, reproducible from source**. Every protocol Riksdagen published is cached under `data/raw/` (~3 200 PDFs), and one offline command rebuilds the entire archive:

```bash
python scripts/rebuild_corpus.py --jobs 12     # no network, no database, deterministic
python scripts/verify_rebuild.py               # 21 checks; exits non-zero on any failure
```

Verification is three independent lines of evidence, since agreement between them is what makes the result trustworthy: internal consistency (unique ids, and facts the data must satisfy: no SD before 2010, no raw FP), content survival against the previous archive (1408/1408 text probes across 60 protocols), and a byte-for-byte re-parse against the source PDFs.

That property was bought the hard way.

**The record id was `f"{protocol_id}_{idx}"`**, where `idx` was an `enumerate()` counter over first-appearance order: a property of the *parser run*, not of the speech. The protocols number every anförande (`Anf. 60`), and the regex matched that number without ever capturing it.

So when the parser was fixed (it had been splicing the PDFs' two columns together and mis-attributing ~31% of replies), the set of extracted speeches changed by design, every subsequent index shifted, and `HD098_60` came to mean one speech before the fix and a different one after. Both got written. Ids stopped being unique, and no join, dedup or upsert could tell them apart.

The id is now the natural key: `f"{protocol_id}_{speaker_slug}_{party}"`, which depends only on the document. Testing that fix surfaced a second bug: the protocols spell one person several ways (`JAN-EMANUEL JOHANSSON` / `JAN EMANUEL JOHANSSON`, `CECILIA WIGSTRÖM I GÖTEBORG` / `... i Göteborg`), and keying on the raw string filed those as different speakers, splitting one person's record in two. Grouping on the slug merges them; party stays in the key, so a genuine clash (the same name under two parties in one protocol) still yields two records.

`scripts/rebuild_corpus.py` refuses to write if any id is duplicated, and the loader in `scripts/_corpus.py` raises rather than let a stale archive skew a published number. The regression test inserts a speech earlier in a protocol and asserts no later id moves:
[`tests/test_transform.py`](tests/test_transform.py).

---

## Architecture

```
Riksdag API ──► extract ──► transform ──► load ──► Supabase
                             (pdfplumber,          (ETL landing zone,
                              column-aware)         recent years only)
                                  │
data/raw/*.pdf ──► rebuild_corpus.py ──► speeches_full.parquet
  (~3 200 cached docs, the source of truth)      │
                                                 ├──► train_party_model_db.py ──► HF Hub
                                                 │      (Colab GPU, speaker-split)      │
                                                 │                                      ▼
                                                 │                        inference_service/ (FastAPI)
                                                 │                        Cloud Run, scale-to-zero
                                                 │                                      │
                                                 └──► build_site_data.py                │
                                                        (Fightin' Words,                │
                                                         static JSON)                   │
                                                              │                         │
                                                              ▼                         ▼
                                                        web/ (Next.js on Vercel) ──► /api/predict
```

Two deliberate decisions:

**The site never queries a database.** All analytics are precomputed to ~100 KB of static JSON, so the pages are instant and a Supabase outage cannot take the site down.

**Parquet is the source of truth, not Postgres.** The full history does not fit the free tier, and it does not need to: Supabase is a landing zone for the weekly ETL, the Parquet archive is the corpus, and `data/raw/` regenerates the Parquet.

---

## The statistics

Raw word frequency surfaces the same generic political vocabulary for every party. The distinctiveness work uses **Fightin' Words** (Monroe, Colaresi & Quinn, 2008): a weighted log-odds-ratio with an informative Dirichlet prior, which shrinks rare words toward the corpus background so nothing tops a ranking on noise alone.

The same estimator drives every lens, with a different grouping axis:

| Lens | Grouped by | Question |
|---|---|---|
| Party fingerprints | party | What vocabulary sets a party apart? |
| `top_movers` | era (early/late) | What rose and fell across the window? |
| `yearly_signatures` | year | What made each year distinct? |
| Issue frames | party × year | Has a party's crime-frame rate climbed toward another's? |
| `party_divergence_by_year` | — | Are the parties' vocabularies converging or diverging? |

Curated word lists are treated as code, with the failure modes documented at the point of use. Every stem in `ISSUE_FRAMES` was audited against the real corpus, and the comments record what was cut and why: `gäng` matched `tillgänglig` (accessibility) in ~85% of hits; `boende` matched `äldreboende` (eldercare), not housing policy; `mäns våld` scored exactly zero forever, because matching is substring-within-token and a stem with a space can never match. `frame_trajectories` now raises on a whitespace stem rather than silently scoring nothing.

---

## What was measured and abandoned

A tone/rhetoric feature (us-versus-them framing, per party, over time) was designed in full and **cut on the evidence**. It is documented because the negative result is stronger than the feature would have been.

Reading real matches killed the obvious approach. `dessa människor` (3 356 hits) and `de här människorna` (1 262) are overwhelmingly **sympathetic**:

> "Vi har all anledning att stötta de här människorna." (S)
> "De som kommer hit ska ges möjlighet att vara med och bygga landet." (V)

Counting those as othering counts compassion as hostility. In the chamber's formal register, hostility lives in the predicate, not in the noun phrase, so a word list over referring expressions measures *talking about a group of people*: which is what a welfare debate consists of.

The defensible version requires an in-group **and** an out-group marker in the same sentence. That yields near-perfect precision and **55 sentences in 75 000 speeches**: too few to chart, so it is published as an exhaustive census rather than a trend. Its distribution is the finding:

| out-group type | count |  | party | count |
|---|---|---|---|---|
| economic | 49 | | S | 26 |
| elite | 4 | | V | 14 |
| origin | **2** | | SD | 7 |

A single undifferentiated "us-vs-them" score would have ranked V highest and SD near the bottom, and been read as a machine calling V populist. The measurement that worked found something no one expected; the one that failed would have confirmed a prior. That is the whole argument for the gate.

Pattern provenance and the audit trail behind every word list: [`data/lexicons/CHANGELOG.md`](data/lexicons/CHANGELOG.md).

---

## Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12, TypeScript |
| ETL | `requests`, `pdfplumber`, `xml.etree`, `tqdm` |
| NLP / ML | `transformers` (KB-BERT), `torch`, `scikit-learn`, `numpy` |
| Storage | Parquet (zstd) as source of truth; Supabase/PostgreSQL as ETL landing zone |
| Serving | FastAPI, Docker, Google Cloud Run |
| Web | Next.js, React, Recharts, Tailwind, Vercel |
| CI | GitHub Actions: `ruff` + `pytest`, and `eslint` + `next build`, path-scoped |

---

## Getting started

```bash
git clone https://github.com/MartinBlomqvistDev/MaktsprakAI.git
cd MaktsprakAI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,app]"
cp .env.example .env                                # SUPABASE_URL, SUPABASE_KEY, ...
```

```bash
pytest tests/ -v                                    # 211 tests, no credentials needed
python scripts/rebuild_corpus.py --jobs 12          # rebuild the corpus, offline
python scripts/build_site_data.py                   # precompute the site's JSON
python -m src.maktsprak_pipeline.main               # one incremental ETL pass
cd web && npm install && npm run dev                # the site
```

The test suite runs without any credentials: the database is mocked, and nothing reaches the network.

---

## Project structure

```
src/maktsprak_pipeline/
├── config.py               # every constant and env var, single source of truth
├── model.py                # BERT inference
├── db/                     # Supabase CRUD (lazy client, no connect-at-import)
├── pipeline/
│   ├── transform.py        # column-aware PDF parsing, natural-key records
│   └── orchestrate.py      # run_etl() + run_historical_backfill()
└── nlp/
    ├── distinctiveness.py  # Fightin' Words (Monroe et al. 2008)
    ├── drift.py            # temporal drift, issue frames, JS divergence
    └── tone/               # tone kernel + dimensions
scripts/
├── rebuild_corpus.py       # offline rebuild from data/raw — the source of truth
├── train_party_model_db.py # BERT fine-tuning, speaker-independent split
├── evaluate_model.py       # honest benchmark against a frozen speaker set
├── build_site_data.py      # precompute the site's static JSON
└── validate_tone.py        # the gate: symmetry, precision, hit density
inference_service/          # FastAPI + Dockerfile, deployed to Cloud Run
web/                        # Next.js site (Vercel)
notebooks/retrain_colab.ipynb
tests/                      # 211 tests
```

---

## Contact

**Martin Blomqvist**
[LinkedIn](https://www.linkedin.com/in/martin-blomqvist) ·
[GitHub](https://github.com/martinblomqvistdev) ·
cm.blomqvist@gmail.com
