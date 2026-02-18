# MaktspråkAI

A live NLP pipeline that tracks how Sweden's eight parliamentary parties argue — in the Riksdag chamber, and on social media.

Every week it pulls new debate protocols from the Riksdag API, parses the PDFs, and feeds the extracted speeches into a fine-tuned Swedish BERT classifier. The results land in a Streamlit dashboard where you can paste any political text and watch the model decide which party wrote it.

**Live demo:** [maktsprak.streamlit.app](https://maktsprak.streamlit.app) *(Supabase free tier — give it a second on first load)*

---

## What it actually does

| Stage | What happens |
|---|---|
| **Extract** | Riksdag API → XML listing → PDF download (cached locally) |
| **Transform** | `pdfplumber` → full text → regex splits on `Anf. N Speaker (PARTY):` markers |
| **Load** | Upsert to Supabase PostgreSQL — 44 000+ speeches, 2015–present |
| **Classify** | Fine-tuned KB-BERT predicts party from raw text |
| **Visualise** | Streamlit dashboard: live prediction, rhetoric fingerprints, word clouds, historical trends |

A Windows Task Scheduler job (`run_etl.bat`) runs the ETL every week and keeps the database current.

---

## Architecture

```
Riksdag API ──────────────────────────────────────────────────────────────────────►┐
                                                                                   │
Twitter/X API ────────────────────────────────────────────────────────────────────►│
                                                                                   ▼
                                                                         pipeline/extract.py
                                                                                   │
                                                                                   ▼
                                                                         pipeline/transform.py
                                                                         (pdfplumber + regex)
                                                                                   │
                                                                                   ▼
                                                                         pipeline/load.py
                                                                                   │
                                                                                   ▼
                                                                    Supabase (PostgreSQL)
                                                                    speeches + tweets tables
                                                                                   │
                                                               ┌───────────────────┘
                                                               │
                                                               ▼
                                                    app/streamlit_app.py
                                                    ├── Party prediction (BERT)
                                                    ├── Rhetoric analysis (tone lexicon)
                                                    ├── Word clouds per party
                                                    └── Historical trend charts
```

---

## Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| ETL | `requests`, `pdfplumber`, `xml.etree`, `tqdm` |
| NLP / ML | `transformers` (KB-BERT), `torch`, `scikit-learn` |
| Database | Supabase (PostgreSQL) via `supabase-py` |
| App | `streamlit`, `plotly`, `wordcloud`, `matplotlib` |
| Logging | `loguru` — rotating 10 MB files, 30-day retention |
| CI | GitHub Actions — `ruff` lint + `pytest` on every push |

---

## Model

Fine-tuned from [KB/bert-base-swedish-cased](https://huggingface.co/KB/bert-base-swedish-cased) on ~44 000 Riksdag speeches (2015–2026) + party-leader tweets.

Training used weighted sampling to handle class imbalance, FGM adversarial training for robustness, OneCycleLR scheduling, and mixed precision (AMP). The final checkpoint lives on Hugging Face Hub:

👉 [`MartinBlomqvist/maktsprak_classifier_clean`](https://huggingface.co/MartinBlomqvist/maktsprak_classifier_clean)

---

## Getting started

```bash
git clone https://github.com/martinblomqvistdev/MaktsprakAI.git
cd MaktsprakAI

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev,app]"

cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY
# (and X_BEARER_TOKEN if you want tweet collection)
```

### Run the ETL once

```bash
python -m src.maktsprak_pipeline.main
```

### Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

### Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
MaktsprakAI/
├── src/maktsprak_pipeline/
│   ├── config.py            # All constants and env vars — single source of truth
│   ├── logger.py            # Loguru setup (10 MB rotation, 30-day retention)
│   ├── main.py              # ETL entry point
│   ├── model.py             # BERT inference
│   ├── db/
│   │   ├── client.py        # Lazy Supabase client (no connect-at-import)
│   │   ├── speeches.py      # Speech CRUD
│   │   └── tweets.py        # Tweet CRUD
│   ├── pipeline/
│   │   ├── extract.py       # Riksdag API + Twitter fetch
│   │   ├── transform.py     # PDF parsing + regex segmentation
│   │   ├── load.py          # Upsert orchestration
│   │   └── orchestrate.py   # run_etl() + run_historical_backfill()
│   └── nlp/
│       ├── cleaning.py      # clean_text + Swedish stop words
│       └── lexicon.py       # Vectorised tone-lexicon scoring
├── app/
│   └── streamlit_app.py     # Dashboard (811 lines, 5 pages)
├── scripts/
│   ├── backfill_speeches.py       # One-time historical backfill
│   ├── create_historic_database.py  # Dump speeches → Parquet snapshot
│   └── train_party_model_db.py    # Full BERT fine-tuning script
├── tests/                   # pytest suite (config, NLP, transform, DB mocks)
├── .github/workflows/ci.yml # Ruff + pytest on every push
├── pyproject.toml           # Package config + tool settings
├── .env.example             # All required environment variables with comments
└── run_etl.bat              # Windows Task Scheduler script
```

---

## Scheduled ETL (Windows)

`run_etl.bat` is set up as a Windows Task Scheduler job — it activates the venv, runs the incremental ETL, and logs output to `logs/`.  The script uses `%~dp0` for a portable path so it works on any machine, not just the original dev box.

To backfill a historical date range:

```bash
python scripts/backfill_speeches.py 2024-01-01
```

---

## Contact

**Martin Blomqvist**
- LinkedIn: [martin-blomqvist](https://www.linkedin.com/in/martin-blomqvist)
- GitHub: [martinblomqvistdev](https://github.com/martinblomqvistdev)
- Email: cm.blomqvist@gmail.com
