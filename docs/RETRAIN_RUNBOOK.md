# Re-index + retrain runbook

End-to-end procedure for rebuilding the `speeches` table with the fixed PDF
parser and retraining the party classifier. Everything here lives on the
`fix/pdf-parser` branch.

## Background

The PDF parser was rewritten to fix three bugs (see the commit
`fix: rewrite PDF parser …`):

1. two-column interleaving + running-header bleed,
2. party-less speakers (`(-)`, `TALMANNEN:`, unlisted ministers) swallowing the
   next speech,
3. `replik` reply headers (~31 % of all speeches) being mis-attributed.

Rows ingested before the fix are therefore **incomplete and mislabelled**. The
table must be re-parsed before retraining, or the model just relearns the old
errors.

---

## Phase 1, Re-index the DB (local, free)

Runs on your machine against the cached PDFs in `data/raw/`. No GPU. Expect
**~2-4 h** (the parser is heavier than before; a big debate protocol takes
~13 s). The app reads Supabase directly, so it degrades during the run, fine
when there's no traffic. Each protocol is replaced atomically, so a crash just
means "resume by re-running"; nothing is left corrupt.

```bash
# 0. (already done once) A full-column backup lands in data/parquet/
#    speeches_backup_<timestamp>.parquet. Re-run any time:
.venv/Scripts/python -c "from scripts.reindex_speeches import backup_speeches; backup_speeches()"

# 1. Sanity dry-run, parse, write nothing:
.venv/Scripts/python scripts/reindex_speeches.py --dry-run --limit 5

# 2. Single-protocol live test, proves the delete+insert path:
.venv/Scripts/python scripts/reindex_speeches.py --limit 1

# 3. Full re-index from the start of the dataset (backs up first):
.venv/Scripts/python scripts/reindex_speeches.py --from-date 2014-01-01
```

Verify afterwards: row count rose (recovered reply speeches), and speaker names
are clean (no body text, no `(-)` independents mislabelled).

### Rollback

The pre-run Parquet backup has every column. To restore, load it and upsert
back, or keep it as the record of the old state.

---

## Phase 2, Retrain (Colab GPU, ~$1-2, ~1 h)

Training reads the (now clean) `speeches` table straight from Supabase, so the
Colab VM only needs the repo + credentials.

**Prerequisite:** push the branch so Colab can clone it.

```bash
git push -u origin fix/pdf-parser
```

Then open `notebooks/retrain_colab.ipynb`:

- VS Code: *Select Kernel → Colab → Auto Connect* (Colab VS Code extension), or
  upload the notebook to colab.research.google.com.
- Add Secrets (key icon at colab.research.google.com, notebook access on):
  `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY`, `HF_TOKEN`.
  If secrets are awkward, replace cell 4 with a `getpass` paste:
  ```python
  import os, getpass
  for k in ('SUPABASE_URL','SUPABASE_KEY','SUPABASE_SERVICE_KEY','HF_TOKEN'):
      os.environ[k] = getpass.getpass(k + ': ')
  ```
- Run the cells top to bottom.

### Cost control

Cell 5a runs the **cheap pass** first: `--no-fgm --max-length 256` (drops
adversarial training, halves sequence length). Get a number from that before
paying for the full run (cell 5b). Levers:

| Flag | Effect |
| --- | --- |
| `--no-fgm` | skips FGM adversarial step (~2× faster) |
| `--max-length 256` | ~2× faster, small accuracy cost |
| `--epochs N` | cap epochs (early stopping also applies) |
| `--batch-size N` | rescales OneCycleLR `max_lr` automatically |

An L4/A10/4090 is plenty for a 110 M-param BERT; you do not need an A100.

---

## Phase 3, Benchmark + promote

Cell 6 scores the **new checkpoint vs the current live model** on one fixed,
clean held-out set (accuracy, macro-F1, per-class report, confusion matrices in
`results/`). Because both models see identical inputs, the delta is honest.

Cell 7 pushes the new model to a **separate** repo
(`maktsprak_classifier_reindexed`) so the live app keeps running the old one
during A/B. Promote to `MartinBlomqvist/maktsprak_classifier_clean` (the app's
default `MODEL_NAME_OR_PATH`) only once the new model wins:

```python
AutoModelForSequenceClassification.from_pretrained(_dir).push_to_hub(
    "MartinBlomqvist/maktsprak_classifier_clean")
AutoTokenizer.from_pretrained(_dir).push_to_hub(
    "MartinBlomqvist/maktsprak_classifier_clean")
```

The Cloud Run inference service picks it up on its next model load / redeploy.

---

## Phase 4, Refresh the app's Parquet (optional)

`create_historic_database.py` dumps Supabase → Parquet for `PARQUET_URL`. That
path (`fetch_combined_speeches`) is currently **unused** by the app, so this is
only needed if you wire it back in. The live app reads Supabase directly and
already reflects the re-indexed data.
