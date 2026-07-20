# Model lineage

Local training artifacts for the party classifier. Weights are gitignored
(`data/models/`); this file, and the config/tokenizer files tracked
alongside `party_classifier/`, are the exceptions.

## `party_classifier/`, current (v2, 2026-07)

Trained on the re-indexed, parser-fixed Riksdag corpus (see the PDF-parser
rewrite on `fix/pdf-parser`) with a **speaker-independent** train/val split:
15% of unique speakers are held out entirely, so validation measures
generalization to politicians the model never saw during training, not
recall of familiar voices.

| | |
|---|---|
| Base model | `KB/bert-base-swedish-cased` |
| Split | 827 train / 146 val speakers · 32,615 / 5,887 rows |
| Trained on | Colab, NVIDIA L4 |
| Held-out accuracy | 0.588 |
| Held-out macro-F1 | 0.595 |

Reproduce the split and the benchmark with `scripts/evaluate_model.py`
(`build_test_set` recreates this exact partition, see its docstring). Not
yet promoted to `MartinBlomqvist/maktsprak_classifier_clean`; see
[`notebooks/retrain_colab.ipynb`](../../notebooks/retrain_colab.ipynb) step 6.

`checkpoints/` holds the per-epoch state dicts from this run, so
`train_party_model_db.py --output-dir data/models/party_classifier` can
resume from the latest one.

## `legacy/v1_2025-09_row-split/`, superseded (2025-09)

The original model, trained before two things were fixed:

1. **PDF parsing**, the pre-fix parser mis-attributed a meaningful share of
   speeches (dropped `replik` replies, mis-labelled party-less speakers).
2. **Evaluation split**, validated with a plain row-based split, so the same
   speakers appeared in both train and test. That leak inflates macro-F1
   substantially (measured here at 0.775 vs. this model's honest 0.595 on
   the corrected split) because the model partly learns to recognize
   *individual politicians' phrasing*, not *party rhetoric*.

Kept for comparison, not for deployment. `MartinBlomqvist/maktsprak_classifier_clean`
on Hugging Face is currently this model.
