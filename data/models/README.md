# Model lineage

Local training artifacts for the party classifier. Weights are gitignored
(`data/models/`); this file, and the config/tokenizer files tracked alongside
`party_classifier/`, are the exceptions.

## `party_classifier/`, current — deployed as `maktsprak_classifier_clean`

The live model behind maktsprak.se. Fine-tuned KB-BERT on the re-indexed,
parser-fixed Riksdag corpus (2015 onward) with a **speaker-independent**
train/val split: 15% of unique speakers (146 of them) are held out entirely, so
validation measures generalization to politicians the model never saw during
training, not recall of familiar voices. The held-out set is persisted in
`val_speakers.json` next to the weights, so nothing re-derives the split.

| | |
|---|---|
| Base model | `KB/bert-base-swedish-cased` |
| Split | speaker-independent, 146 speakers (15%) held out |
| Trained on | Colab, NVIDIA L4 |
| Held-out accuracy | 0.628 |
| Held-out macro-F1 | 0.619 |

Class imbalance is corrected once (loss weighting), not twice. An earlier
version also oversampled rare parties; correcting twice made short slogans
collapse to the smallest class ("Stoppa invandringen!" read Vänsterpartiet).
With one mechanism that input reads SD at 95.9%, which is the sanity check to
run after any retrain.

Evaluate with `scripts/evaluate_model.py`, loading `val_speakers.json` so the
split is exactly the one the model trained against (see
`docs/RETRAIN_RUNBOOK.md`). The Cloud Run inference service bakes these weights
into its image.

## `legacy/v1_2025-09_row-split/`, superseded (2025-09)

The original model, trained before two things were fixed:

1. **PDF parsing**: the pre-fix parser mis-attributed a meaningful share of
   speeches (dropped `replik` replies, mis-labelled party-less speakers).
2. **Evaluation split**: validated with a plain row-based split, so the same
   speakers appeared in both train and test. That leak inflates macro-F1
   substantially (0.775 on the row split vs. the honest 0.619 on the
   speaker-independent split) because the model partly learns to recognize
   *individual politicians' phrasing*, not *party rhetoric*.

Kept for comparison, not for deployment.
