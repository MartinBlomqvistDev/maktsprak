# research/

Two studies that reuse the Maktspråk party classifier as a measurement
instrument. Both are companions to Nordan AI's
[Which Swedish party do LLMs vote for?](https://www.nordan.ai/research/which-swedish-party-do-llms-vote-for),
which measured stated preference by having models answer SVT's valkompass.
These measure something else, and the classifier does the reading.

- `llm_language_profile.py` : which party do LLMs *write* like
- `llm_classify_benchmark.py` : is a fine-tuned 110M model worth it against a
  frontier LLM

Neither touches the database. The classifier loads from local weights,
calibration and sampling read the local parquet, and only the model calls go
over the network.

## 1. Which party do LLMs write like?

Live: <https://maktsprak.se/llm> · Dataset:
<https://huggingface.co/datasets/MartinBlomqvist/maktsprak-llm-language-profile>

Prompt a model to write a riksdag speech with no party named, run the text
through the classifier, and see whose language it produced. 14 frontier models
across all ten of Nordan's providers, 8 policy areas, 3 samples each: 336
speeches, and 336 neutral-prose controls alongside them, 672 generations in all.

**Result.** 12 of the 14 write more like Moderaterna than the baseline does.
Claude Opus 4.8 leans hardest, putting 63% of its mass on M against a 21%
baseline. The two exceptions are Qwen 3.6 Plus and GLM 5.2, and neither leans
centre-left: both put their excess on Sverigedemokraterna. Not one model writes
more like Vänsterpartiet or Centerpartiet than baseline, though three sit above
it on Kristdemokraterna. The lean disappears in the neutral-prose control, so it
sits in the models' picture of parliamentary speech, not in the models in
general.

The classifier is not neutral, and the study accounts for that:

- **Calibration.** The classifier trained on 2015-2026; 2002-2014 speeches were
  never seen. Those go through first, to confirm it recovers a known party
  (48.8% argmax, chance 12.5%) before it is pointed at model output. The script
  reports two calibrations, `in_era` and `out_of_era`; this is the out-of-era one,
  and it is the lower of the two by design. The main README quotes the in-era,
  speaker-held-out figure. The gap between them is era drift, not instrument
  error.
- **Baseline.** Its mean output on party-balanced real speech (M 0.21, L 0.07,
  ...) is subtracted; every profile is a deviation from that baseline.
- **Neutral control.** The same models write non-partisan civil-servant prose,
  as a second reference point.

The main caveat, stated in the notebook: writing like a party is not endorsing
it. Writing style is not ideology.

## 2. Is a fine-tuned model worth it against a frontier LLM?

Live: <https://maktsprak.se/riktmarke>

The reverse question. The LLMs are the classifiers now, on the same task the
110M model was fine-tuned for: read a real speech, name the party. 320 speeches
from the classifier's own speaker-independent held-out set (2015-2026, by
speakers held out of training, verified clean at 0.591 held-out against 0.995 on
training speakers). Every model reads the same input, KB-BERT's own 512-token
window, so no LLM gets more of the speech than the small model can physically
read. Real per-token cost measured.

**Result** (`out/benchmark_results.csv`). All six frontier models beat the
fine-tune. GPT-5.5 leads at 0.762 accuracy, down to Qwen 3.6 Plus at 0.644,
against KB-BERT's 0.591 (0.606 macro-F1). But they cost $0.25 to $5.62 per 1000
classifications and an API round-trip; KB-BERT costs nothing, runs locally, and
no text leaves the machine. Published with the result unspun: the small model
comes last on accuracy, and the cost and latency trade-off is the finding. Not a
privacy argument, since riksdagen's proceedings are public record.

**Two robustness conditions, both kept.**

- `out/benchmark_results_2000char.csv`: a shared 2000-character window instead of
  512 tokens. Everything shifts down together (GPT-5.5 0.709, KB-BERT 0.559) and
  the ordering holds.
- `out/benchmark_results_outofera.csv`: the same task on 2002-2014 speeches,
  outside KB-BERT's training window entirely. Every model drops and the small one
  drops hardest, to 0.328 against Gemini's 0.556. That era is its worst case, and
  the gap there is era drift as much as skill.

The two fairness caveats point opposite ways: the out-of-era condition is the
small model's worst case, and the LLMs may have met riksdag text in pre-training.

## Running them

One OpenRouter key reaches every provider:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." >> .env      # from openrouter.ai/keys

python research/llm_language_profile.py          # study 1 (runs as a script or notebook)
python research/llm_classify_benchmark.py        # study 2
```

Both cache to `research/out/` (generations and predictions keyed so a rerun
skips whatever is already there, and never pays the API twice). The language
study also auto-detects a free Google AI Studio key (`GEMINI_API_KEY` from
aistudio.google.com/apikey) and runs across the Gemini family only, if no
OpenRouter key is set.

Model ids change fast: the current runs used 2026-07 frontier ids, verified live
against <https://openrouter.ai/models>. Gemini makes reasoning mandatory and
400s if it is disabled, so the request layer drops that flag and retries. Check
the ids before any paid rerun.
