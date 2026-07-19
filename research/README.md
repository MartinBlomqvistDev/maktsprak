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
across all ten of Nordan's providers, 8 policy areas, 3 samples each, 669 texts.

**Result.** Every model over-produces the centre-right register (Moderaterna +
Liberalerna). None writes more like V, C or KD than baseline. The newest lean
hardest (Opus 4.8 puts 51% of its mass on M against a 21% baseline). Google's
Gemini models are the exception, leaning Liberalerna rather than Moderaterna.
The lean disappears in the neutral-prose control, so it sits in the models'
picture of parliamentary speech, not in the models in general.

The classifier is not neutral, and the study accounts for that:

- **Calibration.** The classifier trained on 2015-2026; 2002-2014 speeches were
  never seen. Those go through first, to confirm it recovers a known party
  (48.8% argmax, chance 12.5%) before it is pointed at model output.
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
from 2002-2014 (outside KB-BERT's training window, so unseen by it), balanced
40 per party, same truncated input for everyone, real per-token cost measured.

**Result.** The three frontier flagships win on accuracy (Gemini 0.556, GPT-5.5
0.541, Opus 0.534) against the small model's 0.328 on identical input, 0.447 at
its full context window, 0.628 on its own 2015-2026 domain. But they cost $2-3
per 1000 classifications and an API round-trip; KB-BERT costs nothing, runs 320
classifications in 44 seconds locally, and no text leaves the machine. The two
fairness caveats point opposite ways: the test era is the small model's worst
case, and the LLMs may have met riksdag text in pre-training. Published with the
result unspun: the small model loses the accuracy test, and the trade-off is the
finding.

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
