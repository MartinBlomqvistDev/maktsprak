# research/ : which party do LLMs write like?

A small study that reuses the Maktsprak party classifier as a measurement tool.
It is a companion to Nordan AI's
[Which Swedish party do LLMs vote for?](https://www.nordan.ai/research/which-swedish-party-do-llms-vote-for).

Nordan had models answer SVT's valkompass and mapped the answers to parties,
which measures stated preference. This measures something else. Prompt a model to
write a riksdag speech with no party named, run the text through the classifier,
and see whose language it produced. Stated preference and writing style are not
the same axis, so the cases where they disagree are the interesting ones.

## What the controls are for

The classifier is not neutral, and the study accounts for that rather than
hiding it:

- **Calibration.** The classifier trained on 2015-2026, so 2002-2014 speeches
  were never seen. Those real speeches go through the pipeline first, to show it
  recovers a known party before it is pointed at model output.
- **Reference distribution.** The classifier leans toward the larger training
  classes. Feeding it a balanced set of real speech and averaging the output
  gives that lean, and every model is then reported as a deviation from it.
- **Neutral baseline.** Deliberately non-partisan text is generated and
  classified too, as a second reference point.

The main caveat, stated in the notebook: writing like a party is not endorsing
it. Writing style is not ideology.

## Running it

Two providers, picked automatically from whichever key is in `.env`:

```bash
# Easy path: a free Google AI Studio key (no card), runs across the Gemini family
echo "GEMINI_API_KEY=..." >> .env          # from aistudio.google.com/apikey

# Full path: one OpenRouter key reaches every provider (a few dollars)
echo "OPENROUTER_API_KEY=sk-or-..." >> .env

# runs as a script, or open as a notebook (it is cell-marked)
python research/llm_language_profile.py
```

Generation is cached to `research/out/generations.jsonl`, so a second run only
re-classifies and does not pay the API again. Outputs stay gitignored until you
publish them.

No database is touched: the classifier loads from local weights, calibration
reads the local parquet, and only generation goes over the network. Check the
model ids against <https://openrouter.ai/models> before a paid run, since they
change.
