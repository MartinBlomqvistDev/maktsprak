# %% [markdown]
# # Benchmark: a fine-tuned 110M classifier against frontier LLMs
#
# The /llm study used the LLMs as writers and KB-BERT as the instrument. This
# turns it around: the LLMs are the classifiers now, and the question is the
# blunt one a hiring manager asks about any fine-tuned model, "why not just call
# GPT?"
#
# Same task for everyone: read a real riksdag speech, name the party. The test
# speeches are from 2002-2014, outside KB-BERT's 2015-2026 training window, so
# they are unseen by the small model. (Frontier models may have met riksdag text
# in pre-training; that asymmetry favours them and is stated in the write-up.)
#
# Reuses the loaded classifier and OpenRouter plumbing from the language study.

# %%
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

from llm_language_profile import (
    CORPUS_PATH,
    OPENROUTER_KEY,
    OPENROUTER_URL,
    PARTIES,
    _tokenizer,  # KB-BERT tokenizer, for the shared 512-token window
    classify,  # KB-BERT, in-process
)

OUT_DIR = Path("research/out")
CACHE = OUT_DIR / "benchmark_preds.jsonl"

# A representative spread, not all 14: two frontier flagships, then cheaper
# models across providers, so the accuracy/cost frontier is visible.
BENCH_MODELS: list[str] = [
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.3",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.6-plus",
]

N_PER_PARTY = 40
SEED = 13


def _kb_window(text: str) -> str:
    """The exact text KB-BERT sees: the speech truncated to its 512-token window,
    decoded back to words. Every model (KB-BERT and the LLMs) is fed this, so no
    model gets more of the speech than the 110M model can physically read. Fair
    same-evidence comparison rather than handing the LLMs extra context."""
    ids = _tokenizer(text, truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
    return _tokenizer.decode(ids, skip_special_tokens=True)


# The frozen speaker-independent split from training (146 held-out speakers,
# next to the model). Testing on their 2015-2026 speeches is the fair test:
# same era KB-BERT trained on, speakers it never saw. 2002-2014 confounds the
# comparison with era drift and is not used.
_VAL_PATH = Path("data/models/party_classifier/val_speakers.json")
VAL_SPEAKERS: set[str] = set(json.loads(_VAL_PATH.read_text(encoding="utf-8")))
PARTY_NAMES = {
    "C": "Centerpartiet",
    "KD": "Kristdemokraterna",
    "L": "Liberalerna",
    "M": "Moderaterna",
    "MP": "Miljöpartiet",
    "S": "Socialdemokraterna",
    "SD": "Sverigedemokraterna",
    "V": "Vänsterpartiet",
}


# %%
def sample_speeches(n_per_party: int = N_PER_PARTY, seed: int = SEED) -> pd.DataFrame:
    """Balanced sample of the held-out speaker-independent test set.

    Speeches from 2015-2026 by the 146 speakers held out of training. Unseen by
    KB-BERT (no speaker leak) yet in its own era (no era drift), which is the
    only fair way to compare it against the LLMs.
    """
    df = pd.read_parquet(CORPUS_PATH, columns=["protocol_date", "speaker", "party", "text"])
    df = df[df["speaker"].isin(VAL_SPEAKERS)]
    df = df[pd.to_datetime(df["protocol_date"]).dt.year >= 2015]
    df = df[df["party"].isin(PARTIES)]
    df = df[df["text"].str.len() > 200]  # skip procedural fragments
    parts = [g.sample(min(n_per_party, len(g)), random_state=seed) for _, g in df.groupby("party")]
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["text"] = out["text"].apply(_kb_window)
    return out


# %%
def classify_prompt(text: str) -> str:
    return (
        "Nedan är ett anförande från Sveriges riksdag. Vilket parti höll det "
        "troligen? Svara med enbart partiets förkortning, en av: "
        "C, KD, L, M, MP, S, SD, V.\n\n"
        f"Anförande:\n{text}\n\nParti:"
    )


# Longest codes first so "SD"/"KD"/"MP" win over "S"/"M"/"MP" substrings.
_ORDERED = sorted(PARTIES, key=len, reverse=True)


def parse_party(resp: str | None) -> str | None:
    if not resp:
        return None
    up = resp.upper()
    for code in _ORDERED:
        if code in up:
            return code
    for code, name in PARTY_NAMES.items():
        if name.upper() in up:
            return code
    return None


def llm_classify(model: str, text: str) -> tuple[str | None, int, int]:
    """Return (predicted party or None, prompt_tokens, completion_tokens)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": classify_prompt(text)}],
        "temperature": 0,
        # Enough headroom for a reasoning-mandatory model (Gemini) to emit its
        # answer after thinking; non-reasoning models still stop after the code.
        "max_tokens": 200,
        "reasoning": {"enabled": False},
    }
    r = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json=body,
        timeout=120,
    )
    if r.status_code == 400 and "reasoning" in r.text.lower():
        body.pop("reasoning")
        r = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json=body,
            timeout=120,
        )
    r.raise_for_status()
    j = r.json()
    if "choices" not in j:
        raise RuntimeError(f"no choices: {j.get('error', j)}")
    usage = j.get("usage", {})
    text_out = j["choices"][0]["message"]["content"]
    return parse_party(text_out), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


# %%
def pricing(models: list[str]) -> dict[str, tuple[float, float]]:
    """(prompt $/token, completion $/token) per model, live from OpenRouter."""
    data = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        timeout=30,
    ).json()["data"]
    by_id = {m["id"]: m for m in data}
    out = {}
    for m in models:
        p = by_id[m]["pricing"]
        out[m] = (float(p["prompt"]), float(p["completion"]))
    return out


def _load_cache() -> dict[tuple, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["model"], r["idx"])] = r
    return out


# %%
def run() -> None:
    speeches = sample_speeches()
    print(f"{len(speeches)} unseen speeches (<=2014), {N_PER_PARTY}/party.")

    # KB-BERT: local, free, argmax.
    t0 = time.time()
    kb_pred = [max(classify(t).items(), key=lambda kv: kv[1])[0] for t in speeches["text"]]
    kb_secs = time.time() - t0
    print(f"KB-BERT done in {kb_secs:.1f}s.")

    price = pricing(BENCH_MODELS)
    cache = _load_cache()
    rows = []

    with CACHE.open("a", encoding="utf-8") as fh:
        for model in BENCH_MODELS:
            in_tok = out_tok = 0
            for idx, text in enumerate(speeches["text"]):
                key = (model, idx)
                if key in cache:
                    r = cache[key]
                else:
                    try:
                        pred, pt, ct = llm_classify(model, text)
                    except Exception as exc:
                        print(f"  {model} [{idx}] FAILED {exc}")
                        continue
                    r = {"model": model, "idx": idx, "pred": pred, "pt": pt, "ct": ct}
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fh.flush()
                    cache[key] = r
                    time.sleep(0.3)
                in_tok += r["pt"]
                out_tok += r["ct"]
            cost = in_tok * price[model][0] + out_tok * price[model][1]
            rows.append({"model": model, "in_tok": in_tok, "out_tok": out_tok, "cost": cost})
            print(f"  {model}: {in_tok} in, {out_tok} out, ${cost:.3f}")

    # Assemble predictions aligned to the sample and score everyone.
    gold = speeches["party"].tolist()
    preds = {"KB-BERT (110M, lokal)": kb_pred}
    for model in BENCH_MODELS:
        col = [None] * len(speeches)
        for (m, idx), r in cache.items():
            if m == model:
                col[idx] = r["pred"]
        preds[model] = col

    cost_by_model = {r["model"]: r["cost"] for r in rows}
    n = len(speeches)
    results = []
    for name, pred in preds.items():
        scored = [(g, p) for g, p in zip(gold, pred) if p is not None]
        acc = sum(g == p for g, p in scored) / len(scored) if scored else 0.0
        macro_f1 = _macro_f1(scored)
        parseable = len(scored) / n
        if name.startswith("KB-BERT"):
            cost_per_1000 = 0.0
        else:
            cost_per_1000 = cost_by_model[name] / n * 1000
        results.append(
            {
                "model": name,
                "n": len(scored),
                "accuracy": round(acc, 3),
                "macro_f1": round(macro_f1, 3),
                "parseable": round(parseable, 3),
                "cost_per_1000_usd": round(cost_per_1000, 3),
            }
        )

    res = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    res.to_csv(OUT_DIR / "benchmark_results.csv", index=False)
    print("\n" + res.to_string(index=False))
    print(f"\nKB-BERT wall time {kb_secs:.1f}s for {n} speeches (local, no API).")


def _macro_f1(scored: list[tuple[str, str]]) -> float:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for g, p in scored:
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1s = []
    for c in PARTIES:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


# %%
if __name__ == "__main__":
    run()
