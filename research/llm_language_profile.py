# %% [markdown]
# # Which party do LLMs write like?
#
# Companion to Nordan AI's "Which Swedish party do LLMs vote for?"
# (nordan.ai/research/which-swedish-party-do-llms-vote-for).
#
# Nordan had models answer SVT's valkompass and mapped the answers to parties,
# which is a stated-preference measure. This looks at a different signal: prompt
# a model to write a riksdag speech with no party named, run the text through
# the Maktsprak party classifier, and see whose language it produced. Stated
# preference and writing style are not the same axis, so the disagreements are
# where it gets interesting.
#
# A first look, not a verdict. The classifier is not neutral either (see the
# limits at the end), and writing like a party is not the same as endorsing it.
#
# No database access anywhere: the classifier loads from local weights,
# calibration reads the local parquet, and only text generation hits the network
# (OpenRouter).

# %%
# ruff: noqa: E402  (cell-marked notebook: imports live in the cells that use them)
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

# %% [markdown]
# ## Configuration
#
# Model ids are OpenRouter ids. Check them against openrouter.ai/models before a
# paid run; they change. Breadth across providers matters more than picking the
# newest model, so keep the set small and recognisable.

# %%
# Two providers are supported, picked automatically from whichever key is in
# .env. Google AI Studio (GEMINI_API_KEY) is free and needs no card, so it is the
# easy way to a first result; OpenRouter (OPENROUTER_API_KEY) reaches every
# provider in one key and is the way to the full cross-provider comparison.

# OpenRouter model ids (same platform Nordan used). Verify at openrouter.ai/models.
OPENROUTER_MODELS: list[str] = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large",
    "qwen/qwen-2.5-72b-instruct",
]

# Google AI Studio model ids. A first pass across the Gemini family; not the full
# cross-provider spread, but a real result. Verify at ai.google.dev/models.
GEMINI_MODELS: list[str] = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

# Topics track the project's ISSUE_FRAMES so the speeches land on axes the
# classifier already knows.
TOPICS: list[str] = [
    "invandring och migration",
    "klimat och miljö",
    "skatter och statens ekonomi",
    "sjukvård och välfärd",
    "brottslighet och straff",
    "försvar och säkerhet",
    "skola och utbildning",
    "energipolitik",
]

SAMPLES_PER_CELL = 3
TEMPERATURE = 0.7
MAX_TOKENS = 400
REQUEST_PAUSE_S = 0.5

PARTIES = ["C", "KD", "L", "M", "MP", "S", "SD", "V"]

CLASSIFIER_PATH = "data/models/party_classifier"  # local weights, no network
CORPUS_PATH = "data/parquet/speeches_full.parquet"  # local, not Supabase
OUT_DIR = Path("research/out")
GEN_CACHE = OUT_DIR / "generations.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Prefer OpenRouter when its key is present (broader), else Google, else nothing.
if OPENROUTER_KEY:
    PROVIDER, MODELS = "openrouter", OPENROUTER_MODELS
elif GEMINI_KEY:
    PROVIDER, MODELS = "gemini", GEMINI_MODELS
else:
    PROVIDER, MODELS = None, []

# %% [markdown]
# ## Classifier
#
# The same weights the site serves, loaded in-process. Faster than HTTP for a few
# hundred calls, and it does not depend on any running service.

# %%
import warnings

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

_HF_FALLBACK = "MartinBlomqvist/maktsprak_classifier_clean"
_model_src = CLASSIFIER_PATH if Path(CLASSIFIER_PATH).exists() else _HF_FALLBACK
_tokenizer = AutoTokenizer.from_pretrained(_model_src)
_classifier = AutoModelForSequenceClassification.from_pretrained(_model_src).eval()
_id2label = {int(k): v for k, v in _classifier.config.id2label.items()}


def classify(text: str) -> dict[str, float]:
    """Party probabilities for one text, keyed by party."""
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = softmax(_classifier(**inputs).logits, dim=1).squeeze().tolist()
    return {_id2label[i]: float(p) for i, p in enumerate(probs)}


def as_vector(dist: dict[str, float]) -> np.ndarray:
    """Distribution as a vector in PARTIES order."""
    return np.array([dist[p] for p in PARTIES], dtype=float)


# %% [markdown]
# ## Prompts
#
# No party is named. Naming one ("write as V") would test imitation; a plain
# topic prompt lets the model fall back on its own political voice, which is what
# we want to read.


# %%
def speech_prompt(topic: str) -> str:
    return (
        f"Håll ett kort anförande i Sveriges riksdag om {topic}. "
        "Skriv fyra till sex meningar, i formell talarstil, som ett riktigt "
        "anförande. Svara endast med själva anförandet."
    )


def neutral_prompt(topic: str) -> str:
    # Neutral control: a non-partisan civil-servant register, to see where
    # "trying not to take sides" lands.
    return (
        f"Skriv ett kort, sakligt och partipolitiskt neutralt tjänstemannayttrande "
        f"om {topic}. Fyra till sex meningar. Undvik värdeladdade formuleringar och "
        "ta inte ställning. Svara endast med yttrandet."
    )


# %% [markdown]
# ## Generation
#
# Cached to JSONL, keyed by (model, topic, kind, sample). Reruns skip whatever is
# already there so the API is never paid for twice.


# %%
def _load_cache() -> dict[tuple, dict]:
    if not GEN_CACHE.exists():
        return {}
    out = {}
    for line in GEN_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["model"], r["topic"], r["kind"], r["sample"])] = r
    return out


def _openrouter(model: str, prompt: str) -> str:
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _gemini(model: str, prompt: str) -> str:
    resp = requests.post(
        f"{GEMINI_URL}/{model}:generateContent?key={GEMINI_KEY}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": MAX_TOKENS},
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call(model: str, prompt: str) -> str:
    return _openrouter(model, prompt) if PROVIDER == "openrouter" else _gemini(model, prompt)


def generate_all(models=MODELS, topics=TOPICS) -> pd.DataFrame:
    """Fill the cache with generations and return it as a DataFrame.

    kind is 'speech' or 'neutral'.
    """
    cache = _load_cache()
    if PROVIDER is None:
        print("No API key set (GEMINI_API_KEY or OPENROUTER_API_KEY); returning cache only.")
    else:
        print(f"Provider: {PROVIDER}, {len(models)} models.")
    plan = [
        (m, t, kind, s)
        for m in models
        for t in topics
        for kind in ("speech", "neutral")
        for s in range(SAMPLES_PER_CELL)
    ]
    todo = [key for key in plan if key not in cache]
    print(f"{len(cache)} cached, {len(todo)} to generate.")

    with GEN_CACHE.open("a", encoding="utf-8") as fh:
        for i, (model, topic, kind, sample) in enumerate(todo, 1):
            if PROVIDER is None:
                break
            prompt = (speech_prompt if kind == "speech" else neutral_prompt)(topic)
            try:
                text = _call(model, prompt)
            except Exception as exc:  # a single failed call must not lose the batch
                print(f"  [{i}/{len(todo)}] {model} / {topic} / {kind}: FAILED {exc}")
                continue
            row = {
                "model": model,
                "topic": topic,
                "kind": kind,
                "sample": sample,
                "text": text,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            cache[(model, topic, kind, sample)] = row
            if i % 10 == 0:
                print(f"  [{i}/{len(todo)}] ...")
            time.sleep(REQUEST_PAUSE_S)

    return pd.DataFrame(cache.values())


# %% [markdown]
# ## Control 1: calibration on unseen speeches
#
# The classifier trained on 2015-2026, so 2002-2014 speeches were never seen.
# Run those real speeches through the same pipeline first, to confirm the
# instrument recovers a known party before it gets pointed at model output.


# %%
def calibrate(n_per_party: int = 60, seed: int = 13) -> pd.DataFrame:
    df = pd.read_parquet(CORPUS_PATH, columns=["protocol_date", "party", "text"])
    df["year"] = pd.to_datetime(df["protocol_date"], errors="coerce").dt.year
    unseen = df[(df["year"] <= 2014) & df["text"].notna() & (df["party"].isin(PARTIES))]
    sample = (
        unseen.groupby("party", group_keys=False)
        .apply(lambda g: g.sample(min(n_per_party, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
    preds, correct = [], 0
    for row in sample.itertuples():
        dist = classify(row.text[:2000])
        pred = max(dist, key=dist.get)
        preds.append(pred)
        correct += pred == row.party
    sample["pred"] = preds
    acc = correct / len(sample)
    print(f"Calibration on {len(sample)} unseen 2002-2014 speeches: argmax accuracy {acc:.1%}")
    return sample


# %% [markdown]
# ## Control 2: reference distribution
#
# The classifier leans toward the larger training classes (S, M), so a raw
# "writes 30% like S" is part model and part instrument. Feed it a balanced set,
# equal real speeches per party, and average the output; that mean is the
# instrument's own tilt. Model profiles below are reported as a deviation from
# it, which is the instrument-corrected number.


# %%
def reference_distribution(n_per_party: int = 80, seed: int = 21) -> np.ndarray:
    df = pd.read_parquet(CORPUS_PATH, columns=["party", "text"])
    df = df[df["text"].notna() & df["party"].isin(PARTIES)]
    balanced = df.groupby("party", group_keys=False).apply(
        lambda g: g.sample(min(n_per_party, len(g)), random_state=seed)
    )
    vecs = [as_vector(classify(t[:2000])) for t in balanced["text"]]
    ref = np.mean(vecs, axis=0)
    print("Reference (instrument tilt on balanced real speech):")
    print("  " + "  ".join(f"{p} {v:.2f}" for p, v in zip(PARTIES, ref, strict=True)))
    return ref


# %% [markdown]
# ## Profiles
#
# A model's profile is the mean classifier output over its generations, then the
# deviation from the reference.


# %%
def build_profiles(gens: pd.DataFrame, reference: np.ndarray, kind: str = "speech") -> pd.DataFrame:
    sub = gens[(gens["kind"] == kind) & gens["text"].str.len().gt(20)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["vec"] = [as_vector(classify(t[:2000])) for t in sub["text"]]
    rows = []
    for model, g in sub.groupby("model"):
        mean = np.mean(list(g["vec"]), axis=0)
        deviation = mean - reference
        rows.append(
            {
                "model": model,
                "n": len(g),
                **{f"raw_{p}": mean[i] for i, p in enumerate(PARTIES)},
                **{f"dev_{p}": deviation[i] for i, p in enumerate(PARTIES)},
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


# %% [markdown]
# ## Chart


# %%
def plot_profiles(profiles: pd.DataFrame, title: str, path: Path) -> None:
    dev_cols = [f"dev_{p}" for p in PARTIES]
    data = profiles.set_index("model")[dev_cols]
    data.columns = PARTIES
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(data) + 2))
    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
    ax.set_xticks(range(len(PARTIES)), PARTIES)
    ax.set_yticks(range(len(data)), data.index)
    for i in range(len(data)):
        for j in range(len(PARTIES)):
            v = data.values[i, j]
            ax.text(
                j,
                i,
                f"{v:+.02f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if abs(v) < 0.1 else "white",
            )
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label="deviation from instrument baseline", shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"wrote {path}")


# %% [markdown]
# ## Run
#
# Set GEMINI_API_KEY (free, from Google AI Studio) or OPENROUTER_API_KEY in .env,
# then run top to bottom. Generation is cached,
# so a second run only re-classifies.


# %%
def main() -> None:
    calibrate()
    reference = reference_distribution()
    gens = generate_all()
    if gens.empty:
        print("No generations yet. Set GEMINI_API_KEY or OPENROUTER_API_KEY and rerun.")
        return

    speech = build_profiles(gens, reference, kind="speech")
    neutral = build_profiles(gens, reference, kind="neutral")

    speech.to_csv(OUT_DIR / "profiles_speech.csv", index=False)
    neutral.to_csv(OUT_DIR / "profiles_neutral.csv", index=False)
    gens.to_json(
        OUT_DIR / "generations_dataset.jsonl", orient="records", lines=True, force_ascii=False
    )

    plot_profiles(
        speech,
        "Which party do LLMs write like? (deviation from instrument baseline)",
        OUT_DIR / "profiles_speech.png",
    )
    if not neutral.empty:
        plot_profiles(
            neutral,
            "Deliberately-neutral text: where does it land?",
            OUT_DIR / "profiles_neutral.png",
        )


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Limits
#
# Read before concluding anything.
#
# 1. Writing style is not ideology. The classifier predicts who wrote a text in
#    the riksdag register, not an ideological position. "Writes like S" is not
#    "is social democratic", and it is not an endorsement. Same class of caveat
#    Nordan draw about a forced compass answer not being a belief.
# 2. The instrument is not neutral. About 0.62 macro-F1, with a known lean toward
#    the larger classes. Control 2 measures that lean and reports every model as a
#    deviation from it, but the correction is not perfect.
# 3. Generated text is out of distribution. It is not a real anförande, so where
#    the classifier struggles to place it can be a fact about the text rather than
#    the model.
# 4. Small sample, prompt sensitive. One prompt template, eight topics, three
#    samples each. Nordan showed phrasing shifts results; the same holds here.
# 5. The useful result is where this disagrees with the compass. If a model's
#    language leans one way while its compass answers lean another, that says
#    something about the two methods, not that either is wrong.
