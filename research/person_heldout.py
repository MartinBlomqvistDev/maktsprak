"""Re-score the classifier's held-out figures on the Riksdag's person ids.

The deployed classifier (``MartinBlomqvist/maktsprak_classifier_clean``, trained
2026-07-13) held out 15 % of speaker *strings* (``val_speakers.json``). The
protocols print one person in several ways, with a title (``Statsrådet PER
BOLUND`` / ``PER BOLUND``), after a change of surname, misspelt or with broken
spacing, so a person could be held out under one string and trained under
another. Here a person is the Riksdag's own id for a speaker (``person_id`` in
the search corpus), and a person counts as held out only if none of their rows
was on the training side.

Steps, run from the repository root (``--root`` points at the checkout that
holds the data, the model and ``research/out``):

    python research/person_heldout.py split     # training-time keys -> persons
    python research/person_heldout.py headline  # validation rows, deployed model
    python research/person_heldout.py studies   # /riktmarke, calibration, baseline

``split`` reads speaker keys (no text) from Supabase in the state the model
trained on: every row dated up to 2026-07-13. For 2016-2026 that state is intact;
352 of the 2015 rows were deleted later, and every key the model trained on is
still there. ``headline`` reads the text of the validation rows only. No step
calls a paid API: the language models' answers come from the benchmark's caches
in ``research/out``.

Each step writes its section of ``data/reports/person_heldout.json``.
Intermediate files go to ``research/out/person_heldout/``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PARTIES = ["C", "KD", "L", "M", "MP", "S", "SD", "V"]
TRAINED_ON = "2026-07-13"
SEED = 20261127
REPS = 10_000
REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "data" / "reports" / "person_heldout.json"


# ---------------------------------------------------------------- names
def letters(name: str) -> str:
    """Upper-case letters of a name before any party tag, title words kept."""
    s = str(name)
    if "(" in s:
        s = s[: s.index("(")]
    return "".join(ch for ch in s.upper() if ch.isalpha())


def core_name(name: str) -> str:
    """A PDF speaker key reduced to its upper-case words, chair text dropped.

    ``Statsrådet PER BOLUND`` gives ``PERBOLUND``; a key that swallowed the
    chair's interjection (``ANDRE VICE TALMANNEN: ... (ROBERT STENKVIST``) gives
    the name after the last parenthesis.
    """
    s = str(name)
    if ("TALMANNEN:" in s or "PRESIDENTEN:" in s) and "(" in s:
        s = s[s.rindex("(") + 1 :]
    elif "(" in s:
        s = s[: s.index("(")]
    upper = [t for t in s.split() if t.isupper()]
    return "".join(ch for ch in ("".join(upper) if upper else s).upper() if ch.isalpha())


def riksdag_core(name: str) -> str:
    """A Riksdag speaker field without party tag and without an ``i <place>`` suffix."""
    s = str(name)
    if "(" in s:
        s = s[: s.index("(")]
    toks = s.split()
    if "i" in toks[1:]:
        toks = toks[: toks.index("i", 1)]
    return "".join(ch for ch in " ".join(toks).upper() if ch.isalpha())


class PersonMatcher:
    """Match (protocol, party, PDF speaker key) to a Riksdag person id.

    Names are compared within one protocol and one party: the full key first,
    then its core, then a unique suffix (titles are prefixes), then a close
    match. Outside the Riksdag's own records the fallback is a unique person
    active that year whose name ends with the core.
    """

    def __init__(self, root: Path, since: str = "2015-05-01") -> None:
        rk = pd.read_parquet(
            root / "data" / "search" / "anforanden.parquet",
            columns=["dok_id", "date", "speaker_raw", "person_id", "party"],
        )
        rk["d"] = rk["date"].astype(str).str[:10]
        rk = rk[rk["d"] >= since].dropna(subset=["person_id"]).copy()
        rk["u"] = rk["speaker_raw"].map(riksdag_core)
        rk["uf"] = rk["speaker_raw"].map(letters)
        self.names = (
            rk.groupby(["dok_id", "party"])[["u", "uf", "person_id"]]
            .apply(
                lambda g: (
                    dict(zip(g["u"], g["person_id"], strict=True))
                    | dict(zip(g["uf"], g["person_id"], strict=True))
                )
            )
            .to_dict()
        )
        rk["y"] = rk["d"].str[:4].astype(int)
        self.by_person = rk.groupby("person_id").agg(
            us=("u", lambda s: set(s)), y0=("y", "min"), y1=("y", "max")
        )
        self._cache: dict[tuple[str, int], tuple[str, str | None]] = {}

    @staticmethod
    def _in_protocol(n: str, cand: dict[str, str]) -> tuple[str, str | None]:
        if n in cand:
            return "exact_in_protocol", cand[n]
        suffix = {cand[u] for u in cand if u.endswith(n)}
        if len(suffix) == 1:
            return "suffix_in_protocol", next(iter(suffix))
        scored = sorted(
            (
                max(
                    difflib.SequenceMatcher(None, n, u).ratio(),
                    difflib.SequenceMatcher(None, n, u[-len(n) :]).ratio(),
                ),
                pid,
            )
            for u, pid in cand.items()
        )[::-1]
        if (
            scored
            and scored[0][0] >= 0.85
            and (
                len(scored) == 1
                or scored[1][1] == scored[0][1]
                or scored[1][0] < scored[0][0] - 0.1
            )
        ):
            return "fuzzy_in_protocol", scored[0][1]
        return "", None

    def _global(self, n: str, year: int) -> tuple[str, str | None]:
        if (n, year) not in self._cache:
            hits = {
                pid
                for pid, row in self.by_person.iterrows()
                if row.y0 - 1 <= year <= row.y1 + 1 and any(u.endswith(n) for u in row.us)
            }
            if len(hits) == 1:
                self._cache[(n, year)] = ("suffix_global", next(iter(hits)))
            else:
                self._cache[(n, year)] = ("ambiguous_global" if hits else "unmatched", None)
        return self._cache[(n, year)]

    def match(self, protocol: str, party: str, speaker: str, date: str) -> tuple[str, str | None]:
        """Return (how it matched, person id or None)."""
        cand = self.names.get((protocol, party), {})
        full = letters(speaker)
        if full in cand:
            return "exact_full_in_protocol", cand[full]
        how, pid = self._in_protocol(core_name(speaker), cand)
        if how:
            return how, pid
        return self._global(core_name(speaker), int(str(date)[:4]))

    def map_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``how`` and ``person_id`` to rows with protocol_id, party, speaker, d."""
        res = [
            self.match(p, q, s, d)
            for p, q, s, d in zip(
                df["protocol_id"], df["party"], df["speaker"], df["d"], strict=True
            )
        ]
        out = df.copy()
        out["how"] = [r[0] for r in res]
        out["person_id"] = [r[1] for r in res]
        return out


# ---------------------------------------------------------------- metrics
def confusion(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """8x8 counts, row = gold party, column = prediction; unparsed answers dropped."""
    idx = {p: i for i, p in enumerate(PARTIES)}
    m = np.zeros((8, 8))
    for g, p in zip(gold, pred, strict=True):
        if p in idx:
            m[idx[g], idx[p]] += 1
    return m


def scores(m: np.ndarray) -> tuple[float, float]:
    """Accuracy and macro-F1 over all eight parties (an absent class scores 0)."""
    tp = np.diag(m)
    prec = np.divide(tp, m.sum(0), out=np.zeros(8), where=m.sum(0) > 0)
    rec = np.divide(tp, m.sum(1), out=np.zeros(8), where=m.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(8), where=(prec + rec) > 0)
    return float(tp.sum() / m.sum()), float(f1.mean())


def bootstrap(df: pd.DataFrame, cols: list[str], reps: int = REPS) -> dict:
    """Point estimates and 95 % intervals, persons resampled within party.

    ``df`` needs ``gold``, ``person`` (one cluster per person and party) and one
    column per prediction set.
    """
    rng = np.random.default_rng(SEED)
    persons = {p: g["person"].unique() for p, g in df.groupby("gold")}
    out = {}
    for c in cols:
        per = {
            pid: confusion(g["gold"].to_numpy(), g[c].to_numpy()) for pid, g in df.groupby("person")
        }
        acc, f1 = scores(sum(per.values()))
        draws = []
        for _ in range(reps):
            m = np.zeros((8, 8))
            for ps in persons.values():
                for pid in rng.choice(ps, size=len(ps), replace=True):
                    m += per[pid]
            draws.append(scores(m))
        d = np.array(draws)
        out[c] = {
            "n": int(df[c].isin(PARTIES).sum()),
            "accuracy": round(acc, 3),
            "accuracy_ci95": [round(float(q), 3) for q in np.percentile(d[:, 0], [2.5, 97.5])],
            "macro_f1": round(f1, 3),
            "macro_f1_ci95": [round(float(q), 3) for q in np.percentile(d[:, 1], [2.5, 97.5])],
        }
    return out


def paired_difference(df: pd.DataFrame, keep: pd.Series, reps: int = REPS) -> dict:
    """All rows minus the kept rows, on the same person draws."""
    rng = np.random.default_rng(SEED)
    per = {
        pid: (confusion(g["gold"].to_numpy(), g["pred"].to_numpy()), bool(keep.loc[g.index].all()))
        for pid, g in df.groupby("person")
    }
    persons = {p: g["person"].unique() for p, g in df.groupby("gold")}
    diffs = []
    for _ in range(reps):
        m_all, m_kept = np.zeros((8, 8)), np.zeros((8, 8))
        for ps in persons.values():
            for pid in rng.choice(ps, size=len(ps), replace=True):
                m, kept = per[pid]
                m_all += m
                if kept:
                    m_kept += m
        a1, f1 = scores(m_all)
        a2, f2 = scores(m_kept)
        diffs.append((a1 - a2, f1 - f2))
    d = np.array(diffs)
    a_all, f_all = scores(sum(m for m, _ in per.values()))
    a_kept, f_kept = scores(sum(m for m, k in per.values() if k))
    return {
        "accuracy": round(a_all - a_kept, 3),
        "accuracy_ci95": [round(float(q), 3) for q in np.percentile(d[:, 0], [2.5, 97.5])],
        "macro_f1": round(f_all - f_kept, 3),
        "macro_f1_ci95": [round(float(q), 3) for q in np.percentile(d[:, 1], [2.5, 97.5])],
    }


# ---------------------------------------------------------------- shared
def write_section(name: str, payload: dict) -> None:
    """Replace one section of the tracked report."""
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    report[name] = payload
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def supabase_client(root: Path):
    """Read-only client from the checkout's .env; the keys are never printed."""
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv(root / ".env")
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def keyset(client, table: str, cols: str, key: str) -> pd.DataFrame:
    """Read a whole table, 1 000 rows at a time, on an indexed key."""
    rows, last = [], None
    while True:
        q = client.table(table).select(cols).order(key, desc=False).limit(1000)
        if last is not None:
            q = q.gt(key, last)
        data = q.execute().data
        if not data:
            break
        rows.extend(data)
        last = data[-1][key]
        if len(data) < 1000:
            break
    return pd.DataFrame(rows)


class Classifier:
    """The deployed weights, loaded from the local model directory."""

    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def probs(self, texts: list[str], batch: int = 16) -> np.ndarray:
        """Softmax probabilities in ``self.labels`` order, first 512 tokens."""
        lens = [len(x) for x in self.tok(texts, truncation=True, max_length=512)["input_ids"]]
        order = sorted(range(len(texts)), key=lambda i: lens[i])
        out = np.zeros((len(texts), len(self.labels)))
        with self.torch.inference_mode():
            for b in range(0, len(order), batch):
                idx = order[b : b + batch]
                enc = self.tok(
                    [texts[i] for i in idx],
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                )
                out[idx] = self.torch.softmax(self.model(**enc).logits, dim=-1).numpy()
        return out

    def argmax(self, texts: list[str]) -> list[str]:
        """One text at a time, as the research scripts' ``classify()`` does."""
        res = []
        with self.torch.inference_mode():
            for t in texts:
                enc = self.tok(t, return_tensors="pt", truncation=True, max_length=512)
                res.append(self.labels[int(self.model(**enc).logits.argmax(dim=-1))])
        return res

    def window(self, text: str) -> str:
        """The benchmark's shared input: the 512-token window, decoded."""
        ids = self.tok(text, truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
        return self.tok.decode(ids, skip_special_tokens=True)


def load_rows(work: Path) -> pd.DataFrame:
    return pd.read_parquet(work / "rows.parquet")


def trained_persons(rows: pd.DataFrame) -> set[str]:
    return set(rows.loc[(rows["side"] == "train") & rows["person_id"].notna(), "person_id"])


# ---------------------------------------------------------------- steps
def step_split(root: Path, work: Path) -> None:
    """Training-time keys from Supabase, each row mapped to a person."""
    client = supabase_client(root)
    val = set(
        json.loads((root / "data/models/party_classifier/val_speakers.json").read_text("utf-8"))
    )
    sp = keyset(
        client,
        "speeches",
        "supabase_id, id, protocol_id, protocol_date, speaker, party",
        "supabase_id",
    )
    tw = keyset(client, "tweets", "tweet_id, username", "tweet_id")
    sp["d"] = sp["protocol_date"].astype(str).str[:10]
    sp = sp[(sp["d"] <= TRAINED_ON) & sp["party"].isin(PARTIES)].copy()
    sp["speaker"] = sp["speaker"].fillna("Okänd")
    rows = PersonMatcher(root).map_frame(sp)
    rows["side"] = np.where(rows["speaker"].isin(val), "val", "train")
    rows.to_parquet(work / "rows.parquet", index=False)

    per = (
        rows.dropna(subset=["person_id"])
        .groupby("person_id")
        .agg(
            sides=("side", lambda s: sorted(set(s))),
            keys=("speaker", lambda s: sorted(set(s))),
            val_rows=("side", lambda s: int((s == "val").sum())),
            train_rows=("side", lambda s: int((s == "train").sum())),
            party=("party", lambda s: ",".join(sorted(set(s)))),
        )
    )
    leaked = per[per["sides"].map(len) == 2]
    val_persons = per[per["sides"].map(lambda s: "val" in s)]
    keys = set(rows["speaker"]) | set(tw["username"])
    write_section(
        "split",
        {
            "trained_on": TRAINED_ON,
            "keys": len(keys),
            "val_keys": len(val),
            "val_keys_missing_now": sorted(val - keys),
            "rows": len(rows),
            "rows_unmatched": int(rows["person_id"].isna().sum()),
            "match": rows["how"].value_counts().to_dict(),
            "val_rows": int((rows["side"] == "val").sum()),
            "val_persons": len(val_persons),
            "leaked_persons": len(leaked),
            "leaked_val_rows": int(leaked["val_rows"].sum()),
            "strict_persons": len(val_persons) - len(leaked),
            "leaked": [
                {
                    "person_id": pid,
                    "party": r["party"],
                    "val_rows": int(r["val_rows"]),
                    "train_rows": int(r["train_rows"]),
                    "keys": list(r["keys"]),
                }
                for pid, r in leaked.sort_values("val_rows", ascending=False).iterrows()
            ],
        },
    )
    print(f"{len(keys)} keys, {len(val_persons)} val persons, {len(leaked)} on both sides")


def step_headline(root: Path, work: Path) -> None:
    """The deployed model on every validation row, old split against strict persons."""
    rows = load_rows(work)
    leaked = {
        pid
        for pid, g in rows.dropna(subset=["person_id"]).groupby("person_id")
        if g["side"].nunique() == 2
    }
    val = rows[rows["side"] == "val"].copy()
    cache = work / "val_preds.parquet"
    if not cache.exists():
        client = supabase_client(root)
        ids = val["supabase_id"].astype(int).tolist()
        texts = []
        for i in range(0, len(ids), 100):
            sel = (
                client.table("speeches")
                .select("supabase_id, text")
                .in_("supabase_id", ids[i : i + 100])
            )
            texts.extend(sel.execute().data)
        val = val.merge(pd.DataFrame(texts), on="supabase_id", how="left")
        clean = [t.replace("\n", " ").strip() if isinstance(t, str) else "" for t in val["text"]]
        clf = Classifier(root / "data/models/party_classifier")
        p = clf.probs(clean)
        val["pred"] = [clf.labels[i] for i in p.argmax(axis=1)]
        val.drop(columns=["text"]).to_parquet(cache, index=False)
    val = pd.read_parquet(cache)
    val["gold"] = val["party"]
    val["person"] = val["person_id"].fillna("?") + "|" + val["party"]
    keep = ~val["person_id"].isin(leaked)
    write_section(
        "headline",
        {
            "speeches": bootstrap(val, ["pred"])["pred"],
            "strict": bootstrap(val[keep], ["pred"])["pred"],
            "leaked_only": bootstrap(val[~keep], ["pred"], reps=2000)["pred"],
            "old_minus_strict": paired_difference(val, keep),
            "strict_persons": int(val.loc[keep, "person_id"].nunique()),
        },
    )
    print("headline written")


def step_studies(root: Path, work: Path) -> None:
    """/riktmarke and the LLM study's controls, re-scored on strict persons."""
    rows = load_rows(work)
    trained = trained_persons(rows)
    val = set(
        json.loads((root / "data/models/party_classifier/val_speakers.json").read_text("utf-8"))
    )
    clf = Classifier(root / "data/models/party_classifier")
    corpus = pd.read_parquet(
        root / "data/parquet/speeches_full.parquet",
        columns=["id", "protocol_id", "protocol_date", "speaker", "party", "text"],
    )
    corpus["d"] = corpus["protocol_date"].astype(str).str[:10]
    corpus["year"] = pd.to_datetime(corpus["protocol_date"], errors="coerce").dt.year
    modern = corpus[
        (corpus["year"] >= 2015) & corpus["party"].isin(PARTIES) & corpus["text"].notna()
    ]
    people = PersonMatcher(root).map_frame(
        modern[["protocol_id", "party", "speaker", "d"]].drop_duplicates()
    )
    modern = modern.merge(people, on=["protocol_id", "party", "speaker", "d"], how="left")
    modern["strict"] = (
        modern["speaker"].isin(val)
        & modern["person_id"].notna()
        & ~modern["person_id"].isin(trained)
    )
    out: dict = {}

    # /riktmarke: the benchmark's own draw (sample_speeches), then its stored
    # answers by index, so nothing is asked of the language models again.
    b = corpus[corpus["speaker"].isin(val)]
    b = b[pd.to_datetime(b["protocol_date"]).dt.year >= 2015]
    b = b[b["party"].isin(PARTIES)]
    b = b[b["text"].str.len() > 200]
    parts = [g.sample(min(40, len(g)), random_state=13) for _, g in b.groupby("party")]
    bench = pd.concat(parts).sample(frac=1, random_state=13).reset_index(drop=True)
    bench = bench.merge(people, on=["protocol_id", "party", "speaker", "d"], how="left")
    bench["strict"] = bench["person_id"].notna() & ~bench["person_id"].isin(trained)
    bench["kb"] = clf.argmax([clf.window(t) for t in bench["text"]])
    bench["kb2000"] = clf.argmax([t[:2000] for t in bench["text"]])
    cols: list[str] = []
    for name, own, suffix in (
        ("benchmark_preds.jsonl", "kb", ""),
        ("benchmark_preds_2000char.jsonl", "kb2000", "@2000"),
    ):
        cols.append(own)
        lines = (root / "research/out" / name).read_text(encoding="utf-8").splitlines()
        cache = [json.loads(x) for x in lines if x.strip()]
        for m in sorted({r["model"] for r in cache}):
            col = [None] * len(bench)
            for r in cache:
                if r["model"] == m:
                    col[r["idx"]] = r["pred"]
            bench[m + suffix] = col
            cols.append(m + suffix)
    bench["gold"] = bench["party"]
    bench["person"] = bench["person_id"].fillna("?") + "|" + bench["party"]
    strict = bench[bench["strict"]]
    out["riktmarke"] = {
        "sampled": len(bench),
        "left_out": int((~bench["strict"]).sum()),
        "left_out_persons": int(bench.loc[~bench["strict"], "person_id"].nunique()),
        "left_out_unmatched": int(bench["person_id"].isna().sum()),
        "strict_by_party": strict.groupby("party").size().to_dict(),
        "all": bootstrap(bench, cols, reps=200),
        "strict": bootstrap(strict, cols),
    }

    # Training-speaker side of the gap, drawn the same way from training keys.
    tr = modern[
        ~modern["speaker"].isin(val)
        & modern["person_id"].isin(trained)
        & (modern["text"].str.len() > 200)
    ]
    trs = pd.concat([g.sample(min(40, len(g)), random_state=13) for _, g in tr.groupby("party")])
    trs = trs.reset_index(drop=True)
    trs["kb"] = clf.argmax([clf.window(t) for t in trs["text"]])
    trs["gold"], trs["person"] = trs["party"], trs["person_id"] + "|" + trs["party"]
    out["training_speakers"] = bootstrap(trs, ["kb"])["kb"]

    # Calibration: the study's draw (60 per party, seed 13, first 2000 characters).
    cal = (
        modern[modern["strict"]]
        .groupby("party", group_keys=False)
        .apply(lambda g: g.sample(min(60, len(g)), random_state=13))
        .reset_index(drop=True)
    )
    cal["kb"] = clf.argmax([t[:2000] for t in cal["text"]])
    cal["gold"], cal["person"] = cal["party"], cal["person_id"] + "|" + cal["party"]
    out["calibration"] = {
        "persons": int(cal["person_id"].nunique()),
        **bootstrap(cal, ["kb"])["kb"],
    }

    # Baseline: whole corpus against strict persons (80 per party, seed 21).
    def mean_probs(frame: pd.DataFrame) -> dict[str, float]:
        p = clf.probs([t[:2000] for t in frame["text"]], batch=1)
        return {lab: round(float(v), 4) for lab, v in zip(clf.labels, p.mean(axis=0), strict=True)}

    allp = corpus[corpus["text"].notna() & corpus["party"].isin(PARTIES)]
    ref = mean_probs(
        allp.groupby("party", group_keys=False).apply(
            lambda g: g.sample(min(80, len(g)), random_state=21)
        )
    )
    held = mean_probs(
        modern[modern["strict"]]
        .groupby("party", group_keys=False)
        .apply(lambda g: g.sample(min(80, len(g)), random_state=21))
    )
    out["baseline"] = {
        "whole_corpus": ref,
        "strict_persons": held,
        "shift": {p: round(held[p] - ref[p], 4) for p in PARTIES},
    }
    write_section("studies", out)
    print("studies written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["split", "headline", "studies"])
    parser.add_argument(
        "--root", type=Path, default=REPO, help="checkout that holds data/, .env and research/out"
    )
    args = parser.parse_args()
    work = REPO / "research" / "out" / "person_heldout"
    work.mkdir(parents=True, exist_ok=True)
    {"split": step_split, "headline": step_headline, "studies": step_studies}[args.step](
        args.root, work
    )


if __name__ == "__main__":
    main()
