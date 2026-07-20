# Lexicon & pattern-table changelog

Every change to a pattern table is recorded here. This file *is* the audit
trail: it is what makes "the word lists were tuned to make party X look bad"
a checkable claim rather than an unanswerable one. `git log` on the CSV shows
what changed; this file says **why**, and whether the precision audit was re-run.

**Policy.** A change touching **≥5 patterns**, or any change to what a
`subtype` *means*, requires re-running `scripts/validate_tone.py --dimension
<id>` and re-publishing the precision number before merge. Single-typo fixes do
not. Adding a pattern is a change; removing one is a change; re-classifying one
is a change.

---

## 2026-07-16, gate run #2: a corpus bug, a `hen` demotion, a ratio that meant nothing

**The corpus was 6.7% duplicated.** Found via the census printing the same
sentence twice. `speeches_full.parquet` held **80 541 rows but 75 116 unique
ids**, the 2002-2015 backfill overlapped the range the weekly ETL had already
ingested. **2014 was 95.2% duplicated, 2002 76.3%, 2015 75.4%**; the copies are
byte-identical. Blast radius is wider than tone: a duplicated speech is counted
twice in every denominator and inflates the confidence of every z-score built on
it (rates mostly survive, numerator and denominator double together, but
counts and z-scores do not). Fixed at the boundary (`export_corpus.py` now
dedups on `id` and reports the per-year shape), and `scripts/_corpus.py`, the
loader every analysis passes through, now **raises** on a duplicated archive
rather than letting it quietly skew a published number. The database cleanup
(dedup plus a `UNIQUE(id)` constraint) is tracked in `migrations/002`.

**`hen` demoted to a corpus-wide series.** The new density guard rejected the
per-party split: median 3 hits per party-year cell, 55% of cells empty. The plan
allowed 2-3 year binning as a fallback, so it was tested rather than assumed , 
**it does not rescue it**: 2-year 47% empty, 3-year 49%, 5-year 39%. All fail.
Pooled across the chamber the same 324 hits are a clean adoption curve matching
the published SAOL timeline, so that is what ships; per-party becomes pooled
totals (`hen_by_party`), not lines. `DimensionSpec.group_col` added, being
corpus-level rather than party-level is a real property of a dimension.

**The occupational ratio was meaningless for half the table.** It reported
`lärarinna → lärare` at **99.9% neutral** and `fackman → expert` at 99.2%. Both
are nonsense: `lärare` is the unmarked base word for teacher (19 013 hits) and
was never coined to replace `lärarinna`, so the ratio measured how often anyone
mentions teachers. A ratio only means "which word did the speaker choose" when
the neutral form is a **coinage existing solely as the replacement**. New
`measure` column: `ratio` for coinages (`riksdagsledamot`, `talesperson`,
`tjänsteperson`), `decline` for marked feminine forms (counts, no ratio),
`none` for `sjuksköterska`. `fackman → expert` dropped outright, `expert` is an
ordinary word, and pairing them manufactures a number out of nothing.

**And the finding that survives all of it:** the chamber renamed **itself** , 
`riksdagsman` → `riksdagsledamot` at **96.9%**, plural **98.2%**, but did not
rename its civil servants: `tjänsteman` → `tjänsteperson` **2.5%**, plural
**3.8%**, `ombudsperson` **0.0%**. Apolitical, mechanical, and genuinely
interesting.

## 2026-07-14 (later), first validation gate run: two dimensions cut, three bugs fixed

`scripts/validate_tone.py` on the full corpus. Nothing here was predicted from
reading patterns; all of it came from reading **real matches**.

**Cut from the launch set (failed their own gate):**

- **`folk`, unregistered.** 11 848 hits, and precise in the narrow sense: the
  matches really do contain the words. But they are not *people-centrism* in
  Mudde's sense (the people as one virtuous body against a corrupt elite), they
  are the electorate, referred to: "skogen har en speciell plats i hjärtat hos
  svenska folket" (V), "svenska folket tror mer på alliansregeringen" (C).
  Charting this as people-centrism would repeat the `dessa människor` mistake
  with a longer word list.
- **`antielit`, unregistered.** Precision is fine; the *density* is not.
  254 hits over 24 years is a **median of 2 hits per non-empty party-year cell,
  with 91 of 192 cells empty**. That is not a time series. Note the
  speech-count suppression floor could not catch this, the speeches are there,
  the hits are not, so a hit-density check belongs in the gate.

Both survive as measure functions because the census consumes them. A test
asserts they stay unregistered.

**Patterns removed:**

- **`maktens korridorer`**, matched `#imaktenskorridorer`, the **#MeToo
  hashtag**. An L speech about 1 600 women in politics signing an appeal was
  counted as anti-elite rhetoric.
- **`storföretagen` / `storbolagen`**, usually neutral sectoral description
  ("Storföretagen och industrin har kompletterats av…", C), not class framing.
- **`de styrande`**, too often literal (whoever currently governs).

**Homographs fixed rather than discarded** (new `exclude_next` column): `de
rika`/`de rikaste`/`de allra rikaste` now drop a match followed by
`länderna|länder|delarna|regionerna|procenten`. The audit caught "Sverige är
inte längre ett av de rikaste **länderna** i världen" (M) counted as framing a
wealthy out-group. It is about countries. Regression-tested both ways.

**Launch set is now three dimensions across three techniques** , 
`klasskonflikt` (lexical), `lix` (stylometric), `hen` (structural), plus the
census and the occupational report as findings rather than charts. The
technique-diversity claim survives the cuts intact, which is the point of having
built it that way.

## 2026-07-14, `vi_dom_patterns.csv` created (32 patterns)

Initial table. Its shape was decided by an audit of real matches, not by
intuition, and three candidate patterns were **rejected on the evidence**:

- **`dessa människor` (3 356 hits) and `de här människorna` (1 262), cut.**
  Sampled matches were overwhelmingly *sympathetic*, not othering:
  "Vi har all anledning att stötta de här människorna" (S), "Vi måste kunna
  möta de här människorna med behandling" (V), "Vi pratar sällan om dessa
  människor" (MP). These are deictic pointers, they refer to a group, they do
  not construct one. They fail the table's own inclusion rule (an out-group must
  be defined by *identity, origin, class or status*; "these people" defines
  nothing) and would have counted compassion as hostility.
- **Bare `eliten` (222 hits), cut, replaced by qualified forms.** It is a
  homograph: "den internationella eliten" in an L speech means *world-class
  researchers*. `politiska eliten`, `ekonomiska eliten`, `makteliten` and
  `etablissemanget` carry the anti-elite sense unambiguously.
- **`de som kommer hit` and `invandrarna`, kept, but only countable inside a
  construction.** On their own they are referential and used across the whole
  spectrum, often sympathetically ("De som kommer hit ska ges möjlighet att vara
  med och bygga landet", V). They score nothing as bare hits; they only count
  when an in-group marker appears in the same sentence.

Inclusion rule, applied to every future addition: *a pattern qualifies if it
names a group of people defined by identity, origin, class or status, not a
named organisation, and not the specific policy under debate.*

The economic out-group terms (`de rika`, `miljardärerna`, `storföretagen` …)
exist so the instrument can find the same rhetorical move aimed leftward. That
is not balance for its own sake: the census showed the construction in the
Riksdag is *mostly* economic, and a table without those patterns would have
reported the opposite.

## 2026-07-14, `inclusive_occupational_pairs.csv` created (14 pairs)

Measured on the terms the Riksdag actually uses about itself (`riksdagsman` →
`riksdagsledamot`, `talesman` → `talesperson`, `tjänsteman` → `tjänsteperson`),
after a corpus check found the textbook pairs simply do not occur in political
debate: **`brandperson`, `flygvärd` and `ombudsperson` have zero occurrences**
across 2002-2026.

`sjuksköterska` (ISOF records no settled neutral alternative) and `brandman`
(neutral form has zero uptake) are kept **as deliberate null cases**, tracked
to show absence of change where none is claimed, rather than dropped to tidy up
the table.
