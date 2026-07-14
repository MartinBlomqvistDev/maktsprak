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

## 2026-07-14 — `vi_dom_patterns.csv` created (32 patterns)

Initial table. Its shape was decided by an audit of real matches, not by
intuition, and three candidate patterns were **rejected on the evidence**:

- **`dessa människor` (3 356 hits) and `de här människorna` (1 262) — cut.**
  Sampled matches were overwhelmingly *sympathetic*, not othering:
  "Vi har all anledning att stötta de här människorna" (S), "Vi måste kunna
  möta de här människorna med behandling" (V), "Vi pratar sällan om dessa
  människor" (MP). These are deictic pointers — they refer to a group, they do
  not construct one. They fail the table's own inclusion rule (an out-group must
  be defined by *identity, origin, class or status*; "these people" defines
  nothing) and would have counted compassion as hostility.
- **Bare `eliten` (222 hits) — cut, replaced by qualified forms.** It is a
  homograph: "den internationella eliten" in an L speech means *world-class
  researchers*. `politiska eliten`, `ekonomiska eliten`, `makteliten` and
  `etablissemanget` carry the anti-elite sense unambiguously.
- **`de som kommer hit` and `invandrarna` — kept, but only countable inside a
  construction.** On their own they are referential and used across the whole
  spectrum, often sympathetically ("De som kommer hit ska ges möjlighet att vara
  med och bygga landet" — V). They score nothing as bare hits; they only count
  when an in-group marker appears in the same sentence.

Inclusion rule, applied to every future addition: *a pattern qualifies if it
names a group of people defined by identity, origin, class or status — not a
named organisation, and not the specific policy under debate.*

The economic out-group terms (`de rika`, `miljardärerna`, `storföretagen` …)
exist so the instrument can find the same rhetorical move aimed leftward. That
is not balance for its own sake: the census showed the construction in the
Riksdag is *mostly* economic, and a table without those patterns would have
reported the opposite.

## 2026-07-14 — `inclusive_occupational_pairs.csv` created (14 pairs)

Measured on the terms the Riksdag actually uses about itself (`riksdagsman` →
`riksdagsledamot`, `talesman` → `talesperson`, `tjänsteman` → `tjänsteperson`),
after a corpus check found the textbook pairs simply do not occur in political
debate: **`brandperson`, `flygvärd` and `ombudsperson` have zero occurrences**
across 2002-2026.

`sjuksköterska` (ISOF records no settled neutral alternative) and `brandman`
(neutral form has zero uptake) are kept **as deliberate null cases** — tracked
to show absence of change where none is claimed, rather than dropped to tidy up
the table.
