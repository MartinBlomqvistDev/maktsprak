---
license: other
license_name: riksdagens-oppna-data
license_link: https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anvandarstod/anvandningsvillkor/
language:
- sv
pretty_name: Fråga protokollet search index
size_categories:
- 100K<n<1M
task_categories:
- text-retrieval
tags:
- parliament
- riksdagen
- bm25
- hybrid-search
---

# Fråga protokollet search index

This is the index a search service reads to answer questions about what was said in the Swedish Riksdag's chamber. It holds 301 358 speeches from the eight parties, from the 2003/04 session to 18 November 2026. They are cut into 563 909 passages of whole paragraphs, with a BM25 index of the passages and one vector per passage.

Källa: Sveriges riksdag. Source: The Swedish Parliament. The Riksdag is not behind this dataset or the search that uses it.

The service bakes this package into its image when the image is built, and checks it against `package.json` before it answers a question. The code, the measurements that chose this configuration and their pre-registration are in [MartinBlomqvistDev/maktsprak](https://github.com/MartinBlomqvistDev/maktsprak).

## Files

| Path | What it holds |
| --- | --- |
| `package.json` | Counts and dates, how the service searches and answers, the hashes of the reports those settings come from, and the size and SHA-256 of every other file |
| `speeches.parquet` | One row per speech: its id, session, document number, party, the speaker as the Riksdag publishes it, date, debate title, whether it is a reply, link, the SHA-256 of its text, and its first passage |
| `speech_rows.npy` | The speech of each passage, as a row of `speeches.parquet` |
| `store/text.bin`, `store/text.offsets.npy` | Each passage's paragraphs, joined by line feeds |
| `store/passage_ids.bin`, `store/passage_ids.offsets.npy` | Each passage's id: the speech's id, then its first and last paragraph counted from 0 |
| `store/party.npy`, `store/day.npy`, `store/person.npy`, `store/persons.json` | What a search can be filtered on: party, date and speaker |
| `store/lexical/` | The BM25 index: Snowball stemming for Swedish, no stop words removed, `k1` 1.2 and `b` 0.9 |
| `store/vectors-1024-float32.npy` | One vector per passage from `Qwen/Qwen3-Embedding-8B` at revision `1d8ad4ca`, 1 024 dimensions as float32 |
| `store/store.json` | The store's own manifest |

A passage is a run of whole paragraphs from one speech, packed towards 200 words. A speech's passages together make up the whole speech, and their text has the hash the corpus gives the speech.

## How the search uses it

For each question, BM25 ranks the 1 000 best passages and the vectors rank the 1 000 nearest, under the same filter. Reciprocal rank fusion with k = 10 merges the two lists. The question's vector comes from the same model through OpenRouter, routed to Nebius in bf16.

On the natural questions of the test split, this hybrid reached a recall@10 of 0.467, against 0.267 for BM25 alone. The difference of 0.200 has a 95% bootstrap interval from 0.139 to 0.261, and that result put the hybrid in production.

## Terms and personal data

The Riksdag's open data may be used and passed on freely, provided the source is given ([Användningsvillkor](https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anvandarstod/anvandningsvillkor/)). The same terms say that a service built on the data must make clear that the Riksdag is not behind it.

Each speech names its speaker, with the Riksdag's id for that person. That is personal data, and under the Riksdag's terms anyone who uses the data must make sure it is processed in line with the GDPR.
