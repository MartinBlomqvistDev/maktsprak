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
| `speeches.parquet` | One row per speech: its id, session, document number, party, the speaker as the Riksdag publishes it, date, debate title, whether it is a reply, link, the SHA-256 of its text, its first passage and how many passages it has |
| `speech_rows.npy` | The speech of each passage, as a row of `speeches.parquet` |
| `store/text.bin`, `store/text.offsets.npy` | Each passage's paragraphs, joined by line feeds |
| `store/passage_ids.bin`, `store/passage_ids.offsets.npy` | Each passage's id: the speech's id, then its first and last paragraph counted from 0 |
| `store/party.npy`, `store/day.npy`, `store/person.npy`, `store/persons.json` | What a search can be filtered on: party, date and speaker |
| `store/lexical/` | The BM25 index: Snowball stemming for Swedish, no stop words removed, `k1` 1.2 and `b` 0.9 |
| `store/vectors-1024-float32.npy` | One vector per passage, 1 024 dimensions stored as float32. The vectors came from `Qwen/Qwen3-Embedding-8B` through OpenRouter, served by Nebius in bf16 and requested at 1 024 dimensions. The weights are published at revision `1d8ad4ca`; which revision Nebius serves cannot be checked from outside |
| `store/store.json` | The store's own manifest |

A passage is a run of whole paragraphs from one speech, packed towards 200 words. A speech's passages together make up the whole speech, and their text has the hash the corpus gives the speech.

Paragraphs that are the record's own, such as applause, votes and the note that a debate is closed, are set aside by named rules when the corpus is built. A language model also read every distinct closing text those rules cut and found none of the speaker's words in them. Which model read them, and when, was not recorded, and the reading has not been checked by hand.

38 706 of the speeches, 12.8%, come from the Riksdag's snabbprotokoll, the protocol as first published, whose wording can differ from the final protocol's. They are most of the 2010/11 session and nearly all of 2012/13 and 2013/14. The Riksdag's page for such a speech shows the same text. The next version of the corpus rebuilds them from the final protocols.

## How the search uses it

For each question, BM25 ranks the 1 000 best passages and the vectors rank the 1 000 nearest, under the same filter. Reciprocal rank fusion with k = 10 merges the two lists. The question's vector comes from the same model through OpenRouter, routed to Nebius in bf16.

On the natural questions of the test split, this hybrid reached a recall@10 of 0.467, against 0.267 for BM25 alone. The difference of 0.200 has a 95% bootstrap interval from 0.139 to 0.261, and that result put the hybrid in production. On the same questions the vectors alone reached 0.517. Asked as keywords, the questions went the other way: 0.750 for BM25 against 0.556 for the vectors. Only the hybrid against BM25 decided what runs in production.

## Terms and personal data

The Riksdag's open data may be used and passed on freely, provided the source is given ([Användningsvillkor](https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anvandarstod/anvandningsvillkor/)). The same terms say that a service built on the data must make clear that the Riksdag is not behind it.

Each speech names its speaker as the Riksdag publishes the name, and most carry the Riksdag's id for that person: 216 carry none and 246 a placeholder of zeros, which the store treats as no id. That is personal data, and under the Riksdag's terms anyone who uses the data must make sure it is processed in line with the GDPR.

## Known errors

These are fixed in the next version of the corpus:

- Eight speeches keep a note of the record's own inside a paragraph: "(ohörbart)" in GS0914-133, GS0914-135 and GS0914-137, "(talaren gör tecken liknande teckenspråk)" in GU09131-115, "(Applåder)" in GV093-6, GV09126-31 and H10948-75, and "(Applåd)" in GV09122-222.
- Where the record breaks a line inside a word, a paragraph can start or end inside it, as in GR0918-9: "som de nya p" and "ublic service-utredarna".
- Two speeches carry another party than every other speech by the same person: GT0921-34 (S, otherwise M) and GT0932-91 (M, otherwise C).
- Four pairs of speeches hold the same text by the same speaker in one protocol: H109123-100 and -102, H50938-84 and -86, HD09106-171 and -173, HD09106-172 and -174.
- Seventeen speeches end with the short form of the record's note on who concurred in the speech, "I anförandet instämde" followed by names and parties. They come from the sessions 2006/07 to 2022/23. The rule recognises only the longer form, "I detta anförande instämde", so the note stands as their last paragraph: GU0918-133, GU0929-50, GV09130-95, GW09109-64, GW0931-196, GW0952-141, GW0953-29, GW0972-74, GY0910-97, GY0960-131, GY0983-80, GY0987-27, GZ0949-112, GZ0986-4, H10918-19, H909136-43 and HA099-5.
- GY0919-25, the answer to interpellation 2010/11:42, stands under Maria Larsson (KD). The final protocol 2010/11:19 has the same text, word for word, under Gunilla Carlsson (M).
