"""The storage spike: two ways to hold the search index, measured in one container.

Option A keeps everything in one LanceDB table inside the service: the text, the
fields a search filters on, the vectors and LanceDB's own full-text index.
Option B keeps the BM25 index of :mod:`search.index.lexical`, the text and the
filter fields as arrays on disk, and the vectors as one numpy matrix searched
exactly. Both are built from the real passages, with synthetic vectors, and
measured in a container with one CPU and 2 GiB of memory. The decision is
``docs/search/adr-001-index-store.md``.
"""
