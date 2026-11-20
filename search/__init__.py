"""Search over the Riksdag's speeches: corpus, index, answer step and evaluation.

Self-contained on purpose. Nothing here imports ``src.maktsprak_pipeline``:
importing that package requires Supabase credentials, which neither the build
scripts nor the search service have any reason to hold.
"""
