-- Make the weekly ETL idempotent. Run once in the Supabase SQL editor.
--
-- Context: `speeches.id` (e.g. "HD098_maria-malm_m") is the natural key, but the
-- table's PRIMARY KEY is the surrogate `supabase_id`. `.upsert(row)` resolves
-- against the primary key, and `supabase_id` is auto-generated so it is never in
-- the payload — which means every "upsert" the ETL has ever issued was silently
-- a plain INSERT. Re-ingesting a protocol appended a second copy of every speech
-- in it. That is how the archive ended up with 5 425 duplicate ids.
--
-- Two steps, in this order. Step 2 fails while duplicates remain.

-- 1. Drop rows repeating an id, keeping the earliest.
--
--    Safe as written: `id` is now the natural key (protocol + speaker + party),
--    so rows sharing an id describe the same speaker in the same protocol. Under
--    the OLD positional id this delete would NOT have been safe — ids were an
--    enumerate() counter, so one id could name two different speeches.
--
--    Verify before running:
--        select count(*), count(distinct id) from public.speeches;
delete from public.speeches a
      using public.speeches b
      where a.id = b.id
        and a.supabase_id > b.supabase_id;

-- 2. The actual fix. Without this constraint there is nothing for ON CONFLICT
--    to resolve against, and the ETL cannot be idempotent no matter what the
--    client code says.
alter table public.speeches
    add constraint speeches_id_key unique (id);

-- After this runs, switch the writes in src/maktsprak_pipeline/db/speeches.py to
--     .upsert(rows, on_conflict="id")
-- which is a no-op until the constraint above exists (PostgREST needs a real
-- unique index to target).
--
-- Note the corpus itself does not depend on any of this: data/raw holds every
-- source document and `scripts/rebuild_corpus.py` regenerates the Parquet
-- archive offline. Supabase is the ETL landing zone, not the source of truth.
