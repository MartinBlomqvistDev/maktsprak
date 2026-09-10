-- Recover the project from the Free Plan 0.5 GB cap and stop it refilling.
--
-- Run in the Supabase SQL editor, one block at a time, reading the output as you
-- go. Do not run the file in one go: the checks exist to be read between the
-- destructive steps.
--
-- Context: the project reached 0.507 GB on 2026-09-10. Supabase documents that a
-- Free Plan project goes read-only above 500 MB and returns automatically once
-- usage is under 95 % of the cap, so writes may already be failing with "cannot
-- execute INSERT in a read-only transaction". Reads are unaffected either way.
--
-- Nothing user-facing depends on this database. maktsprak.se serves precomputed
-- static JSON, the predict demo runs on Cloud Run, and the only Supabase call in
-- the site code is unused. The corpus lives in data/parquet/speeches_full.parquet
-- (75 148 rows, every id unique, 2002-09-30 to 2026-06-23, verified 2026-09-10),
-- and scripts/rebuild_corpus.py regenerates it from data/raw offline. Supabase is
-- the weekly-ETL landing zone.
--
-- Two causes: the 2002-2015 backfill overlapped the range the weekly ETL had
-- already ingested, and .upsert() resolved against the surrogate primary key
-- rather than `id`, so every re-ingest appended a second copy instead of
-- updating. scripts/export_corpus.py counts 5 425 duplicated speeches, heavily
-- concentrated in the range this script trims: 2014 is 95 % duplicated, 2002
-- 76 %, 2015 75 %. That overlap is why the trim runs before the dedupe, and it
-- means the two steps together free less than their row counts suggest.

-- ---------------------------------------------------------------- 1. LOOK
-- Run alone. Keep the numbers: `dupes` is what step 3 removes, `pre_cutoff` what
-- step 2 removes, and they overlap.
select count(*)                                              as rows_total,
       count(distinct id)                                    as ids_distinct,
       count(*) - count(distinct id)                         as dupes,
       count(*) filter (where protocol_date <  '2015-06-01') as pre_cutoff,
       count(*) filter (where protocol_date >= '2015-06-01') as post_cutoff,
       min(protocol_date)                                    as oldest,
       max(protocol_date)                                    as newest,
       pg_size_pretty(pg_total_relation_size('public.speeches')) as speeches_size,
       pg_size_pretty(pg_database_size(current_database()))      as database_size
  from public.speeches;

-- ---------------------------------------------------------------- 2. TRIM
-- Every row before the cutoff is in the Parquet archive (38 202 of them, checked
-- against the archive on 2026-09-10), so this removes nothing that is not backed
-- up. Skip if step 1 reported pre_cutoff = 0. If the size has not fallen far
-- enough by step 5, raise the cutoff a year at a time and repeat: the model only
-- ever trained on 2015 onward, and the pre-2015 analytics are precomputed from
-- Parquet, not from here.
--
-- The read-write line belongs to the session, not to the project, and the SQL
-- editor may open a fresh session per run. Keep it at the top of every block
-- that writes, including a repeat of this one.
set session characteristics as transaction read write;

delete from public.speeches
      where protocol_date < '2015-06-01';

-- ---------------------------------------------------------------- 3. DEDUPE
-- Drop rows repeating an id, keeping the earliest. Safe because `id` is the
-- natural key (protocol + speaker slug + party), so rows sharing an id describe
-- the same speaker in the same protocol, and export_corpus.py has verified the
-- copies are byte-identical. This would NOT have been safe under the old
-- positional id, where one id could name two different speeches.
set session characteristics as transaction read write;

delete from public.speeches a
      using public.speeches b
      where a.id = b.id
        and a.supabase_id > b.supabase_id;

-- ---------------------------------------------------------------- 4. CONSTRAIN
-- The idempotency fix, and the reason this cannot happen again.
-- src/maktsprak_pipeline/db/speeches.py already calls
-- .upsert(rows, on_conflict="id"), but that is a no-op until a real unique index
-- exists for PostgREST to target. Fails while any duplicate remains, so it runs
-- after both deletes.
set session characteristics as transaction read write;

alter table public.speeches
    add constraint speeches_id_key unique (id);

-- ---------------------------------------------------------------- 5. RECLAIM
-- Postgres marks deleted rows dead rather than freeing them, so the reported
-- size does not fall on its own. Start with the plain form, which is what
-- Supabase's own guidance calls for: it takes no exclusive lock and needs no
-- extra disk.
set session characteristics as transaction read write;

vacuum (analyze) public.speeches;

-- Then re-run step 1 and compare. The dashboard can lag by up to an hour, so
-- trust `database_size` from the query over the usage page.
--
-- ONLY if the size has not moved, escalate on the single table:
--
--     vacuum full public.speeches;
--
-- Understand what that costs before running it. VACUUM FULL rewrites the table
-- under an ACCESS EXCLUSIVE lock, so nothing can read or write it meanwhile, and
-- it needs free disk roughly equal to the table it is rewriting. On a project
-- that is already at its cap that headroom is the thing in question, which is
-- why it is the fallback and not the first move.
