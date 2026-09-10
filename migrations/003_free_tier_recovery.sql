-- Recover the project from the Free Plan 0.5 GB cap and stop it refilling.
-- Run in the Supabase SQL editor, block by block, reading the output as you go.
--
-- Context: the project hit 0.507 GB on 2026-09-10 and Postgres went read-only,
-- so writes fail with "cannot execute INSERT in a read-only transaction" until
-- usage is back under 95 % of the cap. Reads are unaffected, and nothing public
-- depends on this database: maktsprak.se serves precomputed JSON, the predict
-- demo runs on Cloud Run, and `data/parquet/speeches_full.parquet` is the source
-- of truth (75 148 rows, all ids unique, 2002-09-30 to 2026-06-23). Supabase is
-- the weekly-ETL landing zone.
--
-- Two causes, fixed in that order below: duplicate rows from an upsert that was
-- silently an insert, and the pre-cutoff backfill that Parquet already holds.

-- 0. Lift read-only for THIS session. Without it even the deletes are refused.
--    It applies to the current session only; it does not change the project.
set session characteristics as transaction read write;

-- 1. Look before deleting. Run on its own and keep the numbers.
--    `dupes` is how many rows step 2 removes; `pre_cutoff` is what step 4 removes.
select count(*)                                             as rows_total,
       count(distinct id)                                   as ids_distinct,
       count(*) - count(distinct id)                        as dupes,
       count(*) filter (where protocol_date <  '2015-06-01') as pre_cutoff,
       count(*) filter (where protocol_date >= '2015-06-01') as post_cutoff,
       min(protocol_date)                                   as oldest,
       max(protocol_date)                                   as newest
  from public.speeches;

-- 2. Drop rows repeating an id, keeping the earliest.
--    Safe because `id` is the natural key (protocol + speaker slug + party), so
--    rows sharing an id describe the same speaker in the same protocol. This
--    would NOT have been safe under the old positional id, where one id could
--    name two different speeches.
delete from public.speeches a
      using public.speeches b
      where a.id = b.id
        and a.supabase_id > b.supabase_id;

-- 3. The idempotency fix. `.upsert(rows, on_conflict="id")` is already in
--    src/maktsprak_pipeline/db/speeches.py but is a no-op until this exists:
--    PostgREST needs a real unique index to target. Fails while duplicates
--    remain, which is why it runs after step 2.
alter table public.speeches
    add constraint speeches_id_key unique (id);

-- 4. Trim the backfill Supabase does not need. Every row before the cutoff is in
--    the Parquet archive (38 202 rows below 2015-06-01, verified 2026-09-10), so
--    this removes nothing that is not backed up. Skip this block if step 1
--    reported pre_cutoff = 0, and raise the cutoff if step 5 still shows the
--    project above the cap: each later year removes more.
delete from public.speeches
      where protocol_date < '2015-06-01';

-- 5. Reclaim the space. Postgres marks deleted rows dead rather than freeing
--    them, so without this the dashboard keeps reporting the old size and the
--    project stays read-only. Expect it to take a while on a table this size.
vacuum (full, analyze) public.speeches;

-- 6. Confirm. Compare against the dashboard, which can lag by up to an hour.
select pg_size_pretty(pg_total_relation_size('public.speeches')) as speeches_size,
       pg_size_pretty(pg_database_size(current_database()))      as database_size,
       count(*)                                                  as rows_remaining
  from public.speeches;
