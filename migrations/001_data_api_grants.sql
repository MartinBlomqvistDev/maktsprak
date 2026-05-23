-- Supabase Data API access: explicit GRANTs required from October 30, 2026
-- Run once in the Supabase SQL editor for this project.
--
-- Context: Supabase change announced May 2026 — new projects from May 30 and
-- all existing projects from October 30 will no longer expose public schema
-- tables to the Data API without explicit grants. The anon key (read client)
-- needs SELECT on both tables; write operations use service_role which bypasses
-- RLS and does not require explicit grants.

grant select
    on public.speeches
    to anon, authenticated;

grant select
    on public.tweets
    to anon, authenticated;
