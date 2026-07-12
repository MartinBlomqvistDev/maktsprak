import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

/**
 * Server-side Supabase client. `null` when credentials aren't configured yet
 * (e.g. before `.env.local` is filled in) — callers must handle that case
 * rather than crashing the page. See F-03 in the codebase audit: never
 * hardcode a corpus size in copy, always read it live.
 */
export const supabase = url && anonKey ? createClient(url, anonKey) : null;

export async function fetchSpeechesCount(): Promise<number | null> {
  if (!supabase) return null;
  const { count, error } = await supabase
    .from("speeches")
    .select("id", { count: "exact", head: true });
  if (error) {
    console.error("fetchSpeechesCount failed:", error.message);
    return null;
  }
  return count ?? null;
}
