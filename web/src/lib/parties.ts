/**
 * The eight Riksdag parties. Mirrors PARTY_ORDER / VALID_PARTIES in the
 * Python pipeline's config.py, keep both in sync if party composition
 * ever changes.
 */
export type PartyCode = "V" | "MP" | "S" | "C" | "L" | "KD" | "M" | "SD";

export interface Party {
  code: PartyCode;
  name: string;
  colorVar: string; // CSS custom property name, e.g. "--party-s"
}

export const PARTY_ORDER: Party[] = [
  { code: "V", name: "Vänsterpartiet", colorVar: "--party-v" },
  { code: "MP", name: "Miljöpartiet", colorVar: "--party-mp" },
  { code: "S", name: "Socialdemokraterna", colorVar: "--party-s" },
  { code: "C", name: "Centerpartiet", colorVar: "--party-c" },
  { code: "L", name: "Liberalerna", colorVar: "--party-l" },
  { code: "KD", name: "Kristdemokraterna", colorVar: "--party-kd" },
  { code: "M", name: "Moderaterna", colorVar: "--party-m" },
  { code: "SD", name: "Sverigedemokraterna", colorVar: "--party-sd" },
];

export function partyByCode(code: string): Party | undefined {
  return PARTY_ORDER.find((p) => p.code === code);
}
