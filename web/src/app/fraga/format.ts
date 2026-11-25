/**
 * Swedish number and date formatting for Fråga protokollet.
 *
 * Written by hand rather than with Intl, so the server's render and the
 * browser's hydration produce the same characters whatever ICU data each side
 * ships. Thousands are grouped with a no-break space, decimals take a comma.
 */

const NBSP = "\u00a0";

const MONTHS = [
  "januari",
  "februari",
  "mars",
  "april",
  "maj",
  "juni",
  "juli",
  "augusti",
  "september",
  "oktober",
  "november",
  "december",
];

function group(digits: string): string {
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, NBSP);
}

/** An integer with Swedish digit grouping: 301358 -> "301 358". */
export function int(value: number): string {
  const rounded = Math.round(value);
  const sign = rounded < 0 ? "\u2212" : "";
  return sign + group(String(Math.abs(rounded)));
}

/** A decimal with a comma and a fixed number of decimals: 0.4667, 3 -> "0,467". */
export function dec(value: number, digits: number): string {
  const fixed = Math.abs(value).toFixed(digits);
  const [whole, fraction] = fixed.split(".");
  const sign = value < 0 && Number(fixed) !== 0 ? "\u2212" : "";
  return sign + group(whole) + (fraction ? "," + fraction : "");
}

/** A signed decimal, with a plus sign when positive: 0.2 -> "+0,200". */
export function signed(value: number, digits: number): string {
  return (value > 0 ? "+" : "") + dec(value, digits);
}

/** A share as a percentage: 0.2246, 1 -> "22,5 %". */
export function pct(share: number, digits = 0): string {
  return `${dec(share * 100, digits)}${NBSP}%`;
}

/** An interval of two decimals: [0.139, 0.261] -> "0,139 till 0,261". */
export function interval(low: number, high: number, digits: number): string {
  return `${dec(low, digits)} till ${dec(high, digits)}`;
}

/** An ISO date in Swedish: "2025-09-04" -> "4 september 2025". */
export function longDate(iso: string): string {
  const [year, month, day] = iso.split("-").map(Number);
  return `${day} ${MONTHS[month - 1]} ${year}`;
}

/** Milliseconds as seconds with one decimal: 1250 -> "1,3 s". */
export function seconds(ms: number): string {
  return `${dec(ms / 1000, 1)}${NBSP}s`;
}

const WORDS = ["noll", "en", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio", "tio", "elva", "tolv"];

/**
 * A count in words, as Swedish prose writes small numbers: 4 -> "fyra". The
 * word for 1 is the common gender's "en". From 13 the digits are written.
 */
export function words(count: number): string {
  return Number.isInteger(count) && count >= 0 && count < WORDS.length
    ? WORDS[count]
    : int(count);
}

/** Fold case and diacritics, for matching a typed name against the speaker list. */
export function fold(text: string): string {
  return text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}
