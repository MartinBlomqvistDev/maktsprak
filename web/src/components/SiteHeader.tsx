import Link from "next/link";
import { ThemeToggle } from "./ThemeToggle";

const NAV = [
  { href: "/partierna", label: "Partierna" },
  { href: "/utveckling", label: "Utveckling" },
  { href: "/metod", label: "Metod" },
  { href: "/riktmarke", label: "Riktmärke" },
];

export function SiteHeader() {
  return (
    <header className="relative z-10 border-b border-line bg-paper-2/80 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="group flex items-baseline gap-2.5">
          <span className="font-display text-xl tracking-tight">
            Maktspråk
          </span>
          <span className="font-data hidden text-[10px] uppercase tracking-[0.2em] text-ink-3 sm:inline">
            / Protokollet
          </span>
        </Link>

        <nav className="hidden items-center gap-7 md:flex">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="font-data text-[11px] uppercase tracking-widest text-ink-2 transition-colors hover:text-accent"
            >
              {item.label}
            </Link>
          ))}
        </nav>

        <ThemeToggle />
      </div>
    </header>
  );
}
