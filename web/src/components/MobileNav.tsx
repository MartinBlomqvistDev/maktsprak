"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

interface NavItem {
  href: string;
  label: string;
}

/**
 * The navigation for viewports below `md`, where the inline nav is hidden.
 *
 * Client-side because the panel needs open state; the desktop nav stays a
 * server component. Closes on navigation, since Next keeps the layout mounted
 * across route changes and the panel would otherwise stay open over the new
 * page.
 */
export function MobileNav({ items }: { items: NavItem[] }) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <div className="md:hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="mobile-nav"
        className="font-data flex items-center gap-2 rounded-full border border-line-2 px-3 py-1.5 text-[11px] uppercase tracking-widest text-ink-2 transition-colors hover:border-accent hover:text-accent"
      >
        {open ? "Stäng" : "Meny"}
      </button>

      {open && (
        <nav
          id="mobile-nav"
          className="absolute inset-x-0 top-full z-20 border-b border-line bg-paper-2 px-6"
        >
          {items.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={pathname === item.href ? "page" : undefined}
              className="font-data block border-b border-line py-3.5 text-[11px] uppercase tracking-widest text-ink-2 transition-colors last:border-b-0 hover:text-accent aria-[current=page]:text-accent"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      )}
    </div>
  );
}
