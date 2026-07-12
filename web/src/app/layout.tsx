import type { Metadata } from "next";
import { Archivo, Fragment_Mono } from "next/font/google";
import localFont from "next/font/local";
import "./globals.css";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";

// Display face — Redaction, drawn from scanned/degrading legal and political
// documents (SIL OFL, see src/app/fonts/Redaction-OFL.txt). Used sparingly for
// headlines: the typeface itself carries the project's thesis — legibility
// recovered from an official record.
const redaction = localFont({
  src: "./fonts/Redaction-Regular.woff2",
  variable: "--font-display",
  display: "swap",
  weight: "400",
});

// Body face — a quiet, confident grotesque. Deliberately not Inter/Space
// Grotesk (the default AI-generated pairing).
const archivo = Archivo({
  subsets: ["latin", "latin-ext"],
  variable: "--font-body",
  display: "swap",
});

// Data / transcript face — for Anf. speech markers, probabilities, and the
// "record" texture. A typewriter-report register rather than a dev-tool mono.
const fragmentMono = Fragment_Mono({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-data",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MaktspråkAI — Protokollet",
  description:
    "En AI-läsning av den svenska riksdagens språk: vad avslöjar retoriken om vem som talar?",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="sv"
      className={`${redaction.variable} ${archivo.variable} ${fragmentMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col bg-paper text-ink font-body">
        <SiteHeader />
        {children}
        <SiteFooter />
      </body>
    </html>
  );
}
