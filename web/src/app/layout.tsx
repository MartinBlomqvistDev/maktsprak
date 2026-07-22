import type { Metadata } from "next";
import { Archivo, Fragment_Mono } from "next/font/google";
import localFont from "next/font/local";
import "./globals.css";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";

// Display face, Redaction, drawn from scanned/degrading legal and political
// documents (SIL OFL, see src/app/fonts/Redaction-OFL.txt). Used sparingly for
// headlines: the typeface itself carries the project's thesis, legibility
// recovered from an official record.
const redaction = localFont({
  src: "./fonts/Redaction-Regular.woff2",
  variable: "--font-display",
  display: "swap",
  weight: "400",
});

// Body face, a quiet, confident grotesque. Deliberately not Inter/Space
// Grotesk (the default AI-generated pairing).
const archivo = Archivo({
  subsets: ["latin", "latin-ext"],
  variable: "--font-body",
  display: "swap",
});

// Data / transcript face, for Anf. speech markers, probabilities, and the
// "record" texture. A typewriter-report register rather than a dev-tool mono.
const fragmentMono = Fragment_Mono({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-data",
  display: "swap",
});

const DESCRIPTION =
  "En AI-läsning av den svenska riksdagens språk: vad avslöjar retoriken om vem som talar?";

export const metadata: Metadata = {
  // metadataBase has to be set or Next resolves og:image against localhost, which
  // silently ships a broken preview to every platform that scrapes the page.
  metadataBase: new URL("https://maktsprak.se"),
  title: "Maktspråk / Protokollet",
  description: DESCRIPTION,
  openGraph: {
    title: "Maktspråk / Protokollet",
    description: DESCRIPTION,
    url: "https://maktsprak.se",
    siteName: "Maktspråk",
    locale: "sv_SE",
    type: "website",
    images: [{ url: "/og.png", width: 2400, height: 1260, alt: "Maktspråk" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Maktspråk / Protokollet",
    description: DESCRIPTION,
    images: ["/og.png"],
  },
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
