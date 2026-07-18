export function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-line bg-paper-2">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-6 py-8 text-xs text-ink-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="font-data uppercase tracking-widest">
          Maktspråk / Protokollet
        </p>
        <p>
          Byggt av{" "}
          <a
            href="https://www.linkedin.com/in/martin-blomqvist"
            className="text-ink-2 underline underline-offset-2 hover:text-accent"
          >
            Martin Blomqvist
          </a>{" "}
          · källkod på{" "}
          <a
            href="https://github.com/MartinBlomqvistDev/maktsprak"
            className="text-ink-2 underline underline-offset-2 hover:text-accent"
          >
            GitHub
          </a>
        </p>
      </div>
    </footer>
  );
}
