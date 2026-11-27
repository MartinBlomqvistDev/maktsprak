export function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-line bg-paper-2">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-6 pb-3 pt-8 text-xs text-ink-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="label">
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
      <p className="mx-auto max-w-6xl px-6 pb-8 text-xs text-ink-3">
        Källa: Sveriges riksdag. Riksdagen står inte bakom Maktspråk.
      </p>
    </footer>
  );
}
