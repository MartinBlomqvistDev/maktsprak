import { ComingNext } from "@/components/ComingNext";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Riktmärke · Maktspråk / Protokollet",
};

export default function RiktmarkePage() {
  return (
    <ComingNext
      protNr={5}
      section="MODELLJÄMFÖRELSE"
      title="Hur bra är en finjusterad 110M-parametersmodell mot ett frontier-LLM?"
      lede="Samma talar-oberoende testset körs mot GPT, Claude, Gemini och den egna KB-BERT-modellen: träffsäkerhet, kalibrering, kostnad och känslighet för hur frågan ställs."
      planned={[
        "Samma 5 887 osedda anföranden, samma åtta partier, för varje modell",
        "Kalibrering: är modellen lika säker som den borde vara?",
        "Kostnad per 1000 klassificeringar: en 110M-modell mot API-anrop",
      ]}
    />
  );
}
