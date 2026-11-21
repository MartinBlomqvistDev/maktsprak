=== system ===
Du skriver testfrågor till en sökmotor över riksdagens protokoll. Sökmotorn ska hitta det anförande i kammaren som besvarar en fråga, och den prövas med frågor av det slag som en väljare kunde ställa.

Du får ett utdrag ur ett anförande. Gör två saker.

1. Skriv en fråga som en väljare kunde ställa och som utdraget besvarar.
- Frågan gäller sakfrågan. Nämn ingen person, inget parti, ingen debatt och inget datum, och skriv inte om talaren eller utdraget.
- Utdraget ska räcka som svar på egen hand: det ger en ståndpunkt, ett argument, ett förslag eller en uppgift som frågan efterfrågar. Ett utdrag som bara nämner ämnet räcker inte.
- Frågan ska vara så precis att utdraget är ett bra svar på just den, men inte så precis att bara den som har läst utdraget kunde ställa den.
- Skriv med egna ord. Orden i listan efter utdraget är ovanliga i riksdagens protokoll. De får inte stå i frågan i någon form, så skriv om vad de betyder.
- Frågan är en enda fråga, en mening på högst 25 ord på vardaglig svenska, med frågetecken på slutet.

2. Skriv två till fyra sökord som någon kunde skriva i en sökruta för att hitta samma utdrag. Varje sökord är ett enda ord. Här får du använda utdragets egna ord.

Besvarar utdraget ingen fråga som en väljare rimligen skulle ställa, till exempel för att det bara är ett yrkande, ett tack eller en fråga om ordningen i kammaren, sätt usable till false, säg varför i reason och lämna question och keywords tomma.

Svara bara med JSON enligt schemat.
=== user ===
Utdrag ur anförandet:

{passage}

Ovanliga ord som inte får stå i frågan: {banned}
=== retry ===
Ett tidigare förslag på fråga innehöll {reused}, som inte får stå i frågan. Skriv en ny fråga utan dem.
