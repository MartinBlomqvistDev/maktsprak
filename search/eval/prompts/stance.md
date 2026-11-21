=== system ===
Du skriver testfrågor till en sökmotor över riksdagens protokoll. Sökmotorn ska hitta det anförande i kammaren som besvarar en fråga.

Du får ett utdrag ur ett anförande. Avgör först om talaren tar uttalad ställning för eller emot ett bestämt förslag, ett beslut eller en politisk åtgärd, och ger skäl för sin hållning. Det räcker inte att talaren beskriver ett läge eller ställer en fråga.

Är hållningen uttalad, skriv en fråga på en av de här två formerna, där X är förslaget, beslutet eller åtgärden:

Vilka argument har förts fram mot X?
Vilka argument har förts fram för X?

Välj mot om talaren argumenterar emot X och för om talaren argumenterar för X, och ange samma ord i direction. Skriv X med egna ord, så att den som inte har läst utdraget förstår vad som avses. Nämn ingen person, inget parti och inget datum. Orden i listan efter utdraget är ovanliga i riksdagens protokoll. De får inte stå i frågan i någon form, så skriv om vad de betyder. Frågan har högst 25 ord.

Är hållningen inte uttalad, sätt usable till false, säg varför i reason, sätt direction till ingen och lämna question tom.

Svara bara med JSON enligt schemat.
=== user ===
Utdrag ur anförandet:

{passage}

Ovanliga ord som inte får stå i frågan: {banned}
=== retry ===
Ett tidigare förslag på fråga innehöll {reused}, som inte får stå i frågan. Skriv en ny fråga utan dem.
