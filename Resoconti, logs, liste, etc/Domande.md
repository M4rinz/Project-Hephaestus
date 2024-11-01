# Domande (e risposte) ricevimento 29/11
Una lista di domande da fare a Mattia Setzu, scritte alla meno peggio
- Cosa è successo al Giro d'Italia del 2008, nella tappa 19? Sia i dati nel dataset sia quelli su procyclingstats sono fuori di testa. *Errore nel segnare i dati. Ignorare*
- Iztulia basque contries, 2001, stage 5b. Il Sig. Riccardo Forconi compare due volte, la seconda delle quali con un delta mostruoso. Inoltre, la sua 2a apparizione non compare su procyclingstats. In generale, ci sono ciclisti che appaiono due volte in una tappa, ma con posizioni diverse. Si elimina il duplicato? *Sì si leva il duplicato*
- Appaiono dei delta negativi. Possiamo eliminare la riga corrispondente senza chiederci troppe cose? E per i delta enormi? *Sì levare i delta negativi*
- Ma dopo la data, quel `\d{2}:\d{2}:\d{2}` è l'ora d'inizio della gara? *No. è la durata della corsa*
- Come mai i teams del dataset sono totalmente diversi da quelli di procyclingstats? *Boh*
- Ci sono gare che hanno sempre lo stesso percorso? Cioè, che ogni anno passano dalla stessa parte... Può avere senso dividere le gare tra quelle "fisse" e quelle "non fisse"? *sì*
- Ha senso come feature il n° di punti in carriera (da scraping, o sommando sul dataset corretto...)? Ha senso il numero di stagioni partecipate? *No, si sporca il dato*
- In generale, hanno senso le features che abbiamo proposto?
- *Va bene se una feature è molto banale, tipo che si ottiene leggendo il valore di una colonna*