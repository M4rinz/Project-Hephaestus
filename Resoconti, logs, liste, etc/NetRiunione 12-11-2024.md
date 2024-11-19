# NetRiunione 12 Novembre 2024
## TO DO
Ecco le cose da fare:
- [ ] Nel notebook `data_exploration`, studio delle singole features originali del dataset (cyclists: name (non lo farei), races: startlist_quality)
    - [ ] Include l'outlier detection (basta individuare i valori fuori dai quantili...)
    - [ ] Possiamo direttamente confrontare lo studio sul dataset originale con quello sul dataset pulito alla fine
- [ ] Nel notebook `data_understanding_transformation`
    - [ ] Risiedono le features aggiuntive da noi individuate (la decisione sul se calcolare le features nuove dopo l'imputazione di quelle vecchie è in sospeso. Dipende dalle features)
    - [ ] Si studiano le nuove features aggiunte (alcune tramite scraping, alcune no) 
        - cyclists: 
            - bmi ok
            - experience_level
            - full history
            - points total
            - tot_season_attended
        - races:
            - stage type ok
            - time
            - date
            - steepness ok
            - season
            - is_staged
            - race_country
            - age_performance_index
            - quality_adjusted_points
            - stamina_index

    - [ ] Tutto questo studio si conclude con una pulizia generale del dataset. Così che possiamo esportare il `.csv` e usarlo per tasks successivi



    
