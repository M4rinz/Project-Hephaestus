1. Age-Adjusted Performance
    - Feature Name: `age_performance_index`
    - Description: Adjust points or position based on age to assess performance relative to decline or improvement due to aging.
    - Formula: age_performance_index = points|position / cyclist_age (maybe normalize by the average points for that age group?).
    - Notes:

2. Weight-to-Height Ratio

    - Feature Name: `weight_height_ratio`
    - Description: Using weight and height of the cyclists, calculate their body mass index (BMI). Might be useful to gauge impact on races with a lot of climbing.
    - Formula: weight_height_ratio = weight / height
    - Notes:

3. Climbs per Kilometer

    - Feature Name: `climbs_per_km`
    - Description: Determine the climb density with climb_total and length.
    - Formula: climbs_per_km = climb_total / length
    - Notes:

4. Adjusted Points by Race Quality

    - Feature Name: `quality_adjusted_points`
    - Description: Calculate points adjusted by startlist_quality, measuring performance relative to the competition level.
    - Formula: quality_adjusted_points = points * startlist_quality (maybe to normalize in some way)
    - Notes:

5. Stamina

    - Feature Name: `stamina_index`
    - Description: Ratio of position or points to length, indicating stamina.
    - Formula: stamina_index = points|position / length
    - Notes:

6. Experience Level

    - Feature Name: `experience_level`
    - Description: Define an experience level based on total number of races.
    - Formula: Create bins (e.g., novice, intermediate, experienced) based on total race count per cyclist.
    - Notes:

7. Type of stage

    - Feature Name: `stage_type`
    - Description: The type of stage. It can be ITT (Individual Time Trial), TTT (Team TT), RR (Road Race)
    - Formula: pcs.Stage(f'race/{url}).stage_type()
    - Notes: The feature is obtained via scraping, and is included in the new version of the races dataframe

8. Season of the competition

    - Feature: `season`
    - Description: The season in which the race takes place (summer/winter/...). Not the year 
    - Formula: binning based on the date of the competition.
    - Notes:

9. Duration of the stage

    - Feature: `duration`
    - Description: How much time it took for the given cyclist to complete the stage in question
    - Formula: Just separate the time that appears in the "date" column.
    - Notes: This feature is related to the delta, but we discovered that is not as random as one would have thought...

10. Whether the track is fixed of not

    - Feature: `is_track_fixed`
    - Description: Some competitions (e.g. the Tour de France) are designed to take place in different locations each year. Some others (e.g. Paris-Roubaix) have always taken place in the same roads
    - Formula: This is domain knowledge that needs to be injected in the dataset
    - Notes: A small problem could be that maybe the track has changed at some point in history

11. Whether the competition is at stages or not

    - Feature: `is_staged`
    - Description: Some races are Grand Tours and take place in stages, some others don't (es. Tour de France vs Ronde van Vlaanderen)
    - Formula: Domain knowledge, or seeing if the `_url` ends with "results"
    - Notes: Unsure whether is a true feature or not
