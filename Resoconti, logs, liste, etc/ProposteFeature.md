1. Age-Adjusted Performance
    Feature Name: `age_performance_index`
    Description: Adjust points or position based on age to assess performance relative to decline or improvement due to aging.
    Formula: age_performance_index = points|position / cyclist_age (maybe normalize by the average points for that age group?).

2. Weight-to-Height Ratio

    Feature Name: `weight_height_ratio`
    Description: Using weight and height of the cyclists, calculate their body mass index (BMI). Might be useful to gauge impact on races with a lot of climbing.
    Formula: weight_height_ratio = weight / height

3. Climbs per Kilometer

    Feature Name: `climbs_per_km`
    Description: Determine the climb density with climb_total and length.
    Formula: climbs_per_km = climb_total / length

4. Adjusted Points by Race Quality

    Feature Name: `quality_adjusted_points`
    Description: Calculate points adjusted by startlist_quality, measuring performance relative to the competition level.
    Formula: quality_adjusted_points = points * startlist_quality (maybe to normalize in some way)

5. Stamina

    Feature Name: `stamina_index`
    Description: Ratio of position or points to length, indicating stamina.
    Formula: stamina_index = points|position / length

6. Experience Level

    Feature Name: `experience_level`
    Description: Define an experience level based on total number of races.
    Formula: Create bins (e.g., novice, intermediate, experienced) based on total race count per cyclist.