import pandas as pd

# Useless/Duplicated
TO_REMOVE_COLS = [
    '_url_cyc', 'name_cyc', 'victories_by_points'
]

TO_NOT_USE_COLS = [
    'average_speed', 'delta', 'time', 'time_seconds',# known at the end of the race
    'stamina_index', # computed using average_speed
    'points', # points are assigned based on the position
    'position', # this is basically the target
    'target', 
    'age_performance_index',# based on age and points
]

# This columns needs to be recomputed for the classification because 
# they change over time and using them as they are would be in some 
# way look at the future
TO_RECOMPUTE_COLS = [
    'experience_level', 'total_points', 
    'avg_points_per_race', 
    'average_position', 'avg_speed_cyclist', 'mean_stamina_index',
    'race_count'
]

# facts known before the race, that are "static". There are some 
# exceptions but they are out of the scope of the project and we don't
# have the tools to fix them (e.g. weight might change over time)
TO_KEEP_UNCHANGED_COLS = [
    # race related
    '_url_rac', 'name_rac', 'stage', 'stage_type',  
    'uci_points', 'length', 'climb_total', 'profile', 
    'startlist_quality', 'date', 'cyclist', 
    'cyclist_age_rac', 'is_tarmac', 'steepness', 'season', 
    'is_staged', 'race_country'
    # cyclist related
    'birth_year', 'weight', 'height', 'nationality', 'bmi',
    'cyclist_age_cyc',
]

def get_merged_dataset(cyclists:str, races:str) -> pd.DataFrame:
    '''
    Get a version of a merged dataset on which all functions 
    defined in this file can be called this is supposed to help 
    avoid problems caused by naming columns in different ways 
    during the merge 
    
    args:
        - cyclists (str): the path to the cyclist's dataset
        - races (str): the path to the ravces' dataset

    returns:
        - pd.DataFrame: a dataframe containing information on 
        both cyclists and races
    '''
    cyc_df = pd.read_csv(cyclists)
    rac_df = pd.read_csv(races)
    merged = rac_df.merge(right=cyc_df, how='left', on='cyclist', suffixes=('_rac', '_cyc'))
    merged.drop(columns=TO_REMOVE_COLS)
    merged['points'] = merged['points'].fillna(0)
    return merged

def define_target(merged_df:pd.DataFrame)-> pd.DataFrame:
    '''
    Compute and add the target column to the dataset 
    to simplify use during classification

    args:
        - merged_fd (pd.DataFrame): the dataframe to which add the target column

    returns:
        - pd.DataFrame: the modified dataframe
    '''
    merged_df['target'] = merged_df['position'].apply(lambda x: x<=20)
    return merged_df

EXPERIENCE_LEVELS = ['beginner', 'developing', 'competitive', 'semi-pro', 'pro']
EXPERIENCE_BINS = [0, 15, 50, 100, 200, float('inf')]

def get_base_url(url: str) -> str:
    parts = url.split('/')
    return f"{parts[0]}/{parts[1]}"

def recompute_metrics(merged_df: pd.DataFrame,
                         total_points_D: int | None,
                         avg_points_per_race_D: float | None,
                         average_position_D: float | None,
                         avg_speed_cyclist_D: float | None,
                         mean_stamina_index: float | None,
                         missing_value_policy: str = 'mean'
                         ) -> pd.DataFrame:
    '''
    Compute the metrics of each cyclist at a given race.
    Default values are used to initialize the metrics. This is needed for when
    a cyclist appears for the first time in the dataset.

    NOTE: no deafult default values are provided because they should likely change based on the model used

    args:
        - merged_df (pd.DataFrame): the dataframe on which recompute the metrics
        - total_points_D (int): the default value for the total points
        - avg_points_per_race_D (float): the default value for the average points per
        - average_position_D (float): the default value for the average position
        - avg_speed_cyclist_D (float): the default value for the average speed
        - mean_stamina_index (float): the default value for the mean stamina index
        - missing_value_policy {drop|mean} (str): the policy to handle missing values. 

    returns:
        - pd.DataFrame: the updated dataframe (input dataframe is probably not modified (python is weird))
    '''
    # Complexity analysis:
    # O(NlogN) to order by date
    # O(N) scan to create an hash table with all names of cyclists (it will be a generic hash Ferragina would be disappointed)
    # O(N) scan and compute the value

    # Sort the dataset by date
    merged_df = merged_df.sort_values(by='date').reset_index(drop=True) # Per Dina, qual giubilo scoprire che non funzionava nulla perche' dovevo resettare gli indici
    # Initialize hash table to store cumulative metrics for each cyclist
    cyclist_metrics = {}
    
    tot_iterations = len(merged_df)
    prints = 0
    each = tot_iterations // 100
    # Iterate over the rows to compute the metrics
    for index, row in merged_df.iterrows():
        if prints >= each:
            # print is a slow operation
            prints = 0
            print(f'{((index + 1)/tot_iterations)*100:.2f}%  ', end='\r')
        else:
            prints += 1

        cyclist = row['cyclist']
        
        if cyclist not in cyclist_metrics:
            # Initialize metrics for the cyclist
            cyclist_metrics[cyclist] = {
                'total_points': 0,
                'total_races': 0,
                'total_positions': 0,
                'total_speed': 0,
                'total_stamina': 0
            }
            # the first time we see a cyclist we need to assign the default values
            merged_df.at[index, 'total_points'] = total_points_D
            merged_df.at[index, 'avg_points_per_race'] = avg_points_per_race_D
            merged_df.at[index, 'average_position'] = average_position_D
            merged_df.at[index, 'avg_speed_cyclist'] = avg_speed_cyclist_D
            merged_df.at[index, 'mean_stamina_index'] = mean_stamina_index
            merged_df.at[index, 'race_count'] = 0
            merged_df.at[index, 'experience_level'] = EXPERIENCE_LEVELS[0]
        else:
            # we already have metrics for the cyclist. Assign computed values to the dataframe
            merged_df.at[index, 'total_points'] = cyclist_metrics[cyclist]['total_points']
            merged_df.at[index, 'avg_points_per_race'] = cyclist_metrics[cyclist]['total_points'] / cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'average_position'] = cyclist_metrics[cyclist]['total_positions'] / cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'avg_speed_cyclist'] = cyclist_metrics[cyclist]['total_speed'] / cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'mean_stamina_index'] = cyclist_metrics[cyclist]['total_stamina'] / cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'race_count'] = cyclist_metrics[cyclist]['total_races']
            # Compute experience level
            for i in range(len(EXPERIENCE_BINS)):
                if cyclist_metrics[cyclist]['total_races'] >= EXPERIENCE_BINS[i] and cyclist_metrics[cyclist]['total_races'] < EXPERIENCE_BINS[i + 1]:
                    merged_df.at[index, 'experience_level'] = EXPERIENCE_LEVELS[i]
                    break
                    
        # Update metrics
        # This is after the updating of the cyclist to avoid using the current race in the computation
        if pd.isna(row['stamina_index']):
            if missing_value_policy == 'drop': # drop the row
                raise NotImplementedError("This is not implemented yet XD")
                merged_df.drop(index, inplace=True)
                continue
            elif missing_value_policy == 'mean': # use the mean of points computed so far
                if cyclist_metrics[cyclist]['total_races'] == 0: # super rare case
                    row['stamina_index'] = 0
                else:
                    row['stamina_index'] = cyclist_metrics[cyclist]['total_stamina'] / cyclist_metrics[cyclist]['total_races']

        cyclist_metrics[cyclist]['total_points'] += row['points']
        cyclist_metrics[cyclist]['total_stamina'] += row['stamina_index']
        cyclist_metrics[cyclist]['total_races'] += 1
        cyclist_metrics[cyclist]['total_positions'] += row['position']
        cyclist_metrics[cyclist]['total_speed'] += row['average_speed']

    print('100.00%  ')
    return merged_df