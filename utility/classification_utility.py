import pandas as pd

# Useless/Duplicated
TO_REMOVE_COLS = [
    '_url_cyc', 'name_cyc'
]

TO_NOT_USE_COLS = [
    'average_speed', 'delta', 'time', 'time_seconds',
    'stamina_index', # computed using average_speed
]

# This columns needs to be recomputed for the classification becasuse 
# they change over time and using them as they are would be in some 
# way look at the future
TO_RECOMPUTE_COLS = [
    'experience_level', 'total_points', 
    'victories_by_points', 'avg_points_per_race', 
    'average_position', 'avg_speed_cyclist', 'mean_stamina_index'
]

# facts known before the race, that are "static" there are some 
# exceptions but they are out of the scope of the project and we don't
# have the tools to fix them (e.g. weight might change over time)
TO_KEEP_UNCHANGED_COLS = [
    # race related
    '_url_rac', 'name_rac', 'stage', 'stage_type', 'points', 
    'uci_points', 'length', 'climb_total', 'profile', 
    'startlist_quality', 'date', 'position', 'cyclist', 
    'cyclist_age_rac', 'is_tarmac', 'steepness', 'season', 
    'is_staged', 'race_country'
    # cyclist related
    'birth_year', 'weight', 'height', 'nationality', 'bmi',
    'race_count', 'cyclist_age_cyc',
    'age_performance_index',# non mi sembra chissa che sta feature
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
    return merged

def define_target(merged_fd:pd.DataFrame)-> pd.DataFrame:
    '''
    Compute and add the target column to the dataset 
    to simplify use during classification

    args:
        - merged_fd (pd.DataFrame): the dataframe to which add the target column

    returns:
        - pd.DataFrame: the modified dataframe
    '''
    merged_fd['target'] = merged_fd['position'].apply(lambda x: x<=20)
    return merged_fd