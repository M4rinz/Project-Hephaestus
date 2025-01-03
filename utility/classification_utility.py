from datetime import datetime
import numpy as np
import pandas as pd

# Useless/Duplicated
TO_REMOVE_COLS = [
    '_url_cyc', 'name_cyc', 'victories_by_points', #'uci_points',
    'normalized_length', 'normalized_quality',
    'normalized_steepness', 'normalized_time',
    'norm_points',
    'quality_adjusted_points' # we agreed to delete this
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
    'race_count',
    'elapsed_from_last',# new feature (elapsed time in days from the previous race)
    'average_position_var', # new feature (variance of the position)
]

# facts known before the race, that are "static". There are some 
# exceptions but they are out of the scope of the project and we don't
# have the tools to fix them (e.g. weight might change over time)
TO_KEEP_UNCHANGED_COLS = [
    # race related
    '_url_rac', 'name_rac', 'stage', 'stage_type',  
    'length', 'climb_total', 'profile', 
    'startlist_quality', 'date', 'cyclist', 
    'cyclist_age_rac', 'is_tarmac', 'steepness', 'season', 
    'is_staged', 'race_country'
    'partecipants_number'# new feature (number of partecipants in the race) Is it really to keep unchanged?
    # cyclist related
    'birth_year', 'weight', 'height', 'nationality', 'bmi',
    'cyclist_age_cyc',
]

def get_merged_dataset(cyclists:str, 
                        races:str,
                        is_RNN:bool=False) -> pd.DataFrame:
    '''
    Get a version of a merged dataset on which all functions 
    defined in this file can be called this is supposed to help 
    avoid problems caused by naming columns in different ways 
    during the merge 
    
    args:
        - cyclists (str): the path to the cyclist's dataset
        - races (str): the path to the ravces' dataset
        - is_RNN (bool): if true the dataset will contain the `uci_points` column. Defaults to False
    returns:
        - pd.DataFrame: a dataframe containing information on 
        both cyclists and races
    '''
    cyc_df = pd.read_csv(cyclists)
    rac_df = pd.read_csv(races)
    merged = rac_df.merge(right=cyc_df, how='left', left_on='cyclist', right_on='_url', suffixes=('_rac', '_cyc'))
    merged.drop(columns=TO_REMOVE_COLS, inplace=True)
    merged['points'] = merged['points'].fillna(0)
    if is_RNN:
        merged['uci_points'] = merged['uci_points'].fillna(0)
    else:
        merged.drop(columns=['uci_points'], inplace=True)

    merged['partecipants_number'] = merged['_url_rac'].map(merged.groupby('_url_rac').size())
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
                         mean_stamina_index_D: float | None,
                         elapsed_from_last_race_D: int | None,
                         average_position_var_D: float | None,
                         missing_value_policy: str = 'mean',
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
        - mean_stamina_index_D (float): the default value for the mean stamina index
        - elapsed_from_last_race_D (int): the default value for the elapsed time from the last race (in days)
        - average_position_var_D (float): the default value for the average position variance
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
        cyclist = row['cyclist_rac']
        
        if cyclist not in cyclist_metrics:
            # Initialize metrics for the cyclist
            cyclist_metrics[cyclist] = {
                'total_points': 0,
                'total_races': 0,
                'total_positions': 0,
                'total_position_squared': 0,
                'total_speed': 0,
                'total_stamina': 0,
                'last_race_date': None
            }
            # the first time we see a cyclist we need to assign the default values
            merged_df.at[index, 'total_points'] = total_points_D
            merged_df.at[index, 'avg_points_per_race'] = avg_points_per_race_D
            merged_df.at[index, 'average_position'] = average_position_D
            merged_df.at[index, 'avg_speed_cyclist'] = avg_speed_cyclist_D
            merged_df.at[index, 'mean_stamina_index'] = mean_stamina_index_D
            merged_df.at[index, 'race_count'] = 0
            merged_df.at[index, 'elapsed_from_last'] = elapsed_from_last_race_D
            merged_df.at[index, 'average_position_var'] = average_position_var_D
            merged_df.at[index, 'experience_level'] = EXPERIENCE_LEVELS[0]
        else:
            # we already have metrics for the cyclist. Assign computed values to the dataframe
            merged_df.at[index, 'total_points'] = cyclist_metrics[cyclist]['total_points']
            merged_df.at[index, 'avg_points_per_race'] = cyclist_metrics[cyclist]['total_points'] / cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'average_position'] = cyclist_metrics[cyclist]['total_positions'] / cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'avg_speed_cyclist'] = cyclist_metrics[cyclist]['total_speed'] / cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'mean_stamina_index'] = cyclist_metrics[cyclist]['total_stamina'] / cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'race_count'] = cyclist_metrics[cyclist]['total_races']
            merged_df.at[index, 'elapsed_from_last'] = (datetime.strptime(row['date'], "%Y-%m-%d") - datetime.strptime(cyclist_metrics[cyclist]['last_race_date'], "%Y-%m-%d")).days
            merged_df.at[index, 'average_position_var'] = ((cyclist_metrics[cyclist]['total_position_squared'] + 1e-7) / (cyclist_metrics[cyclist]['total_races'] + 1e-7)) - (merged_df.at[index, 'average_position'] ** 2)
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
        cyclist_metrics[cyclist]['last_race_date'] = row['date']
        cyclist_metrics[cyclist]['total_position_squared'] += row['position'] ** 2

    print('100.00%  ')
    return merged_df

def make_dataset_for_classification(races_df, 
                                    cyclists_df, 
                                    avg_points_per_race_D=-1,
                                    average_position_D=-1, 
                                    avg_speed_cyclist_D=-1, 
                                    mean_stamina_index_D=-1, 
                                    total_points_D=-1, 
                                    elapsed_from_last_race_D=-1, 
                                    average_position_var_D=-1, 
                                    missing_value_policy='mean', 
                                    make_home_game=True,
                                    make_stage_type=False, 
                                    # make_race_participants=False
                                    ):
    full_df = get_merged_dataset(cyclists_df, races_df)
    full_df = recompute_metrics(full_df,
                  avg_points_per_race_D=avg_points_per_race_D,
                  average_position_D=average_position_D,
                  avg_speed_cyclist_D=avg_speed_cyclist_D,
                  mean_stamina_index_D=mean_stamina_index_D,
                  total_points_D=total_points_D,
                  elapsed_from_last_race_D=elapsed_from_last_race_D,
                  average_position_var_D=average_position_var_D,
                  missing_value_policy=missing_value_policy)
    full_df = define_target(full_df)
    if make_home_game: full_df['home_game'] = full_df.apply(lambda x: 1 if x['race_country'] == x['nationality'] else 0, axis=1)
    if make_stage_type: full_df['stage_type'] = full_df.apply(lambda x: 1 if x['stage_type'] == 'ITT' else 0, axis=1)
    # if make_race_participants: don't need this anymore because someone already did the feature in another place meow
    #     race_participants = full_df['_url_rac'].value_counts()
    #     full_df['total_participants'] = full_df['_url_rac'].map(race_participants)
    return full_df

def make_dataset_for_RNN_classification(races_path:str,
                                        cyclists_path:str) -> pd.DataFrame:
    """Creates the dataset for the RNN classification.
    Merges the cyclist and races dataframes, creates the target column,
    converts the time-related columns to time datatypes, removes 
    the columns in TO_RECOMPUTE_COLS that are not needed for the RNNs,
    then finally groups the entries by cyclist and sorts them by date.

    Args:
        races_path (str): path to the races CSV data
        cyclists_path (str): path to the cyclists CSV data

    Returns:
        pd.DataFrame: the dataset for the RNN classification
    """
    full_df = get_merged_dataset(cyclists=cyclists_path, 
                                    races=races_path, 
                                    is_RNN=True)
    # define the target
    full_df = define_target(full_df)
    # convert datatypes
    full_df['date'] = full_df['date'].astype('datetime64[ns]')
    full_df['delta'] = full_df['delta'].astype('timedelta64[s]')
    full_df['time'] = full_df['time'].astype('timedelta64[s]')
    # For the RNNs the assumption is that we don't need the recomputed
    # values in the TO_RECOMPUTE_COLS, as the RNN will have direct
    # access to the (past) values used to obtain such recomputed values
    full_df.drop(columns=set(TO_RECOMPUTE_COLS) - {'elapsed_from_last'}, inplace=True)
    # Let's remove some columns now
    full_df.drop(columns=['time_seconds', 'cyclist_age_cyc'], inplace=True)
    # Now is time to sort the values
    ## Sort by date all the entries of a given cyclist, and put them together
    sorted_by_date = full_df.groupby('cyclist').apply(lambda x: x.sort_values(by='date'), include_groups=False)
    
    return sorted_by_date

def shift_columns_for_RNN(
        sorted_by_date:pd.DataFrame, 
) -> pd.DataFrame:
    # Conversely, we can use the values in the TO_NOT_USE_COLS,
    # provided that we don't use the current values for the current prediction,
    # but only the past values. Thus we just shift the column
    # Copy the df to avoid the SettingWithCopyWarning
    new_df = sorted_by_date.copy()
    # Shift the columns in TO_NOT_USE_COLS for each cyclist
    for col in set(TO_NOT_USE_COLS) - {'time_seconds'}:
        new_df.loc[:,f"{col}_shifted"] = new_df.groupby('cyclist')[col].shift(1)
    return new_df

def get_train_val_split(full_df, val_size=0.2, random_state=42):
    '''
    Split the dataset into a training and validation set

    args:
        - full_df (pd.DataFrame): the dataframe to split
        - val_size (float): the size of the validation set

    returns:
        - pd.DataFrame, pd.DataFrame: the training and validation set
    '''
    train_df = full_df.sample(frac=1 - val_size, random_state=random_state)
    val_df = full_df.drop(train_df.index)
    return train_df, val_df

def get_data_split(df):
    '''
    The split proposed in the XGB notebook is reused here for compatibility purposes.
    Should we decide to modify it, just remove the variable and the code proposed.
    '''
    df_tr = df[(df['date'] >= '1996-01-01') & (df['date'] < '2019-01-01')]
    df_v = df[(df['date'] >= '2019-01-01') & (df['date'] < '2022-01-01')]                                    
    df_ts = df[df['date'] >= '2022-01-01']

    return df_tr, df_v, df_ts

def split_features_target(df):
    '''
    ugly to be here, right? better than reusing and declaring it 20 times
    '''
    y = df['target']
    X = df.drop(columns=['target'])
    return X, y