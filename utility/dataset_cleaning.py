import pandas
import numpy as np

def delta_based_dataset_cleaning(dataset:pandas.DataFrame) -> pandas.DataFrame:
    """Cleans the dataset as done in the delta notebook.
    Precisely, only races with nonnegative deltas that are in order are kept.
    Based on the exploration (i.e. comparison with the deltas scraped on the web), 
    some races are manually fixed. 
    Also rows with `delta`>17000 are removed (3 rows in total).
    Note that the duplicated cyclists inside the races aren't removed.

    Args:
        dataset (pandas.DataFrame): the original dataset with the races, not scraped

    Returns:
        pandas.DataFrame: copy of dataset with the rows removed
    """
    # Not sure if the variable is passed by value or by reference...
    races_df_copy = dataset.copy()

    
    # urls of the stage for which delta is not ordered
    stages_monotonic_delta = races_df_copy.groupby('_url')['delta'].apply(lambda delta_series: delta_series.is_monotonic_increasing)
    urls_stages_non_monotonic_delta = [stage_url for stage_url, truth in stages_monotonic_delta.items() if not truth]
    urls_stages_negative_delta = races_df_copy[races_df_copy['delta']<0]['_url'].unique()
    
    fixed_races = []

    def are_problem_fixed(url:str) -> bool:
        # Are now the deltas ordered?
        c1 = races_df_copy[races_df_copy['_url'] == url]['delta'].is_monotonic_increasing
        # Are now the deltas nonnegative?
        c2 = np.all(races_df_copy[races_df_copy['_url'] == url]['delta'] >= 0)
        return c1 and c2
    
    # Remove Leonardo Bertagnolli from Stage 12 of 2003 Tour de France
    races_df_copy = races_df_copy[
            ~(races_df_copy['cyclist'] == 'leonardo-bertagnolli') | 
            ~(races_df_copy['_url'] == 'tour-de-france/2003/stage-12')
        ]
    if are_problem_fixed('tour-de-france/2003/stage-12'):
        fixed_races.append('tour-de-france/2003/stage-12')

    # Remove Simone Ponzi from GP Quebec of 2011
    races_df_copy = races_df_copy[
            ~(races_df_copy['cyclist'] == 'simone-ponzi') | 
            ~(races_df_copy['_url'] == 'gp-quebec/2011/result')
        ]
    if are_problem_fixed('gp-quebec/2011/result'):
        fixed_races.append('gp-quebec/2011/result')

    # Remove Moreno Moser and Carlos Bentancur from Stage 2 of 2015 Tirreno-Adriatico
    races_df_copy = races_df_copy[
            ~(
                (races_df_copy['cyclist'] == 'moreno-moser') | 
                (races_df_copy['cyclist'] == 'carlos-betancur')
            ) | 
            ~(races_df_copy['_url'] == 'tirreno-adriatico/2015/stage-2')
        ]
    if are_problem_fixed('tirreno-adriatico/2015/stage-2'):
        fixed_races.append('tirreno-adriatico/2015/stage-2')

    # Remove the duplicate result for Arsenio Gonzalez from Stage 15 of 1995's Tour de France
    races_df_copy.drop([302540], inplace=True)
    if are_problem_fixed('tour-de-france/1995/stage-15'):
        fixed_races.append('tour-de-france/1995/stage-15')

    # Remove the duplicate result for Ignacio Garcia from Stage 11 of 1993's Vuelta a Espana
    races_df_copy.drop([492194], inplace=True)
    if are_problem_fixed('vuelta-a-espana/1993/stage-11'):
        fixed_races.append('vuelta-a-espana/1993/stage-11')

    # Remove Eric Vanderaerden from Stage 14 of 1983's Vuelta a Espana
    races_df_copy = races_df_copy[
            ~(races_df_copy['cyclist'] == 'eric-vanderaerden') | 
            ~(races_df_copy['_url'] == 'vuelta-a-espana/1983/stage-14')
        ]
    if are_problem_fixed('vuelta-a-espana/1983/stage-14'):
        fixed_races.append('vuelta-a-espana/1983/stage-14')

    urls_guilty = np.unique(np.concatenate([urls_stages_non_monotonic_delta, urls_stages_negative_delta]))
    urls_races_to_drop = [url for url in urls_guilty if url not in fixed_races]

    # Drop the problematic races that haven't been fixed
    races_df_copy = races_df_copy[~races_df_copy['_url'].isin(urls_races_to_drop)]

    # Remove three outliers that remain there despite all this filtering
    races_df_copy = races_df_copy[races_df_copy['delta'] < 17000]

    return races_df_copy


def speed_based_dataset_cleaning(dataset:pandas.DataFrame, 
                                  speed_min:float = 3,
                                  speed_max:float = 60,
                                  keep_tdf:bool = True) -> pandas.DataFrame:
    """Cleans the dataset by removing the races with suspicious values for `average_speed`.
    It removes all the rows with `average_speed` outside the range [`speed_min`, `speed_max`].
    If `keep_tdf` is `True`, the 12th stage of the 2003 Tour de France 
    (whose values of `average_speed` are outside that range) are kept.

    Args:
        dataset (pandas.DataFrame): the original dataset
        speed_min (float): the minimum value for `average_speed` that is kept. Defaults to 3
        speed_max (float): the maximum value for `average_speed` that is kept. Defaults to 60
        keep_tdf (bool): whether to keep the 12th stage of the 2003 Tour de France. Defaults to True

    Returns:
        pandas.DataFrame: copy of dataset with the rows removed
    """

    # Not sure if the variable is passed by value or by reference...
    races_df_copy = dataset.copy()

    # We keep Stage 12 of 2003's Tour de France
    condition = races_df_copy['_url'] == 'tour-de-france/2003/stage-12' if keep_tdf else False
    races_df_copy = races_df_copy[((races_df_copy['average_speed'] > speed_min) 
                                  & (races_df_copy['average_speed'] <= speed_max)) 
                                  | (condition) ]
    
    return races_df_copy

