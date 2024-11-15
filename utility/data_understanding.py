from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
import pandas
import numpy as np
import re
import tqdm
import procyclingstats as pcs

def __transform_single_features(dataset: pandas.DataFrame, transformation: str) -> Tuple[
    pandas.DataFrame, Dict[str, Any]]:
    match transformation:
        case "standard":
            transformed_dataset = dataset.copy().select_dtypes(exclude=["object", "category", "bool", "datetime64"])
            transformations = dict()

            for feature in transformed_dataset.columns:
                transformations[feature] = StandardScaler()
                transformed_feature = transformations[feature].fit_transform(transformed_dataset[[feature]]).squeeze()
                transformed_dataset = transformed_dataset.astype({feature: transformed_feature.dtype})
                transformed_dataset.loc[:, feature] = transformed_feature
        case _:
            raise ValueError(f"Unknown transformation: {transformation}")

    return transformed_dataset, transformations


def center_and_scale(dataset: pandas.DataFrame) -> Tuple[pandas.DataFrame, Dict[str, Any]]:
    """Shifts data to the origin: removes mean and scales by standard deviation all numeric features. Returns a copy of the dataset."""
    return __transform_single_features(dataset, "standard")


def drop_boolean(dataset: pandas.DataFrame) -> pandas.DataFrame:
    return dataset.select_dtypes(exclude="bool")

def check_if_same(race1:str, 
                  race2:str, 
                  races_df:pandas.DataFrame,
                  pattern:str=r"([a-z0-9-]+)/\d{4}/(prologue|result|stage-\d)") -> tuple:
    """Checks if two names refer to the same race, comparing the name that appears in the 
    `_url` column. It uses the regular expression passed as `pattern` to extract the race ID

    Args:
        race1 (str): name of the first race to compare
        race2 (str): name of the second race to compare
        pattern (str, optional): The pattern against which to check the URLs. Defaults to r"([a-z0-9-]+)/\\d{4}/(prologue|result|stage-\\d)".

    Returns:
        tuple: equality (True|False), race1's ID(s), race2's ID(s)
    """
    race1_urls = races_df[races_df['name'] == race1]['_url'].unique()
    race2_urls = races_df[races_df['name'] == race2]['_url'].unique()
    # Just checks
    if len(race1_urls) == 0:
        print(f"The race name {race1} has no corresponding URLs.\n Are you sure you didn't misspell?")
        return
    if len(race2_urls) == 0:
        print(f"The race name {race2} has no corresponding URLs.\n Are you sure you didn't misspell?")
        return
    
    # This pattern matches all the races' URLs
    #pattern = r"([a-z0-9-]+)/\d{4}/(prologue|result|stage-\d)"

    def extract_race_ID(race_url:str) -> str|None:
        match = re.match(pattern,race_url)
        if match:
            # We target the name of the race
            return match.group(1)
        return None

    race_ID_1 = np.unique(np.array([extract_race_ID(url) for url in race1_urls]))
    race_ID_2 = np.unique(np.array([extract_race_ID(url) for url in race2_urls]))

    return np.array_equal(race_ID_1,race_ID_2), race_ID_1, race_ID_2

def correlations(dataset: pandas.DataFrame) -> pandas.DataFrame:
    correlations_dictionary = {
        correlation_type: dataset.corr(numeric_only=True, method=correlation_type)
        for correlation_type in ("kendall", "pearson", "spearman")
    }
    for i, k in enumerate(correlations_dictionary.keys()):
        correlations_dictionary[k].loc[:, "correlation_type"] = k
    correlations_matrix = pandas.concat(correlations_dictionary.values())
    
    return correlations_matrix


def convert_seconds_date(time: int) -> str:
    if pandas.isna(time):
        return np.nan
    time = int(time)
    hh = time//3600
    mm = (time-hh*3600)//60
    ss = (time - hh*3600 - mm*60)
    return f"{hh:02}:{mm:02}:{ss:02}"

def convert_date_seconds(data:str) -> int:
    hh, mm, ss = data.split(':')
    return 3600*int(hh)+60*int(mm)+int(ss)


def delta_computer(cyclist_url:str, 
                   lista: list[dict], 
                   in_date_format:bool=True,
                   which:str|int='all') -> list[str]:
    """Given the url of the cyclist and the list of the stage's participants data,
    computes the delta of the cyclist by taking the difference between the cyclist's 
    time to complete the race and the time of the first.

    Args:
        cyclist_url (str): url of the cyclist (in the format of the dataframe)
        lista (list[dict]): the list returned by pcs.Stage.results(). Must include the parameters `rider_url` and `time`
        in_date_format (bool): whether return the delta in the format hh:mm:ss. If False it will return the delta in n° of seconds. Defaults to True
        which (str|int): which cyclist to consider. If 'all' it will return a list of all the deltas of that cyclist. Defaults to 'all'

    Returns:
        list(str): the list of deltas of that cyclist, in format hh:mm:ss if in_date_dormat
    """
    try:
        rider_dicts = list(filter(lambda diz: diz['rider_url'] == f'rider/{cyclist_url}',lista))
        if which == 'all':
            tempo = [convert_date_seconds(diz['time']) for diz in rider_dicts]
        else:
            try:
                tempo = [convert_date_seconds(rider_dicts[which]['time'])]
            except IndexError:
                return [np.nan] #tempo = np.nan
    except AttributeError:
        return [np.nan] #tempo = np.nan
    delta_sec = [t - convert_date_seconds(lista[0]['time']) for t in tempo] #if which == 'all' else tempo - convert_date_seconds(lista[0]['time'])
    
    return [convert_seconds_date(d_s) for d_s in delta_sec] if in_date_format else delta_sec



def scrape_stages(indices:pandas.Index,
                  races_df:pandas.DataFrame,
                  same_races_dict:dict[str|list[str]]) -> list[dict]:
    new_races = []

    # Helper function to handle exceptions
    def safe_getattr(obj, attr, fun:callable=lambda x: x):
        try:
            return fun(getattr(obj, attr)())
        except (IndexError, AttributeError, ValueError):
            return np.nan
        
    def name_returner(dictionary:dict[str,list[str]], val_to_find:str) -> str|float:
        for key, val in dictionary.items():
            if val_to_find in val:
                return key
        # Well... Turns out that in the dictionary there isn't really everything...
        return val_to_find #np.nan


    prev_url = None
    for i in tqdm.tqdm(indices, desc='scraping...'):
        url = races_df.loc[i, '_url']

        # We process only the different URLs
        if url == prev_url:
            continue

        try:
            tappa = pcs.Stage(f"race/{url}")
            same_url_races = races_df[races_df['_url'] == url]

            ## Non-cyclist oriented
            # New feature: ITT (Individual Time Trial), TT (Team TT), RR (Road Race)
            tipo_tappa = safe_getattr(tappa, 'stage_type')
            elevazione = races_df.loc[i, 'climb_total'] if not pandas.isna(races_df.loc[i, 'climb_total']) else safe_getattr(tappa, 'vertical_meters')
            # In our dataset missing profile is treated as NaN, while the scraper treats them as 0
            profilo = races_df.loc[i, 'profile'] if not pandas.isna(races_df.loc[i, 'profile']) else safe_getattr(tappa, 'profile_icon', lambda profile: np.float64(profile[1]) if np.float64(profile[1]) != 0 else np.nan)
            # If the temperature isn't there then tappa.avg_temperature() is None. None becomes NaN in the dataframe
            avgtemp = races_df.loc[i, 'average_temperature'] if not pandas.isna(races_df.loc[i, 'average_temperature']) else safe_getattr(tappa, 'avg_temperature')

            for _, row in same_url_races.iterrows():
                posizione, url_ciclista = row[['position', 'cyclist']]

                ## Cyclist-oriented
                lista = tappa.results('rider_url', 'age', 'pcs_points', 'uci_points', 'team_url')   
                try:
                    # There can be cyclists that appear in our df but not in the website, so we have to do like this
                    # Duplicated cyclists haven't been already dealt with. We just get the first one we find...
                    # ... Hopefully it works out... I mean, is more or less consistent with the original dataset...
                    diz_valori_ciclista = next(filter(lambda diz: diz['rider_url'] == f'rider/{url_ciclista}', lista))
                except StopIteration:
                    # This happens when the name is in the dataframe but not in procyclingstats.
                    # Most probably the dataframe is wrong and pcs is right, but let's keep it anyways...
                    diz_valori_ciclista = {}

                # In our dataset missing points are treated as NaN, while the scraper treats them as 0
                punti = diz_valori_ciclista.get('pcs_points', np.nan) if diz_valori_ciclista.get('pcs_points', np.nan) != 0 else np.nan
                # In our dataset missing UCI points are treated as NaN, while the scraper treats them as 0
                punti_uci = diz_valori_ciclista.get('uci_points', np.nan) if diz_valori_ciclista.get('uci_points', np.nan) != 0 else np.nan
                # For the age we can look at our dataframe first (hopefully everything is ok)
                eta = row['cyclist_age'] if not pandas.isna(row['cyclist_age']) else diz_valori_ciclista.get('age', np.nan)
        
                # The teams in the dataset are completely different from those of pcs...
                team = diz_valori_ciclista.get('team_url', np.nan)
                #team = races_df.loc[idx, 'cyclist_team'] if not races_df.loc[idx, 'cyclist_team'] else diz_valori_ciclista.get('team_url', np.nan)     

                # And now let's create the new entry
                race_new_data = {
                    '_url': url,
                    'name': ' '.join(name_returner(same_races_dict, races_df.loc[i, 'name']).split()),
                    'stage_type': tipo_tappa,
                    'points': punti,
                    'uci_points': punti_uci,
                    'length': races_df.loc[i, 'length'],
                    'climb_total': elevazione,
                    'profile': profilo,
                    'startlist_quality': races_df.loc[i, 'startlist_quality'],
                    'average_temperature': avgtemp,
                    'date': row['date'],
                    'position': posizione,
                    'cyclist': url_ciclista,
                    'cyclist_age': eta,
                    # Until we know how to get these...
                    'is_tarmac': races_df.loc[i, 'is_tarmac'],
                    'is_cobbled': races_df.loc[i, 'is_cobbled'], # ... as if they weren't all False...
                    'is_gravel': races_df.loc[i, 'is_gravel'],
                    'cyclist_team': team,
                    'delta': row['delta']
                }
                new_races.append(race_new_data)       

            # Update url
            prev_url = url
        except ValueError:
            print(f"Encountered error at url {url}, iteration {i}")
            print("Exiting (I'm sorry you might have lost a lot of time)")
            return
        
    return new_races


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