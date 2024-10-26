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
    `_url` colon. It uses the regular expression passed as `pattern` to extract the race ID

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
    for i in tqdm.tqdm(indices):
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
                    # (hopefully duplicated cyclists have been already dealt with)
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
