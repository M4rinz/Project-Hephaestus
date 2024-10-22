from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
import pandas
import numpy as np
import re

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