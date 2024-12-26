import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from itertools import chain
from matplotlib.colors import ListedColormap
import re

def plot_missing_stages_heatmap(editions, 
                                stages_in_df_dict,
                                missing_stages_dict, 
                                race_name, ax) -> None: 
    """Plots a heatmap of the stages presence in the `edition` dataframe,
    for the race `race_name`. 


    Args:
        editions (_type_): _description_
        stages_in_df_dict (_type_): _description_
        missing_stages_dict (_type_): _description_
        race_name (_type_): _description_
        ax (_type_): _description_

    Returns:
        _type_: _description_
    """
    # We combine the stages (missing and not missing in df) for a given year year
    def combiner(url, y):
        return stages_in_df_dict.get(f'{url}_{int(y)}_stages',[]) + missing_stages_dict.get(f'{url}_{int(y)}_missing_stages',[])
    # We accumulate over the years, removing duplicates and sorting them according to the stage number
    all_stages_ever = sorted(set(chain.from_iterable([combiner(race_name, y) for y in editions])), key=lambda x: 0 if x in ['prologue','result','prelude'] else int(re.search(r'\d+', x).group()))

    data = []
    for year in editions:
        row = []
        stages_present_in_that_year_in_df = stages_in_df_dict.get(f'{race_name}_{int(year)}_stages', [])
        if stages_present_in_that_year_in_df == []:
            row = [0] * len(all_stages_ever)
        else:
            for stage in all_stages_ever:
                if stage in missing_stages_dict.get(f'{race_name}_{int(year)}_missing_stages', []):
                    row.append(0)
                else:
                    row.append(1)
        data.append(row)
    
    df = pd.DataFrame(data, index=[int(y) for y in editions], columns=all_stages_ever)
    cmap = ListedColormap(['red', 'lightgreen'])

    sns.heatmap(df, cmap=cmap, cbar=False, linewidths=.5, ax=ax, vmin=0, vmax=1, annot=False)
    ax.set_xlabel('Stages')
    ax.set_ylabel('Year')
    ax.set_title(f'{race_name}', fontweight='bold')


def plot_participations(df: pd.DataFrame, 
                        ax,
                        race_name:str,
                        shorten_xaxis:bool = True) -> None:
    """Plots the average number of participants (average over the stages, if any) 
    for the competition `race_name`, using data in the dataframe `df`.
    The plot is done on the axis `ax`.

    Args:
        df (pd.DataFrame): The races dataframe
        ax (_type_): the axis where to plot the data
        race_name (str): the name of the race to plot. Must be the portion of the url
        shorten_xaxis (bool, optional): whether to show all the years or to subsample. Defaults to True.

    Returns:
        None: 
    """
    # Copy, so that we don't modify the original dataframe
    df_copy = df.copy()
    # We need the date column to just have the date
    if "time" not in df_copy.columns:
        df_copy["time"] = df_copy["date"].apply(lambda string: string.split(' ')[1])
        df_copy["date"] = df_copy["date"].apply(lambda string: string.split(' ')[0])
        df_copy["year"] = df_copy["date"].apply(lambda string: string.split('-')[0]).astype(int)

    def extract_name_stage(url:str,
              pattern:str=r"([a-z0-9-]+)/\d{4}/(prologue|result|stage-\d+[ab]?)") -> str:
        match = re.match(pattern,url)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
        else:
            return None
    
    # We need this to do the groupby. The df in data_understanding_transformed.py doesn't have this
    if "name_stage" not in df_copy.columns:
        df_copy["name_stage"] = df_copy["_url"].apply(extract_name_stage)

    # No I'm not saying that the names we use are wrong! :D
    df_copy['name'] = df_copy['name_stage'].apply(lambda s: s.split('_')[0])

    gara_e_tappa = df_copy.groupby(["name", "name_stage", "year"])
    res = gara_e_tappa['year'].value_counts()

    # modo che non mi piace, ma che funziona
    res_frame = res.to_frame().reset_index()

    # We need to filter the dataframe
    small_df = res_frame[res_frame['name'] == race_name]
    sns.lineplot(data=small_df, x='year', y='count', err_style='bars', errorbar='pi', marker='o', ax=ax)
    sns.scatterplot(data=small_df, x='year', y='count', ax=ax, color='red', alpha=0.2)
    ax.set_title(race_name, fontweight='bold')
    ax.set_ylabel('Number of participants')
    ax.set_xlabel('Year')

    xticks = np.sort(small_df['year'].unique())
    if shorten_xaxis and len(xticks) > 20:
        xticks = [xticks[i] for i in np.linspace(0, len(xticks)-1, num = 20, dtype = int)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=90)


def plot_kdistances(
    dist_matrix:np.ndarray,
    k:int,
    eps_value:float,
    color,
    ax,
    y_lim:float = 10
) -> None:
    kth_distances = [d[np.argsort(d)[k]] for d in dist_matrix]

    ax.plot(np.sort(kth_distances), label=f'Min_samples = {k}', 
            alpha=0.6, color=color)
    ax.axhline(y=eps_value, linestyle='--', 
               color=color, alpha=0.5,
               label=fr'$\epsilon=${eps_value}')
    ax.set_title(f'K-distances plot, K = {k}')
    ax.set_xlabel('Cyclist index (sorted)')
    ax.set_ylabel(f'Distance from {k}-th neighbour')
    ax.set_ylim(0, y_lim)
    ax.legend()
    ax.grid(True)       
        


