from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
import numpy as np
import copy
import pandas as pd

from typing import Tuple

def scale_data(data) -> Tuple[StandardScaler, np.ndarray]:
    """
    Preprocess the data using the StandardScaler class from scikit

    args:
     - data np.ndarray : The data to normalize

    returns:
        - StandardScaler : The normalizer object (to be used for inverse transformation)
        - np.ndarray : The normalized data
    """
    normalizer = StandardScaler()
    normalizer.fit(data)
    return normalizer, normalizer.transform(data)

def inverse_scale_data(scaler: StandardScaler, data: np.ndarray) -> np.ndarray:
    """
    Inverse scale the data using the StandardScaler class from scikit

    args:
     - scaler StandardScaler : The normalizer object
     - data np.ndarray : The data to inverse normalize

    returns:
        - np.ndarray : The inverse normalized data
    """
    return scaler.inverse_transform(data)

def k_search(max_clusters: int, n_init: int, data: np.ndarray, r_state:int = 42) -> list[float]:
    """
    Search for the best number of clusters using the silhouette score

    args:
        - max_clusters int : The maximum number of clusters to try
        - n_init int : Number of time the k-means algorithm will be run with different centroid seeds.
        - data np.ndarray : The data to cluster
        - r_state int : Random state for the KMeans algorithm

    returns:
        - np.ndarray : The silhouette scores for each cluster number (from 2 to max_clusters)
    """
    silhouettes: list[float] = []
    for cluster in range(2, max_clusters):
        kmeans = KMeans(n_clusters=cluster, random_state=r_state, n_init=n_init)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette = silhouette_score(data, labels)
        silhouettes.append(silhouette)
    return silhouettes


def hier_search(hyperparameters, data, r_state= 42, samples = 80, calinski=False):
    """
    Search for the best hyperparameters using the silhouette score

    args:
        - hyperparameters dict : The hyperparameters for the AgglomerativeClustering algorithm
        - data np.ndarray : The data to cluster

    returns:
        - np.ndarray : The silhouette scores for each cluster number (from 2 to max_clusters)
    """
    results_per_algorithm = list()
    clusterings = list()
    # sample the hyperpamrameters
    sampled_hyperparameters = list(ParameterSampler(
        copy.deepcopy(hyperparameters),
        n_iter=samples,
        random_state=r_state
    ))
    # fit the models for all sampled hyperparamters
    models = [
        AgglomerativeClustering(**selected_hyperparameters).fit(data)
        for selected_hyperparameters in sampled_hyperparameters
    ]
    # extract the clusterings
    clusterings += [
        model.labels_
        for model in models
    ]
    
    # initialize the results dataframe
    results_per_algorithm += sampled_hyperparameters
    results_df = pd.DataFrame.from_records(results_per_algorithm)
    results_df.loc[:, "random_state"] = r_state

    results_df = results_df.astype({"n_clusters": int, "random_state": int})

    # get the silhouette score for each model
    silhouette_per_model = [
        silhouette_score(data, clustering) if len(set(clustering)) > 1 else -1
        for clustering in clusterings
    ]

    results_df.loc[:, "silhouette"] = silhouette_per_model
    results_df = results_df.sort_values(by="silhouette", ascending=False)

    # get the calinski score for each model
    if calinski:
        calinski_per_model = [
            calinski_harabasz_score(data, clustering) if len(set(clustering)) > 1 else -1
            for clustering in clusterings
        ]
        results_df.loc[:, "calinski"] = calinski_per_model

    return results_df


def get_average_per_cluster(labels, df):
    """
    Get the average values of the attributes per cluster (can return std too)

    args:
        - labels np.ndarray : The cluster labels
        - df pd.DataFrame : data

    returns:
        - pd.DataFrame : The average values of the cyclists per cluster
    """
    clusters_sizes = np.unique(labels, return_counts=True)[1]
    average_per_cluster = df.groupby("cluster")\
        .describe()\
        .xs(                  # select from a multi-index dataframe
            "mean",           # which columns to select?
            level=1,          # at what level of the index?
            drop_level=True,  
            axis="columns"
        )
    average_per_cluster.loc[:, "cluster_size"] = clusters_sizes
    std_per_cluster = df.groupby("cluster")\
        .describe()\
        .xs(                  # select from a multi-index dataframe
            "std",           # which columns to select?
            level=1,          # at what level of the index?
            drop_level=True,  
            axis="columns"
        )
    std_per_cluster.loc[:, "cluster_size"] = clusters_sizes


    return average_per_cluster, std_per_cluster