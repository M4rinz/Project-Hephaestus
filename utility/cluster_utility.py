from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
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


def hier_search(hyperparameters, data, r_state= 42, calinski=False):
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
        n_iter=80,
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


def get_average_cyclist_per_cluster(labels, cyclists_df):
    """
    Get the average values of the cyclists per cluster

    args:
        - labels np.ndarray : The cluster labels
        - cyclists_df pd.DataFrame : The cyclists data

    returns:
        - pd.DataFrame : The average values of the cyclists per cluster
    """
    clusters_sizes = np.unique(labels, return_counts=True)[1]
    average_cyclist_per_cluster = cyclists_df.groupby("cluster")\
        .describe()\
        .xs(                  # select from a multi-index dataframe
            "mean",           # which columns to select?
            level=1,          # at what level of the index?
            drop_level=True,  
            axis="columns"
        )
    average_cyclist_per_cluster.loc[:, "cluster_size"] = clusters_sizes
    std_cyclist_per_cluster = cyclists_df.groupby("cluster")\
        .describe()\
        .xs(                  # select from a multi-index dataframe
            "std",           # which columns to select?
            level=1,          # at what level of the index?
            drop_level=True,  
            axis="columns"
        )
    std_cyclist_per_cluster.loc[:, "cluster_size"] = clusters_sizes


    return average_cyclist_per_cluster

def DBSCAN_grid_search(
        data,
        min_samples_range:list[float], 
        eps_range:list[list[float]],
        distance:str='euclidean',
        print_results:bool = True
) -> tuple[dict]:
    if distance == 'mahalanobis':
        cov_matrix = data.cov()
    cluster_labels_dict = {}
    dbscans_dict = {}
    silhouettes_dict = {}
    for min_samples, eps_values in zip(min_samples_range, eps_range):
        for eps in eps_values:
            if distance == 'mahalanobis':
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, 
                                metric=distance, metric_params={'V': cov_matrix})
            else:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=distance)
            dbscan.fit(data)
            dbscans_dict[f"min_samples={min_samples}_eps={eps}"] = dbscan
            # Get the labels and save them
            labels = dbscan.labels_
            cluster_labels_dict[f"min_samples={min_samples}_eps={eps}"] = dbscan.labels_
            # Silhouette score
            silhouette = silhouette_score(data, labels)
            silhouettes_dict[f"min_samples={min_samples}_eps={eps}"] = silhouette
            # N° of clusters, noise points
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_percentage = 100 * n_noise / len(labels)
            if print_results:
                print(f'eps = {eps:<4}, min_samples = {min_samples:>3}, n_clusters = {n_clusters:>2}, n_noise = {n_noise:>4}, noise % = {noise_percentage:>2.2f}, Silhouette score: {silhouette:.3f}')

    return cluster_labels_dict, dbscans_dict, silhouettes_dict

def prepare_data_for_DBSCAN_heatmaps(
        silhouettes_dict:dict[str, float],
        cluster_labels_dict:dict[str, np.ndarray],
        min_samples_range:list[float],
        eps_range:list[list[float]]
) -> tuple[list]:
    heatmap_sil_data, heatmap_noise_data = [], []
    heatmap_ncl_data, heatmap_in0_data = [], []
    for n_samples in min_samples_range:
        row1, row2, row3, row4 = [], [], [], []
        for eps in np.unique(np.ravel(eps_range)):
            try:
                row1.append(silhouettes_dict[f'min_samples={n_samples}_eps={eps}'])
                n_noise = sum(cluster_labels_dict[f'min_samples={n_samples}_eps={eps}'] == -1)
                row2.append(n_noise)
                row3.append(len(np.unique(cluster_labels_dict[f'min_samples={n_samples}_eps={eps}'])) - 1)
                # compute ratio between n° of points in cluster 0 and total n° of points
                numerator = sum(cluster_labels_dict[f'min_samples={n_samples}_eps={eps}'] == 0)
                denominator = len(cluster_labels_dict[f'min_samples={n_samples}_eps={eps}'])
                row4.append(numerator / denominator)
            except KeyError:
                row1.append(np.nan)
                row2.append(np.nan)
                row3.append(np.nan)
                row4.append(np.nan)
        heatmap_sil_data.append(row1)
        heatmap_noise_data.append(row2)
        heatmap_ncl_data.append(row3)
        heatmap_in0_data.append(row4)

    return heatmap_sil_data, heatmap_noise_data, heatmap_ncl_data, heatmap_in0_data