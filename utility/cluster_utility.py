from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
import numpy as np
import copy
import pandas as pd

# You need to install the kneed package to use this function
# pip install kneed, or conda install -c conda-forge kneed
from kneed import KneeLocator

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

def k_search(max_clusters: int, n_init: int, data: np.ndarray, r_state:int = 42, init_method:str = 'random') -> list[float]:
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
        kmeans = KMeans(n_clusters=cluster, random_state=r_state, n_init=n_init, init=init_method)
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

def DBSCAN_grid_search(
        data:pd.DataFrame,
        min_samples_range:list[float], 
        eps_range:list[list[float]],
        distance:str='euclidean',
        print_results:bool = True
) -> tuple[dict]:
    """Performs the grid search used in the `density_based` clustering notebook.
    For each `min_samples` value, the function expects a list of possible `eps` values to try.
    Each clustering is evaluated using the Silhouette score.

    Args:
        data (pandas.DataFrame): the (normalized) dataset
        min_samples_range (list[float]): the values for the `min_samples` hyperparameter 
        eps_range (list[list[float]]): the values for the `eps` hyperparameter
        distance (str, optional): the metric used to compute the distances. Defaults to 'euclidean'.
        print_results (bool, optional): Whether to print the infos about a certain configuration of hyperparameters. Defaults to True.

    Returns:
        tuple[dict]: _description_
    """
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
            try:
                silhouette = silhouette_score(data, labels)
            except ValueError:
                silhouette = -1     # if there is only one cluster, we put a nice -1
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
    """Prepares the matrices of values that are then printed as heatmaps in
    the `density_based` notebook.
    each of the dictionaries has `min_samples=x_eps=y` as keys, where 
    `x` and `y` are retrieved from the `min_samples_range` and `eps_range` list
    respectively. The values of the hyperparameters correspond to the rows and columns
    of these heatmaps. These are:
    - `heatmap_sil_data`: the silhouette score of the clustering obtained with the combination of hyperparameters
    - `heatmap_noise_data`: the n° of noise points in the clustering obtained with the combination of hyperparameters
    - `heatmap_ncl_data`: the n° of clusters (excluding noise points!) in the clustering obtained with the combination of hyperparameters
    - `heatmap_in0_data`: the proportion of points in cluster 0 in the clustering obtained with the combination of hyperparameters

    Args:
        silhouettes_dict (dict[str, float]): _description_
        cluster_labels_dict (dict[str, np.ndarray]): _description_
        min_samples_range (list[float]): _description_
        eps_range (list[list[float]]): _description_

    Returns:
        tuple[list]: _description_
    """
    heatmap_sil_data, heatmap_noise_data = [], []
    heatmap_ncl_data, heatmap_in0_data = [], []
    for n_samples in min_samples_range:
        row1, row2, row3, row4 = [], [], [], []
        for eps in np.unique(np.concatenate(eps_range)):
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

def compute_eps_values(
        k:int, 
        dist_matrix:np.ndarray, 
        n_eps:int = 7,
        eps_val:float = None,
        policy:str = 'mid',
        elbow_proportion:int = 3,
    ) -> list[float]:
    """
    Computes `n_eps` values for the `eps` parameter, for the DBSCAN algorithm. 
    If `eps_val` is not provided, one is found using the elbow method
    automatically with the KneeLocator class.
    The `policy` parameter can be one of 'mid', 'max', 'min', to determine
    if the values should be centered around the `eps_val`, or if they should
    be all smaller or all bigger (respectively).
    The `elbow_proportion` parameter is used to determine the step size for the
    `eps` values. It is such that the step size is `max(kth_distances) - min(kth_distances) / (n_eps * elbow_proportion)`.
    Hence, it should be roughly equal to the (god of Mathematics forgive me) 
    proportion of overall w.r.t. the elbow part of the graph.

    Args:
        k (int): Number of neighbors to consider.
        dist_matrix (np.ndarray): Distance matrix.
        n_eps (int, optional): Number of values for eps to produce. Defaults to 7.
        eps_max (float, optional): Maximum value for eps. If not provided, it will be determined using KneeLocator.
        policy (str, optional): Policy to determine the eps values. Defaults to 'mid'.
        elbow_proportion (int, optional): Proportion of the elbow w.r.t the overall graph. Defaults to 3.
    Returns:
        list[float]: List of computed eps values.
    """
    kth_distances = [d[np.argsort(d)[k]] for d in dist_matrix]
    if eps_val is None:
        klocator = KneeLocator(np.arange(len(kth_distances)), np.sort(kth_distances), curve='convex', direction='increasing')
        eps_val = np.round(klocator.knee_y, 3)
        #print(f"Knee found at index {klocator.knee} with value {eps_val}")
    
    step = np.round((np.max(kth_distances) - np.min(kth_distances)) / (n_eps*elbow_proportion), 3)
    
    if policy in ['mid', 'middle', 'center']:
        eps_list = [eps_val + i * step for i in range(-(n_eps // 2), (n_eps // 2) + 1)]
    elif policy in ['max', 'maximum']:
        eps_list = [eps_val - i * step for i in range(n_eps)]
    elif policy in ['min', 'minimum']:
        eps_list = [eps_val + i * step for i in range(n_eps)]
    else:
        raise ValueError(f"Policy {policy} not recognized. Choose one of 'mid', 'max', 'min'.")
    # There may be less than n_eps values...
    eps_list = np.unique(sorted([np.round(eps, 2) for eps in eps_list if eps > 0]))

    return eps_list