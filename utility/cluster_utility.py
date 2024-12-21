from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

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