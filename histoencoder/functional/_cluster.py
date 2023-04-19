from collections.abc import Iterable
from typing import Union

import numpy as np
import polars as pl
import torch
from sklearn.cluster import MiniBatchKMeans

ERROR_NO_FEATURES = "Could not find any feature columns."
ERROR_FEATURE_TYPE = "Expected features to be a dataframe, array or tensor, not '{}'."
ERROR_FEATURE_NDIM = "Expected features to have dimensionality of 2 not '{}'."


def cluster_features(
    features: Union[pl.DataFrame, np.ndarray, torch.Tensor],
    n_clusters: Union[int, Iterable[int]] = (4, 8, 16, 32, 64),
    **kwargs,
) -> pl.DataFrame:
    """Cluster features with KMeans into different number of clusters.

    Args:
        features: Dataframe with feature columns (`feat[1..=num_clusters]`) or
            array/tensor of features.
        n_clusters: An iterable of cluster numbers or an integer. Defaults to
            `(4, 8, 16, 32, 64)`.
        **kwargs: Passed to `sklearn.cluster.MiniBatchKMeans`.

    Returns:
        Polars dataframe with cluster assignments for each number of clusters.
    """
    if isinstance(features, pl.DataFrame):
        feature_columns = [x for x in features.columns if x.startswith("feat")]
        if len(feature_columns) == 0:
            raise ValueError(ERROR_NO_FEATURES)
        X_feats = features[feature_columns].to_numpy()
    elif isinstance(features, np.ndarray):
        X_feats = features.copy()
    elif isinstance(features, torch.Tensor):
        X_feats = features.numpy()
    else:
        raise TypeError(ERROR_FEATURE_TYPE.format(type(features)))
    if X_feats.ndim != 2:  # noqa
        raise ValueError(ERROR_FEATURE_NDIM.format(X_feats.ndim))
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]
    cluster_assignments = {}
    for n in n_clusters:
        if n >= X_feats.shape[0]:
            break
        cluster_assignments[f"n_clusters={n}"] = _cluster_features(
            X_feats, n_clusters=n, **kwargs
        )
    return pl.DataFrame(cluster_assignments)


def _cluster_features(
    features: np.ndarray, n_clusters: int, n_init: str = "auto", **kwargs
) -> None:
    return MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init, **kwargs).fit_predict(
        features
    )
