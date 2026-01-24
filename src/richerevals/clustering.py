"""Embedding + clustering utilities for diversity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


@dataclass
class ClusterResult:
    """Clustering summary for a set of embeddings."""

    labels: np.ndarray
    centroids: np.ndarray
    exemplar_indices: List[int]
    mean_cosine_distance: float


def embed_texts(
    texts: List[str], model_name: str, device: Optional[str] = None
) -> np.ndarray:
    """Embed texts with a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def cluster_kmeans(
    embeddings: np.ndarray, n_clusters: int, random_state: int = 1337
) -> ClusterResult:
    """Run KMeans and return exemplar indices and summary stats."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    exemplar_indices: List[int] = []
    for cluster_id in range(n_clusters):
        cluster_idx = np.where(labels == cluster_id)[0]
        if cluster_idx.size == 0:
            exemplar_indices.append(-1)
            continue
        cluster_embeddings = embeddings[cluster_idx]
        centroid = centroids[cluster_id]
        distances = pairwise_distances(
            cluster_embeddings, centroid.reshape(1, -1), metric="cosine"
        )
        exemplar_idx = cluster_idx[int(np.argmin(distances))]
        exemplar_indices.append(exemplar_idx)

    mean_cosine_distance = float(pairwise_distances(embeddings, metric="cosine").mean())

    return ClusterResult(labels, centroids, exemplar_indices, mean_cosine_distance)
