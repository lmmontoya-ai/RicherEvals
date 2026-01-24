"""Cluster outputs from a generations JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from richerevals.clustering import cluster_kmeans, embed_texts


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for clustering outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to generations JSON file")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Cluster generations and print a summary JSON."""
    args = parse_args()
    path = Path(args.input)
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    texts = [item["text"] for item in results]

    embeddings = embed_texts(texts, args.model)
    clusters = cluster_kmeans(embeddings, n_clusters=args.k)

    output = {
        "n": int(len(texts)),
        "k": int(args.k),
        "mean_cosine_distance": float(clusters.mean_cosine_distance),
        "exemplar_indices": [int(idx) for idx in clusters.exemplar_indices],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
