from pathlib import Path

import polars as pl

import histoencoder.functional as F

from ._utils import create_tile_loader, generate_features


def test_clustering_dataframe(tmp_path: Path) -> None:
    F.save_features(
        F.create_encoder("prostate_small"),
        output_dir=tmp_path,
        loader=create_tile_loader(num_samples=20),
    )
    feats = pl.read_parquet(tmp_path / "features_1.parquet")
    clusters = F.cluster_features(feats, n_clusters=[4, 8, 16, 32])
    assert clusters.columns == ["n_clusters=4", "n_clusters=8", "n_clusters=16"]
    assert clusters.shape == (20, 3)


def test_clustering_torch() -> None:
    clusters = F.cluster_features(
        generate_features(num_samples=20), n_clusters=[4, 8, 16, 32]
    )
    assert clusters.columns == ["n_clusters=4", "n_clusters=8", "n_clusters=16"]
    assert clusters.shape == (20, 3)


def test_clustering_numpy() -> None:
    clusters = F.cluster_features(
        generate_features(num_samples=20).numpy(), n_clusters=[4, 8, 16, 32]
    )
    assert clusters.columns == ["n_clusters=4", "n_clusters=8", "n_clusters=16"]
    assert clusters.shape == (20, 3)
