from pathlib import Path

import polars as pl
from timm.models.xcit import XCiT
from torch.utils.data import DataLoader

import histoencoder.functional as F

from ._utils import (
    create_custom_loader,
    create_reader_loader,
    create_tile_loader,
)


def test_write_features_reader(tmp_path: Path) -> None:
    encoder = F.create_encoder("prostate_small")
    loader = create_reader_loader(batch_size=2, num_samples=8)
    save_and_check_feature_dataframes(
        columns=["x", "y", "w", "h"],
        tmp_path=tmp_path,
        encoder=encoder,
        loader=loader,
    )


def test_write_features_tiles(tmp_path: Path) -> None:
    encoder = F.create_encoder("prostate_small")
    loader = create_tile_loader(batch_size=2, num_samples=8)
    save_and_check_feature_dataframes(
        columns=["path", "x", "y", "w", "h"],
        tmp_path=tmp_path,
        encoder=encoder,
        loader=loader,
    )


def test_write_features_custom(tmp_path: Path) -> None:
    encoder = F.create_encoder("prostate_small")
    loader = create_custom_loader(batch_size=2)
    save_and_check_feature_dataframes(
        columns=[],
        tmp_path=tmp_path,
        encoder=encoder,
        loader=loader,
    )


def save_and_check_feature_dataframes(
    tmp_path: Path, encoder: XCiT, loader: DataLoader, columns: list[str]
) -> None:
    # Save features.
    output_dir = tmp_path / "features"
    F.save_features(encoder, output_dir, loader, max_samples=5)
    # Check files.
    excepted = {f"features_{i+1}.parquet" for i in range(2)}
    assert {x.name for x in output_dir.iterdir()} == excepted
    # Check dataframes.
    feature_columns = [f"feat{i+1}" for i in range(encoder.embed_dim)]
    feats_1 = pl.read_parquet(output_dir / "features_1.parquet")
    feats_2 = pl.read_parquet(output_dir / "features_2.parquet")
    all_columns = columns + feature_columns
    assert feats_1.columns == all_columns
    assert feats_2.columns == all_columns
    # first should have 6 samples (due to batch_size=2) and latter 2.
    assert feats_1.shape == (6, len(all_columns))
    assert feats_2.shape == (2, len(all_columns))


def test_yield_features_reader() -> None:
    encoder = F.create_encoder("prostate_small")
    for feats, __ in F.yield_features(encoder, loader=create_reader_loader()):
        assert feats.shape == (2, encoder.embed_dim)
        break


def test_yield_features_tiles() -> None:
    encoder = F.create_encoder("prostate_small")
    for feats, __ in F.yield_features(encoder, loader=create_tile_loader()):
        assert feats.shape == (2, encoder.embed_dim)
        break
