import shutil
from pathlib import Path
from typing import Union

import numpy as np
import polars as pl
import tqdm
from histoprep.utils import SlideReaderDataset, TileImageDataset
from timm.models.xcit import XCiT
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ._features import yield_features

ERROR_OUTPUT_EXISTS = "Output directory exists but `overwrite=False`."


def save_features(
    encoder: XCiT,
    output_dir: Union[str, Path],
    loader: DataLoader,
    *,
    max_samples: int = 2**16,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """Write features to disk.

    Args:
        encoder: XCiT encoder model for extracting features.
        output_dir: Output directory for feature parquet files.
        loader: `DataLoader` yielding tensor images as the first or only element.
        max_samples: Maximum samples per parquet file. Defaults to 65536.
        overwrite: Remove everything in output directory. Defaults to False.
        verbose: Enable `tqdm.tqdm` progress bar. Defaults to True.

    Raises:
        FileExistsError: Output directory exists but `overwrite=False`.
        TypeError: Encoder model is not `XCiT`.
        ValueError: Loader `batch_size` is `None`.
        TypeError: The first or only batch element is not a batch of image tensors.
    """
    output_dir = _prepare_output_dir(output_dir, overwrite=overwrite)
    batches = []
    parquet_index = 1
    for batch in tqdm.tqdm(
        yield_features(encoder=encoder, loader=loader),
        desc="Extracting features",
        disable=not verbose,
    ):
        batches.append(batch)
        if len(batches) * loader.batch_size >= max_samples:
            _collect_to_dataframe(
                batches=batches, dataset=loader.dataset
            ).write_parquet(file=output_dir / f"features_{parquet_index}.parquet")
            batches = []
            parquet_index += 1
    if len(batches) > 0:
        _collect_to_dataframe(batches=batches, dataset=loader.dataset).write_parquet(
            file=output_dir / f"features_{parquet_index}.parquet"
        )


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    if output_dir.exists() and len(list(output_dir.iterdir())) > 0:
        if not overwrite:
            raise FileExistsError(ERROR_OUTPUT_EXISTS)
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir


def _collect_to_dataframe(
    batches: list[Union[Tensor, tuple[Tensor, ...]]], dataset: Dataset
) -> pl.DataFrame:
    data = {}
    for batch in batches:
        for key, val in _batch_to_columns(batch, dataset).items():
            if key not in data:
                data[key] = []
            data[key].append(val)
    return pl.DataFrame({k: np.concatenate(v, axis=0) for k, v in data.items()})


def _batch_to_columns(
    batch: Union[Tensor, tuple[Tensor, ...]], dataset: Dataset
) -> dict[str, np.ndarray]:
    if isinstance(batch, Tensor):
        return _unpack_features(batch)
    output = {}
    if isinstance(dataset, TileImageDataset):
        # (feats, paths, *extras)
        output.update(_unpack_paths(batch[1]))
        output.update(_unpack_coordinates_from_paths(batch[1]))
    elif isinstance(dataset, SlideReaderDataset):
        # (feats, coords)
        output.update(_unpack_coordinates(batch[1]))
    output.update(_unpack_features(batch[0]))
    return output


def _unpack_features(x: Tensor) -> dict[str, np.ndarray]:
    return {f"feat{i+1}": x[:, i].numpy() for i in range(x.shape[1])}


def _unpack_coordinates(x: Tensor) -> dict[str, np.ndarray]:
    return {name: x[:, i] for i, name in enumerate(list("xywh"))}


def _unpack_paths(x: list[str]) -> dict[str, np.ndarray]:
    return {"path": np.array(x)}


def _unpack_coordinates_from_paths(paths: list[str]) -> dict[str, np.ndarray]:
    """Collect coordinates from filename if possible."""
    output = {"x": [], "y": [], "w": [], "h": []}
    for path in (Path(x) for x in paths):
        filename = path.name.removesuffix(path.suffix)
        # format: x{coord}_y{coord}_w{coord}_h{coord}
        tile_coords = filename.split("_")
        if len(tile_coords) != len(output):
            for key in list(output.keys()):
                output[key].append(np.nan)
            continue
        for key, text in zip(list("xywh"), tile_coords):
            if text.startswith(key) and text[1:].isalnum():
                output[key].append(int(text[1:]))
            else:
                output[key].append(np.nan)
    return {k: np.array(v, dtype=int) for k, v in output.items()}
