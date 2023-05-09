from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import tqdm
from histoprep.utils import SlideReaderDataset, TileImageDataset
from timm.models.xcit import XCiT
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ._features import yield_features

ERROR_FEATURES_EXISTS = "Output directory contains features but `overwrite=False`."


def save_features(
    encoder: XCiT,
    output_dir: Union[str, Path],
    loader: DataLoader,
    *,
    num_blocks: int = 1,
    avg_pool: bool = False,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    verbose: bool = False,
    **tqdm_kwargs,
) -> None:
    """Write features to disk.

    Args:
        encoder: XCiT encoder model for extracting features.
        output_dir: Output directory for feature parquet files.
        loader: `DataLoader` yielding tensor images as the first or only element.
        num_blocks: Number of attention blocks to include in the extracted features.
            When `num_blocks>1`, the outputs of the last `num_blocks` attention blocks
            are concatenated to make up the features. Defaults to 1.
        avg_pool: Whether to concat the global average pool of the final attention block
            features to the extracted features. Defaults to False.
        max_samples: Maximum samples per parquet file. If `None`, all features are saved
            into a single file. Defaults to 65536.
        overwrite: Remove all parquet files with the word `'features'` in output
            directory. Defaults to False.
        verbose: Enable `tqdm.tqdm` progress bar. Defaults to False.
        tqdm_kwargs: Passed to `tqdm.tqdm`.

    Raises:
        FileExistsError: Output containes features but `overwrite=False`.
        TypeError: Encoder model is not `XCiT`.
        ValueError: Loader `batch_size` is `None`.
        TypeError: The first or only batch element is not a batch of image tensors.
    """
    if "disable" not in tqdm_kwargs:
        tqdm_kwargs["disable"] = not verbose
    if "total" not in tqdm_kwargs:
        tqdm_kwargs["total"] = _try_length(loader)
    output_dir = _prepare_output_dir(output_dir, overwrite=overwrite)
    batches = []
    parquet_index = 1
    for batch in tqdm.tqdm(
        yield_features(
            encoder=encoder, loader=loader, num_blocks=num_blocks, avg_pool=avg_pool
        ),
        **tqdm_kwargs,
    ):
        batches.append(batch)
        if max_samples is not None and len(batches) * loader.batch_size >= max_samples:
            _collect_to_dataframe(
                batches=batches, dataset=loader.dataset
            ).write_parquet(file=output_dir / f"features_{parquet_index}.parquet")
            batches = []
            parquet_index += 1
    if len(batches) > 0:
        name = "features" if parquet_index == 1 else f"features_{parquet_index}"
        _collect_to_dataframe(batches=batches, dataset=loader.dataset).write_parquet(
            file=output_dir / f"{name}.parquet"
        )


def _try_length(loader: DataLoader) -> Optional[int]:
    try:
        return len(loader)
    except AttributeError:
        return None


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    output_dir = output_dir if isinstance(output_dir, Path) else Path(output_dir)
    if output_dir.exists():
        for file in output_dir.iterdir():
            if file.name.startswith("features") and file.name.endswith(".parquet"):
                if not overwrite:
                    raise FileExistsError(ERROR_FEATURES_EXISTS)
                file.unlink()
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
    return {"path": np.array([str(Path(z).resolve(strict=False)) for z in x])}


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
