import functools
import os
import random
import tempfile
from pathlib import Path
from typing import Optional

import polars as pl
import rich_click as click
import torch
import torchvision.transforms as T
import tqdm
from histoprep.functional import get_mean_and_std_from_paths
from histoprep.utils import TileImageDataset
from PIL import Image
from timm.models.xcit import XCiT
from torch.utils.data import DataLoader
from tqdm.contrib.concurrent import process_map

import histoencoder.functional as F

from ._utils import BAR_FORMAT, error, warning

DEFAULTS = {
    "use_cpu": False,
    "num_blocks": 1,
    "avg_pool": False,
    "overwrite": False,
    "pool_tiles": False,
    "input_size": 224,
    "batch_size": 16,
    "num_samples": 1000,
    "max_samples": 2**16,
    "num_workers": os.cpu_count(),
}


@click.command()
@click.option(  # input_dir
    "-i",
    "--input_dir",
    metavar="DIRECTORY",
    required=True,
    help="Directory with processed slide outputs.",
    type=click.Path(exists=True, file_okay=False),
    callback=lambda *args: Path(args[-1]),
)
@click.option(  # model_name
    "-m",
    "--model-name",
    type=click.STRING,
    required=True,
    help="Encoder model for extracting features.",
)
@click.option(  # num-blocks
    "-n",
    "--num-blocks",
    type=click.INT,
    default=DEFAULTS["num_blocks"],
    show_default=True,
    help="Number of attention block outputs to save.",
)
@click.option(  # avg-pool
    "-a",
    "--avg-pool",
    show_default="False",
    is_flag=True,
    help="Add global average pool to features.",
)
@click.option(  # pool_tiles
    "-p",
    "--pool-tiles",
    show_default="False",
    is_flag=True,
    help="Pool all tile images before running inference.",
)
@click.option(  # overwrite
    "-z",
    "--overwrite",
    show_default="False",
    is_flag=True,
    help="Overwrite any existing extracted features.",
)
@click.option(  # use_cpu
    "-c",
    "--use-cpu",
    show_default="False",
    is_flag=True,
    help="Disable automatic GPU detection.",
)
@click.option(  # input_size
    "-s",
    "--input-size",
    type=click.INT,
    default=DEFAULTS["input_size"],
    show_default=True,
    help="Input image size for the encoder.",
)
@click.option(  # batch_size
    "-b",
    "--batch-size",
    type=click.INT,
    default=DEFAULTS["batch_size"],
    show_default=True,
    help="Batch size for dataloader.",
)
@click.option(  # num_workers
    "-j",
    "--num-workers",
    type=click.INT,
    default=DEFAULTS["num_workers"],
    show_default=True,
    help="Number data loading processes.",
)
@click.option(  # num_samples
    "--num-samples",
    type=click.INT,
    default=DEFAULTS["num_samples"],
    show_default=True,
    help="Number samples for estimating mean & std.",
)
def extract(
    *,
    input_dir: Path,
    model_name: str,
    num_blocks: int = DEFAULTS["num_blocks"],
    avg_pool: bool = DEFAULTS["avg_pool"],
    overwrite: bool = DEFAULTS["overwrite"],
    use_cpu: bool = DEFAULTS["use_cpu"],
    input_size: int = DEFAULTS["input_size"],
    batch_size: int = DEFAULTS["batch_size"],
    num_workers: int = DEFAULTS["num_workers"],
    num_samples: int = DEFAULTS["num_samples"],
    pool_tiles: bool = DEFAULTS["pool_tiles"],
) -> None:
    """Extract features for histological slides."""
    encoder = prepare_encoder(model_name, use_cpu=use_cpu)
    tile_directories = collect_tile_directories(
        input_dir, overwrite=overwrite, num_workers=num_workers
    )
    save_kwargs = {
        "encoder": encoder,
        "num_blocks": num_blocks,
        "avg_pool": avg_pool,
        "overwrite": True,
    }
    loader_kwargs = {
        "input_size": input_size,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "num_workers": num_workers,
        "use_cpu": use_cpu,
    }
    if pool_tiles:
        tile_paths = pool_tile_images(tile_directories, num_workers=num_workers)
        click.echo(
            f"Extracting features for {len(tile_paths)/1e6:.3f} million tile images."
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            loader = prepare_loader(tile_paths=tile_paths, **loader_kwargs)
            F.save_features(
                output_dir=tmp_dir,
                loader=loader,
                verbose=True,
                max_samples=2**16,
                **save_kwargs,
                bar_format=BAR_FORMAT,
            )
            unpack_features(input_dir=input_dir, feature_dir=Path(tmp_dir))
    else:
        click.echo(f"Extracting features for {len(tile_directories)} tile directories.")
        for tile_dir in tqdm.tqdm(tile_directories, bar_format=BAR_FORMAT):
            loader = prepare_loader(
                tile_paths=list(tile_dir.iterdir()), **loader_kwargs
            )
            F.save_features(output_dir=tile_dir.parent, loader=loader, **save_kwargs)


def prepare_encoder(model_name: str, *, use_cpu: bool) -> XCiT:
    if model_name not in F.list_encoders():
        error(
            f"Model name '{model_name}' not recognised, choose from:"
            f"{F.list_encoders()}"
        )
    click.echo(f"Creating encoder '{model_name}'.")
    encoder = F.create_encoder(model_name)
    if use_cpu:
        return encoder
    if not torch.cuda.is_available():
        warning("Could not locate any GPU devices, extracting features on the CPU.")
        return encoder
    return encoder.cuda()


def collect_tile_directories(
    input_dir: Path, *, num_workers: int, overwrite: bool
) -> list[Path]:
    click.echo(f"Collecting tile directories from '{input_dir}/*'.")
    all_files = list(input_dir.iterdir())
    output = []
    for path in process_map(
        functools.partial(_get_tile_directory, overwrite=overwrite),
        all_files,
        chunksize=_get_chucksize(len(all_files), num_workers=num_workers),
        max_workers=num_workers,
        bar_format=BAR_FORMAT,
    ):
        if path is not None:
            output.append(path)
    if len(output) == 0:
        error("Could not find any unprocessed tile images.")
    return output


def pool_tile_images(
    tile_directories: list[Path], num_workers: int
) -> tuple[list[str], list[str]]:
    click.echo("Pooling all tile images.")
    output = []
    for tile_paths in process_map(
        _list_paths,
        tile_directories,
        bar_format=BAR_FORMAT,
        chunksize=_get_chucksize(len(tile_directories), num_workers=num_workers),
        max_workers=num_workers,
    ):
        output.extend(tile_paths)
    return output


def _list_paths(tile_dir: Path) -> list[str]:
    return list(tile_dir.iterdir())


def unpack_features(input_dir: Path, feature_dir: Path) -> None:
    click.echo("Loading extracted features from temporary directory.")
    dataframe = pl.read_parquet(feature_dir / "features*.parquet")
    dataframe = dataframe.with_columns(
        pl.Series(
            "slide_name",
            [
                # path: .../slide_name/tiles/tile_image.img
                os.path.basename(os.path.dirname(os.path.dirname(x)))  # noqa: faster
                for x in dataframe["path"]
            ],
        )
    )
    click.echo("Saving features to corresponding slide directories.")
    for name, data in tqdm.tqdm(
        dataframe.groupby("slide_name"),
        total=len(dataframe["slide_name"].unique()),
        bar_format=BAR_FORMAT,
    ):
        data.drop("slide_name").clone().write_parquet(
            input_dir / name / "features.parquet"
        )


def prepare_loader(
    tile_paths: list[str],
    *,
    input_size: int,
    batch_size: int,
    num_samples: int,
    num_workers: int,
    use_cpu: bool,
) -> DataLoader:
    """Prepare dataloader."""
    mean, std = get_mean_and_std_from_paths(
        paths=random.sample(tile_paths, k=num_samples)
        if len(tile_paths) > num_samples
        else tile_paths,
        num_workers=num_workers,
    )
    transform = T.Compose(
        [Image.fromarray, T.Resize(input_size), T.ToTensor(), T.Normalize(mean, std)]
    )
    return DataLoader(
        TileImageDataset(tile_paths, transform=transform),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=not use_cpu and torch.cuda.is_available(),
    )


def _get_tile_directory(slide_dir: Path, *, overwrite: bool) -> Optional[Path]:
    """Paralliseable tile directory checker."""
    tile_dir = slide_dir / "tiles"
    if tile_dir.exists():
        if overwrite:
            return tile_dir
        if not (slide_dir / "features.parquet").exists():
            return tile_dir
    return None


def _get_chucksize(total: int, num_workers: int) -> int:
    chunksize, extra = divmod(total, num_workers * 4)
    if extra:
        return chunksize + 1
    return chunksize
