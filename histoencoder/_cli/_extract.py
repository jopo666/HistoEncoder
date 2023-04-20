import glob
import os
import random
from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import rich_click as click
import torch
import torchvision.transforms as T
from histoprep.functional import get_mean_and_std_from_paths
from histoprep.utils import TileImageDataset
from PIL import Image
from timm.models.xcit import XCiT
from torch.utils.data import DataLoader

import histoencoder.functional as F

from ._utils import error, info, info_exit, warning

DEFAULTS = {
    "use_cpu": False,
    "overwrite": False,
    "input_size": 224,
    "batch_size": 16,
    "num_samples": 1000,
    "max_samples": 2**16,
    "num_workers": os.cpu_count(),
}


@click.group()
def extract_group() -> None:
    pass


@extract_group.group()
def extract() -> None:
    """Extract features with an encoder model."""


@extract.command()
# Required.
@click.option(  # pattern
    "-i",
    "--pattern",
    metavar="PATTERN",
    required=True,
    help="File pattern for images.",
    type=click.STRING,
)
@click.option(  # output_dir
    "-o",
    "--output-dir",
    metavar="DIRECTORY",
    required=True,
    help="Directory for extracted feature parquet files.",
    type=click.Path(file_okay=False),
    callback=lambda *args: Path(args[-1]),
)
# COMMON ARGUMENTS
@click.option(  # model_name
    "-m",
    "--model-name",
    type=click.STRING,
    required=True,
    help="Encoder model for extracting features.",
)
@click.option(  # overwrite
    "-z",
    "--overwrite",
    show_default="False",
    is_flag=True,
    help="Overwrite existing extracted features.",
)
@click.option(  # use_cpu
    "-c",
    "--use-cpu",
    show_default="False",
    is_flag=True,
    help="If not set uses GPU if available.",
)
@click.option(  # input_size
    "-s",
    "--input-size",
    type=click.INT,
    default=DEFAULTS["input_size"],
    show_default=True,
    help="Number of samples for mean & std estimation.",
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
def from_pattern(
    *,
    pattern: str,
    output_dir: Path,
    model_name: str,
    # Common arguments.
    overwrite: bool = DEFAULTS["overwrite"],
    use_cpu: bool = DEFAULTS["use_cpu"],
    input_size: int = DEFAULTS["input_size"],
    batch_size: int = DEFAULTS["batch_size"],
    num_workers: int = DEFAULTS["num_workers"],
    num_samples: int = DEFAULTS["num_samples"],
) -> None:
    """Extract features for images matching pattern."""
    encoder = _prepare_encoder(model_name, use_cpu=use_cpu)
    paths = _glob_pattern(pattern)
    mean, std = _mean_and_std_paths(
        paths, num_samples=num_samples, num_workers=num_workers
    )
    loader = DataLoader(
        TileImageDataset(
            paths=paths,
            transform=_get_transform(input_size=input_size, mean=mean, std=std),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=not use_cpu and torch.cuda.is_available(),
    )
    F.save_features(
        encoder=encoder,
        output_dir=output_dir,
        loader=loader,
        max_samples=2**16,
        overwrite=overwrite,
        verbose=True,
    )


@extract.command()
# Required.
@click.option(  # input_dir
    "-i",
    "--input_dir",
    metavar="DIRECTORY",
    required=True,
    help="Directory with processed slide output",
    type=click.Path(exists=True, file_okay=False),
    callback=lambda *args: Path(args[-1]),
)
# COMMON ARGUMENTS
@click.option(  # model_name
    "-m",
    "--model-name",
    type=click.STRING,
    required=True,
    help="Encoder model for extracting features.",
)
@click.option(  # overwrite
    "-z",
    "--overwrite",
    show_default="False",
    is_flag=True,
    help="Overwrite existing extracted features.",
)
@click.option(  # use_cpu
    "-c",
    "--use-cpu",
    show_default="False",
    is_flag=True,
    help="If not set uses GPU if available.",
)
@click.option(  # input_size
    "-s",
    "--input-size",
    type=click.INT,
    default=DEFAULTS["input_size"],
    show_default=True,
    help="Number of samples for mean & std estimation.",
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
def from_processed(
    *,
    input_dir: Path,
    model_name: str,
    # Common arguments.
    overwrite: bool = DEFAULTS["overwrite"],
    use_cpu: bool = DEFAULTS["use_cpu"],
    input_size: int = DEFAULTS["input_size"],
    batch_size: int = DEFAULTS["batch_size"],
    num_workers: int = DEFAULTS["num_workers"],
    num_samples: int = DEFAULTS["num_samples"],
) -> None:
    """Extract features for histological slides processed with [bold
    blue]HistoPrep[/bold blue]."""
    encoder = _prepare_encoder(model_name, use_cpu=use_cpu)
    slide_info = _collect_slide_info(input_dir, overwrite=overwrite)
    for idx, (slide_dir, slide_tiles) in enumerate(slide_info.items()):
        info(f"{_get_prefix(idx, len(slide_info))} Processing '{slide_dir}/tiles/*'")
        slide_tiles = list(slide_tiles)  # noqa
        info(f"Found {len(slide_tiles)} tile images.")
        mean, std = _mean_and_std_paths(
            paths=slide_tiles, num_samples=num_samples, num_workers=num_workers
        )
        dataset = TileImageDataset(
            slide_tiles, transform=_get_transform(input_size, mean=mean, std=std)
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=not use_cpu and torch.cuda.is_available(),
        )
        F.save_features(
            encoder=encoder,
            output_dir=slide_dir,
            loader=loader,
            overwrite=overwrite,
            max_samples=None,
            verbose=True,
        )


def _mean_and_std_paths(
    paths: list[Path], num_samples: int, num_workers: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(paths) > num_samples:
        info(f"Estimating mean and std with {num_samples} samples.")
        paths = random.sample(paths, k=num_samples)
    else:
        info("Calculating mean & std.")
    return get_mean_and_std_from_paths(paths=paths, num_workers=num_workers)


def _glob_pattern(pattern: str) -> list[Path]:
    info(f"Globbing '{pattern}'.")
    output = [
        z for z in (Path(x) for x in glob.glob(pattern, recursive=True)) if z.is_file()
    ]
    if len(output) == 0:
        error(f"Found no files matching pattern '{pattern}'.")
    info(f"Found {len(output)} files matching pattern '{pattern}'.")
    return output


def _get_prefix(current: int, total: int) -> str:
    total_str = str(total)
    return f"[{str(current + 1).rjust(len(total_str))}/{total_str}]"


def _get_transform(
    input_size: int, mean: list[float], std: list[float]
) -> Callable[[np.ndarray], torch.Tensor]:
    """Generate transform function."""
    return T.Compose(
        [Image.fromarray, T.Resize(input_size), T.ToTensor(), T.Normalize(mean, std)]
    )


def _prepare_encoder(model_name: str, *, use_cpu: bool) -> XCiT:
    if model_name not in F.list_encoders():
        error(
            f"Model name '{model_name}' not recognised, choose from:"
            f"{F.list_encoders()}"
        )
    encoder = F.create_encoder(model_name)
    info(f"Encoder created ('{model_name}').")
    if use_cpu:
        return encoder
    if not torch.cuda.is_available():
        warning("Could not locate any GPU devices, extracting features on the CPU.")
        return encoder
    info("Sending encoder to GPU.")
    return encoder.cuda()


def _collect_slide_info(  # noqa
    input_dir: Path, *, overwrite: bool
) -> dict[str, Generator[Path, None, None]]:
    """Collect slide directories and exit on possible errors."""
    if not input_dir.exists():
        error(f"'{str(input_dir)}' does not exist.")
    if not input_dir.is_dir():
        error(f"'{str(input_dir)}' isn't a directory.")
    output = []
    num_overwrite, num_processed = 0, 0
    info(f"Collecting tile images from '{input_dir}'.")
    for path in input_dir.iterdir():
        if (path / "tiles").exists():
            if (path / "features.parquet").exists():
                num_processed += 1
                if not overwrite:
                    continue
                num_overwrite += 1
            output.append(path)
    if len(output) == 0:
        if num_processed == 0:
            error(f"Could not find any processed slides in '{input_dir}'.")
        info_exit(
            f"Features have been extracted for all processed slides in '{input_dir}'."
        )
    if num_overwrite > 0:
        warning(f"Overwriting {num_overwrite} existing feature parquet file(s).")
    elif num_processed > 0:
        info(f"Found {num_processed} processed slides with extracted features.")
    info(f"Extracting features for {len(output)} processed slides.")
    return {x: (x / "tiles").iterdir() for x in output}
