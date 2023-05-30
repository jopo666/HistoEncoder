from collections.abc import Generator
from pathlib import Path

import polars as pl
import rich_click as click
import tqdm

import histoencoder.functional as F

from ._utils import BAR_FORMAT, error

DEFAULTS = {
    "num_clusters": (8, 16, 24, 32, 64, 128),
    "overwrite": False,
    "verbose": False,
}


@click.command()
@click.option(  # input_dir
    "-i",
    "--input_dir",
    metavar="DIRECTORY",
    required=True,
    help="Directory with extracted features.",
    type=click.Path(exists=True, file_okay=False),
    callback=lambda *args: Path(args[-1]),
)
@click.option(  # nclusters
    "-n",
    "--num-clusters",
    type=click.IntRange(min=2),
    multiple=True,
    metavar="INTEGER",
    default=DEFAULTS["num_clusters"],
    show_default=True,
    help="Number of clusters.",
)
@click.option(  # overwrite
    "-z",
    "--overwrite",
    show_default="False",
    is_flag=True,
    help="Overwrite any existing clusters.parquet files.",
)
@click.option(  # verbose
    "-v",
    "--verbose",
    show_default="False",
    is_flag=True,
    help="Enable verbose output from `MiniBatchKmeans`.",
)
def cluster(
    *,
    input_dir: Path,
    num_clusters: tuple[int] = DEFAULTS["num_clusters"],
    overwrite: bool = DEFAULTS["overwrite"],
    verbose: bool = DEFAULTS["verbose"],
) -> None:
    """Cluster extracted features."""
    features = _collect_feature_files(input_dir=input_dir, overwrite=overwrite)
    click.echo(f"Clustering features for {features.shape[0]} tile images")
    columns = []
    for n in num_clusters:
        click.echo(f"... into {n} clusters")
        columns.append(
            F.cluster_features(features=features, n_clusters=n, verbose=verbose)
        )
    clusters = pl.concat([features, *columns], how="horizontal")
    click.echo("Saving cluster information.")
    all_groups = list(clusters.groupby("tmp_path"))
    for path, df in tqdm.tqdm(all_groups, bar_format=BAR_FORMAT):
        columns = [x for x in df.columns if not x.startswith("feat")]
        df[columns].drop("tmp_path").write_parquet(Path(path) / "clusters.parquet")


def _collect_feature_files(
    input_dir: Path, *, overwrite: bool
) -> dict[str, Generator[Path, None, None]]:
    output = []
    click.echo("Collecting extracted features.")
    all_files = list(input_dir.iterdir())
    for path in tqdm.tqdm(all_files, bar_format=BAR_FORMAT):
        feature_file = path / "features.parquet"
        if not feature_file.exists():
            continue
        if (path / "clusters.parquet").exists() and not overwrite:
            error("Found an existing `clusters.parquet` file but `overwrite=False`.")
        output.append(
            pl.read_parquet(feature_file).with_columns(
                pl.lit(str(path)).alias("tmp_path")
            )
        )
    if len(output) == 0:
        error("Could not find any extracted features.")
    return pl.concat(output, how="vertical")
