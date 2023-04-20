from ._utils import DATA_DIR, TILE_DIR

COMMAND = "HistoEncoder extract"
DEFAULTS = "-m prostate_small -j 8 -b 2 --num-samples 10 -c"


def test_from_processed(script_runner) -> None:  # noqa
    feature_path = DATA_DIR / "slide" / "features.parquet"
    if feature_path.exists():
        feature_path.unlink()
    ret = script_runner.run(
        *f"{COMMAND} from-processed -i {DATA_DIR} {DEFAULTS}".split(" ")
    )
    assert ret.success
    assert feature_path.exists()
    feature_path.unlink()

    assert ret.stdout.split("\n")[:-1] == [
        "INFO: Encoder created ('prostate_small').",
        "INFO: Collecting tile images from '/data/jopo/HistoEncoder/tests/data'.",
        "INFO: Extracting features for 1 processed slides.",
        "INFO: [1/1] Processing '/data/jopo/HistoEncoder/tests/data/slide/tiles/*'",
        "INFO: Found 90 tile images.",
        "INFO: Estimating mean and std with 10 samples.",
    ]
    assert "Extracting features" in ret.stderr


def test_from_pattern(script_runner, tmp_path) -> None:  # noqa
    output_dir = tmp_path / "features"
    ret = script_runner.run(
        *f"{COMMAND} from-pattern -i {TILE_DIR / '*'} -o {output_dir} {DEFAULTS}".split(
            " "
        )
    )
    assert ret.success
    assert ret.stdout.split("\n")[:-1] == [
        "INFO: Encoder created ('prostate_small').",
        "INFO: Globbing '/data/jopo/HistoEncoder/tests/data/slide/tiles/*'.",
        "INFO: Found 90 files matching pattern '/data/jopo/HistoEncoder/tests/data/slide/tiles/*'.",  # noqa
        "INFO: Estimating mean and std with 10 samples.",
    ]
    assert "Extracting features" in ret.stderr
