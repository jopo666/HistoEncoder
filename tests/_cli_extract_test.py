from ._utils import DATA_DIR

COMMAND = "HistoEncoder extract"
DEFAULTS = "-m prostate_small -j 8 -b 2 --num-samples 10 -c"


def test_extract_sequential(script_runner) -> None:  # noqa
    feature_path = DATA_DIR / "slide" / "features.parquet"
    if feature_path.exists():
        feature_path.unlink()
    ret = script_runner.run(*f"{COMMAND} -i {DATA_DIR} {DEFAULTS}".split(" "))
    assert ret.success
    assert feature_path.exists()
    feature_path.unlink()
    assert ret.stdout.split("\n")[:-1] == [
        "Creating encoder 'prostate_small'.",
        "Collecting tile directories from '/data/jopo/HistoEncoder/tests/data/*'.",
        "Extracting features for 1 tile directories.",
    ]
    assert "100%|####################|" in ret.stderr


def test_extract_parallel(script_runner) -> None:  # noqa
    feature_path = DATA_DIR / "slide" / "features.parquet"
    if feature_path.exists():
        feature_path.unlink()
    ret = script_runner.run(*f"{COMMAND} -i {DATA_DIR} -p {DEFAULTS}".split(" "))
    assert ret.success
    assert feature_path.exists()
    feature_path.unlink()
    assert ret.stdout.split("\n")[:-1] == [
        "Creating encoder 'prostate_small'.",
        "Collecting tile directories from '/data/jopo/HistoEncoder/tests/data/*'.",
        "Pooling all tile images.",
        "Extracting features for 0.000 million tile images.",
        "Loading extracted features from temporary directory.",
        "Saving features to corresponding slide directories.",
    ]
    assert "100%|####################|" in ret.stderr
