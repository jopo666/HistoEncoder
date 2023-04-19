import pytest
from timm.models.xcit import XCiT

import histoencoder.functional as F


def test_create_all_models() -> None:
    for model_name in F.list_encoders():
        model = F.create_encoder(model_name)
        assert isinstance(model, XCiT)


def test_bad_model_name() -> None:
    with pytest.raises(
        ValueError, match="Model 'resnet50' could not be found, select from:"
    ):
        F.create_encoder("resnet50")
