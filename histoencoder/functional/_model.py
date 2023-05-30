import torch
from timm.models.xcit import (
    XCiT,
)
from timm.models.xcit import (
    xcit_medium_24_p16_224 as prostate_medium,
)
from timm.models.xcit import (
    xcit_small_12_p16_224 as prostate_small,
)

ERROR_MODEL_NOT_FOUND = "Model '{}' could not be found, select from: {}"
MODEL_URLS = {
    "prostate_small": "https://dl.dropboxusercontent.com/s/tbff9wslc8p7ie3/prostate_small.pth?dl=0",
    "prostate_medium": "https://dl.dropboxusercontent.com/s/k1fr09x5auki8sp/prostate_medium.pth?dl=0",
}
NAME_TO_MODEL = {
    "prostate_small": prostate_small,
    "prostate_medium": prostate_medium,
}


def create_encoder(model_name: str) -> XCiT:
    """Create XCiT encoder model.

    Args:
        model_name: Name of the encoder model checkpoint.

    Raises:
        ValueError: Model name not found.

    Returns:
        XCiT encoder model.
    """
    model_url = MODEL_URLS.get(model_name.lower())
    if model_url is None:
        raise ValueError(ERROR_MODEL_NOT_FOUND.format(model_name, list_encoders()))
    encoder = NAME_TO_MODEL[model_name](num_classes=0)
    encoder.load_state_dict(torch.hub.load_state_dict_from_url(model_url))
    return encoder


def list_encoders() -> list[str]:
    return list(MODEL_URLS.keys())
