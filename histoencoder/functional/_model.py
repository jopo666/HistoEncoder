import timm
import torch

ERROR_MODEL_NOT_FOUND = "Model '{}' could not be found, select from: {}"
AVAILABLE_ENCODERS = {
    "prostate_small": "/data/models/prostate_small.pth",
    "prostate_medium": "/data/models/prostate_medium.pth",
    "prostate_large": "/data/models/prostate_large.pth",
}
SIZE_TO_MODEL_NAME = {
    "small": "xcit_small_12_p16_224",
    "medium": "xcit_medium_24_p16_224",
    "large": "xcit_large_24_p16_224",
}


def create_encoder(model_name: str) -> torch.nn.Module:
    """Create XCiT encoder model.

    Args:
        model_name: Name of the encoder model checkpoint.

    Raises:
        ValueError: Model name not found.

    Returns:
        XCiT encoder model.
    """
    model_url = AVAILABLE_ENCODERS.get(model_name.lower())
    if model_url is None:
        raise ValueError(ERROR_MODEL_NOT_FOUND.format(model_name, list_encoders()))
    *_, model_size = model_name.split("_")
    encoder = timm.create_model(
        model_name=SIZE_TO_MODEL_NAME[model_size], num_classes=0
    )
    encoder.load_state_dict(torch.load(model_url))
    return encoder


def list_encoders() -> list[str]:
    return list(AVAILABLE_ENCODERS.keys())
