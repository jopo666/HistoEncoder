__all__ = ["create_encoder", "list_encoders"]

import timm
import torch

ERROR_MODEL_NOT_FOUND = "Model '{}' could not be found, select from: {}"
MODEL_STATE_DICTS = {
    "prostate_small": "/data/models/prostate_small.pth",
    "prostate_medium": "/data/models/prostate_medium.pth",
    "prostate_large": "/data/models/prostate_large.pth",
}
AVAILABLE_MODELS = list(MODEL_STATE_DICTS.keys())
SIZE_TO_MODEL_NAME = {
    "small": "xcit_small_12_p16_224",
    "medium": "xcit_medium_24_p16_224",
    "large": "xcit_large_24_p16_224",
}


def create_encoder(model_name: str) -> tuple[torch.nn.Module, str]:
    """Create encoder model and load state_dict from model url."""
    model_url = MODEL_STATE_DICTS.get(model_name)
    if model_url is None:
        raise ValueError(ERROR_MODEL_NOT_FOUND.format(model_name, AVAILABLE_MODELS))
    *_, model_size = model_name.split("_")
    encoder = timm.create_model(
        model_name=SIZE_TO_MODEL_NAME[model_size], num_classes=0
    )
    encoder.load_state_dict(torch.load(model_url))
    return encoder, model_url


def list_encoders():
    return AVAILABLE_MODELS
