from typing import Union

from timm.models.xcit import XCiT

from histoencoder._wrapper import HistoEncoder

ERROR_NOT_XCIT = "Expected encoder to be XCiT model, got {}."


def check_encoder(encoder: Union[XCiT, HistoEncoder]) -> XCiT:
    if isinstance(encoder, HistoEncoder):
        encoder = encoder.encoder
    if not isinstance(encoder, XCiT):
        raise TypeError(ERROR_NOT_XCIT.format(type(encoder).__name__))
    return encoder
