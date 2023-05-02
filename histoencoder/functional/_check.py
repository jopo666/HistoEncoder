from timm.models.xcit import XCiT

ERROR_NOT_XCIT = "Expected encoder to be XCiT model, got {}."


def check_encoder(encoder: XCiT) -> XCiT:
    if not isinstance(encoder, XCiT):
        raise TypeError(ERROR_NOT_XCIT.format(type(encoder).__name__))
    return encoder
