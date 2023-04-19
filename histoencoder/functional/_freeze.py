from timm.models.xcit import XCiT

from ._check import check_encoder

ERROR_NOT_XCIT = "Expected encoder to be XCiT model, got {}."
ERROR_DECAY = "Learning rate decay should be in range (0, 1], got {}."
NO_DECAY = 1.0
LR_SCALE_INDEX_ZERO = ("cls_token", "patch_embed", "pos_embed")


def freeze_encoder(
    encoder: XCiT,
    num_liquid: int = 0,
    *,
    freeze_cls_token: bool = True,
    freeze_patch_embed: bool = True,
    freeze_pos_embed: bool = True,
    freeze_layer_norm: bool = True,
    freeze_last_mlp_layer: bool = False,
) -> None:
    """Freeze XCiT encoder parameters.

    Args:
        encoder: XCiT encoder model.
        num_liquid: Number of liquid attention blocks. Defaults to 0.
        freeze_cls_token: Freeze cls_token parameters. Defaults to True.
        freeze_patch_embed: Freeze patch_embed parameters. Defaults to True.
        freeze_pos_embed: Freeze pos_embed parameters. Defaults to True.
        freeze_layer_norm: Freeze layer_norm parameters. Defaults to True.
        freeze_last_mlp_layer: Freeze the last mlp-layer in the last cls attention
            block. Defaults to False.

    Raises:
        TypeError: Encoder model is not `XCiT`.
    """
    encoder = check_encoder(encoder)
    # Calculate number of blocks.
    num_cls_blocks = len(encoder.cls_attn_blocks)
    num_liquid_cls_blocks = min(num_cls_blocks, num_liquid)
    num_blocks = len(encoder.blocks)
    num_liquid_blocks = max(0, num_liquid - num_liquid_cls_blocks)
    # Check if we're not freezing anything.
    if num_blocks + num_cls_blocks <= num_liquid:
        return
    # Define liquid block indices.
    liquid_cls_block_idx = list(reversed(range(num_cls_blocks)))[:num_liquid_cls_blocks]
    liquid_block_idx = list(reversed(range(num_blocks)))[:num_liquid_blocks]
    # Freeze encoder layers.
    for name, param in encoder.named_parameters():
        if name.startswith("cls_attn_blocks"):
            block_idx = int(name.split(".")[1])
            if block_idx in liquid_cls_block_idx or (
                not freeze_last_mlp_layer
                and block_idx == num_cls_blocks - 1
                and "mlp" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif name.startswith("blocks"):
            block_idx = int(name.split(".")[1])
            if block_idx in liquid_block_idx:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif (
            name.startswith("head")  # Head is always liquid
            or (not freeze_cls_token and name.startswith("cls_token"))
            or (not freeze_patch_embed and name.startswith("patch_embed"))
            or (not freeze_pos_embed and name.startswith("pos_embed"))
            or (not freeze_layer_norm and name.startswith("norm"))
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False
