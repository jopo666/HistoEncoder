import torch
from timm.models.xcit import XCiT

from ._check import check_encoder

NO_DECAY = 1.0
LR_SCALE_INDEX_ZERO = ("cls_token", "patch_embed", "pos_embed")
ERROR_DECAY = "Learning rate decay should be in range (0, 1], got '{}'."


def get_parameter_groups(
    encoder: XCiT,
    *,
    lr: float = 0.001,
    weight_decay: float = 0.05,
    lr_decay: float = 0.75,
    filter_1d: bool = True,
) -> list[dict]:
    """Prepare XCiT encoder parameter groups for an optimizer.

    Args:
        encoder: XCiT encoder model.
        lr: Learning rate.
        weight_decay: Weight decay.
        lr_decay: Learning rate decay (per attention blocks). Defaults to 0.75.
        filter_1d: Filter 1 dimensional parameters from weight decay. Defaults to True.

    Returns:
        Parameter groups.
    """
    encoder = check_encoder(encoder)
    if not 0 < lr_decay <= 1:
        raise ValueError(ERROR_DECAY.format(lr_decay))
    num_param_groups = len(encoder.blocks) + len(encoder.cls_attn_blocks) + 1
    layer_lr_scales = [
        lr_decay ** (num_param_groups - i) for i in range(num_param_groups + 1)
    ]
    parameter_groups = {}
    for name, param in encoder.named_parameters():
        if not param.requires_grad:
            continue
        # Determine weight decay.
        if name in encoder.no_weight_decay() or (param.ndim == 1 and filter_1d):
            decay_name = "no_decay"
            group_weight_decay = 0.0
        else:
            decay_name = "decay"
            group_weight_decay = weight_decay
        # Determine decay index.
        if lr_decay == NO_DECAY or name.startswith(LR_SCALE_INDEX_ZERO):
            scale_index = 0
        elif name.startswith("blocks"):
            # blocks.{number}.{name}
            scale_index = int(name.split(".")[1]) + 1
        elif name.startswith("cls_attn_blocks"):
            # cls_attn_blocks.{number}.{name}
            scale_index = int(name.split(".")[1]) + len(encoder.blocks) + 1
        else:
            scale_index = num_param_groups
        group_name = f"params_{scale_index}_{decay_name}"
        # Initialize group.
        if group_name not in parameter_groups:
            parameter_groups[group_name] = {
                "name": group_name,
                "lr": lr * layer_lr_scales[scale_index],
                "lr_scale": layer_lr_scales[scale_index],
                "weight_decay": group_weight_decay,
                "params": [],
            }
        # Add parameter.
        parameter_groups[group_name]["params"].append(param)
    return list(parameter_groups.values())


def update_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    """Update optimizer weight decay."""
    for param_group in optimizer.param_groups:
        if "no_decay" not in param_group.get("name", "no_name"):
            param_group["weight_decay"] = weight_decay


def update_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Update optimizer learning rate (taking `lr_scale` into consideration)."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group.get("lr_scale", 1.0) * lr
