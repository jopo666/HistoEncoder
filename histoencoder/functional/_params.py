__all__ = ["prepare_parameter_groups", "update_weight_decay", "update_lr"]

import torch


def prepare_parameter_groups(
    model: torch.nn.Module,
    *,
    lr: float = 0.001,
    weight_decay: float = 0.05,
    lr_decay: float = 0.75,
    filter_1d: bool = True,
) -> list[dict]:
    """Prepare `histoencoder` model parameterss for an optimizer.

    Args:
        model: Model to be optimizer.
        lr: Learning rate.
        weight_decay: Weight decay.
        lr_decay: Learning rate decay for ViT models. Defaults to 0.75.
        filter_1d: Filter 1 dimensional params from weight decay. Defaults to True.

    Returns:
        Parameter groups
    """
    num_layers = (
        len(model.blocks) + len(model.cls_attn_blocks) + 1 if lr_decay < 1.0 else 1
    )
    layer_scales = [lr_decay ** (num_layers - i) for i in range(num_layers + 1)]
    # Define no weight decay list.
    no_weight_decay = (
        model.no_weight_decay() if hasattr(model, "no_weight_decay") else []
    )
    # Collect all parameters.
    parameter_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Determine weight decay.
        if name in no_weight_decay or (param.ndim == 1 and filter_1d):
            decay_name = "no_decay"
            layer_weight_decay = 0.0
        else:
            decay_name = "decay"
            layer_weight_decay = weight_decay
        # Determine block index.
        if (
            any(x in name for x in ["cls_token", "patch_embed", "pos_embed"])
            or num_layers == 1
        ):
            block_idx = 0
        elif name.startswith("blocks"):
            block_idx = int(name.split(".")[1]) + 1
        elif name.startswith("cls_attn_blocks"):
            block_idx = int(name.split(".")[1]) + len(model.blocks) + 1
        else:
            block_idx = num_layers
        # Define group name.
        group_name = f"layer_{block_idx}_{decay_name}"
        # Initialize group.
        if group_name not in parameter_groups:
            parameter_groups[group_name] = {
                "name": group_name,
                "lr": lr * layer_scales[block_idx],
                "lr_scale": layer_scales[block_idx],
                "weight_decay": layer_weight_decay,
                "params": [],
                "names": [],  # Just used for debugging.
            }
        # Add parameter.
        parameter_groups[group_name]["params"].append(param)
        parameter_groups[group_name]["names"].append(name)
    return list(parameter_groups.values())


def update_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    """Update optimizer weight decay. Useful for using custom schedules."""
    for param_group in optimizer.param_groups:
        if "no_decay" not in param_group.get("name", "no_name"):
            param_group["weight_decay"] = weight_decay


def update_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Update optimizer learning rate. Useful for using custom schedules."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group.get("lr_scale", 1.0) * lr
