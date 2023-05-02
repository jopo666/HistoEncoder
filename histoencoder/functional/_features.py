from collections.abc import Generator, Iterable
from typing import Any, Union

import torch
from timm.models.xcit import XCiT
from torch import Tensor

from ._check import check_encoder

ERROR_NON_BATCHED = "Batch size should not be None."
ERROR_BATCH_TYPE = "Batch should be a tensor, or a list/tuple."
ERROR_ELEMENT = "Expected the first batch element to be a tensor."


@torch.no_grad()
def yield_features(
    encoder: XCiT,
    loader: Iterable[Union[Tensor, tuple[Tensor, ...]]],
    *,
    num_blocks: int = 1,
    avg_pool: bool = False,
) -> Generator[Union[Tensor, tuple[Tensor, ...]], None, None]:
    """Yield features for images in `loader` by replacing images in the batch with
    features.

    Args:
        encoder: XCiT encoder model.
        loader: `DataLoader` yielding a batches with images as the first or only
            element.
        num_blocks: Number of attention blocks to include in the extracted features.
            When `num_blocks>1`, the outputs of the last `num_blocks` attention blocks
            are concatenated to make up the features. Defaults to 1.
        avg_pool: Whether to concat the global average pool of the final attention block
            features to the extracted features. Defaults to False.

    Raises:
        TypeError: Encoder model is not `XCiT`.
        ValueError: Loader `batch_size` is `None`.
        TypeError: The first or only batch element is not a batch of image tensors.

    Yields:
        Loader batches with images replaced by features extracted by the encoder.
    """
    encoder = check_encoder(encoder)
    if loader.batch_size is None:
        raise ValueError(ERROR_NON_BATCHED)
    set_to_train = encoder.training
    encoder.eval()
    model_device = next(encoder.parameters()).device
    for batch in loader:
        batch_images, *batch_extras = _unpack_batch(batch)
        if model_device.type != batch_images.type:
            batch_images = batch_images.to(model_device)
        features = extract_features(
            encoder, batch_images, num_blocks=num_blocks, avg_pool=avg_pool
        ).cpu()
        if isinstance(batch, Tensor):
            yield features
        else:
            yield features, *batch_extras
    if set_to_train:
        encoder.train()


def extract_features(
    encoder: XCiT,
    images: Tensor,
    *,
    num_blocks: int = 1,
    avg_pool: bool = False,
) -> Tensor:
    """Extract features for a batch of images.

    Args:
        encoder: Encoder model for extracting features.
        images: Batch of images.
        num_blocks: Number of attention blocks to include in the extracted features.
            When `num_blocks>1`, the outputs of the last `num_blocks` attention blocks
            are concatenated to make up the features. Defaults to 1.
        avg_pool: Whether to concat the global average pool of the final attention block
            features to the extracted features. Defaults to False.

    Returns:
        Extracted features for the batch of images.
    """
    B = images.shape[0]
    x, (Hp, Wp) = encoder.patch_embed(images)
    if encoder.use_pos_embed:
        pos_encoding = (
            encoder.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
        )
        x = x + pos_encoding
    x = encoder.pos_drop(x)
    # Collect intermediate outputs.
    intermediate_outputs = []
    for blk in encoder.blocks:
        x = blk(x, Hp, Wp)
        intermediate_outputs.append(x)
    cls_tokens = encoder.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    for blk in encoder.cls_attn_blocks:
        x = blk(x)
        intermediate_outputs.append(x)
    # Collect actual output based on num_blocks.
    norm_outputs = [encoder.norm(x) for x in intermediate_outputs[-num_blocks:]]
    output = torch.cat([x[:, 0] for x in norm_outputs], axis=-1)
    if avg_pool:
        output = torch.cat(
            [output, torch.mean(norm_outputs[-1][:, 1:], dim=1)], axis=-1
        )
    return output


def _unpack_batch(batch: Union[torch.Tensor, tuple, list]) -> tuple[Tensor, Any]:
    if isinstance(batch, Tensor):
        return (batch,)
    if isinstance(batch, (tuple, list)):
        if len(batch) > 0 and isinstance(batch[0], Tensor):
            return batch
        raise TypeError(ERROR_ELEMENT)
    raise TypeError(ERROR_BATCH_TYPE)
