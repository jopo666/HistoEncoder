__all__ = ["yield_features"]

from collections.abc import Generator, Iterable
from typing import Any, Union

import torch
from torch import Tensor

ERROR_NON_BATCHED = "Batch size should not be None."
ERROR_BATCH_TYPE = "Batch should be a tensor, or a list/tuple."
ERROR_ELEMENT = "Expected the first batch element to be a tensor."


@torch.no_grad()
def yield_features(
    encoder: torch.nn.Module, loader: Iterable[Union[Tensor, tuple[Tensor, ...]]]
) -> Generator[Union[Tensor, tuple[Tensor, ...]], None, None]:
    """Yield encoded features by replacing batch images with encoded features."""
    if loader.batch_size is None:
        raise ValueError(ERROR_NON_BATCHED)
    encoder.eval()
    model_device = next(encoder.parameters()).device
    for batch in loader:
        batch_images, *batch_extras = _unpack_batch(batch)
        if model_device.type != batch_images.type:
            batch_images = batch_images.to(model_device)
        features = encoder(batch_images).cpu()
        if isinstance(batch, Tensor):
            yield features
        else:
            yield features, *batch_extras
    encoder.train()


def _unpack_batch(batch: Union[torch.Tensor, tuple, list]) -> tuple[Tensor, Any]:
    if isinstance(batch, Tensor):
        return (batch,)
    if isinstance(batch, (tuple, list)):
        if len(batch) > 0 and isinstance(batch[0], Tensor):
            return batch
        raise TypeError(ERROR_ELEMENT)
    raise TypeError(ERROR_BATCH_TYPE)
