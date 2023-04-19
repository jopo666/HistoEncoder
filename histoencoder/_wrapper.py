from collections.abc import Generator
from pathlib import Path
from typing import Union

from torch import Tensor
from torch.utils.data import DataLoader

import histoencoder.functional as F


class HistoEncoder:
    """Lightweight wrapper around the `XCiT` encoder model and
    `histoencoder.functionals`, which have been implemented as class methods."""

    def __init__(self, model_name: str) -> None:
        """Create a model based on given name.

        Args:
            model_name: Name of the encoder model.

        Raises:
            ValueError: Model name not found.

        Returns:
            Requested encoder model.
        """
        super().__init__()
        self.__model_name = model_name.lower()
        self.encoder = F.create_encoder(model_name)

    @property
    def model_name(self) -> str:
        """Name of the encoder model checkpoint."""
        return self.__model_name.lower()

    @staticmethod
    def list_encoders() -> list[str]:
        """List all available model checkpoints."""
        return F.list_encoders()

    def yield_features(
        self, loader: DataLoader
    ) -> Generator[Union[Tensor, tuple[Tensor, ...]], None, None]:
        """Yield features for images in `loader` by replacing images in the batch with
        features.

        Args:
            loader: `DataLoader` yielding a batches with images as the first or only
                element.

        Raises:
            ValueError: Loader `batch_size` is `None`.
            TypeError: The first or only batch element is not a batch of image tensors.

        Yields:
            Loader batches with images replaced by features extracted by the encoder.
        """
        yield from F.yield_features(encoder=self.encoder, loader=loader)

    def save_features(
        self,
        output_dir: Union[str, Path],
        loader: DataLoader,
        *,
        max_samples: int = 2**16,
        overwrite: bool = False,
        verbose: bool = True,
    ) -> None:
        """Write features to disk.

        Args:
            encoder: XCiT encoder model for extracting features.
            output_dir: Output directory for feature parquet files.
            loader: `DataLoader` yielding tensor images as the first or only element.
            max_samples: Maximum samples per parquet file. Defaults to 65536.
            overwrite: Remove everything in output directory. Defaults to False.
            verbose: Enable `tqdm.tqdm` progress bar. Defaults to True.

        Raises:
            FileExistsError: Output directory exists but `overwrite=False`.
            TypeError: Encoder model is not `XCiT`.
            ValueError: Loader `batch_size` is `None`.
            TypeError: The first or only batch element is not a batch of image tensors.
        """
        F.save_features(
            encoder=self.encoder,
            output_dir=output_dir,
            loader=loader,
            max_samples=max_samples,
            overwrite=overwrite,
            verbose=verbose,
        )

    def freeze_encoder(
        self,
        num_liquid: int = 0,
        *,
        freeze_cls_token: bool = True,
        freeze_patch_embed: bool = True,
        freeze_pos_embed: bool = True,
        freeze_layer_norm: bool = True,
        freeze_last_mlp_layer: bool = False,
    ) -> None:
        """Freeze encoder parameters for finetuning.

        Args:
            num_liquid: Number of liquid attention blocks. Defaults to `0`.
            freeze_cls_token: Freeze `cls_token` parameters. Defaults to `True`.
            freeze_patch_embed: Freeze `patch_embed` parameters. Defaults to `True`.
            freeze_pos_embed: Freeze `pos_embed` parameters. Defaults to `True`.
            freeze_layer_norm: Freeze `layer_norm` parameters. Defaults to `True`.
            freeze_last_mlp_layer: Freeze the last mlp-layer in the last cls attention
                block. Defaults to `False`.
        """
        F.freeze_encoder(
            encoder=self.encoder,
            num_liquid=num_liquid,
            freeze_cls_token=freeze_cls_token,
            freeze_patch_embed=freeze_patch_embed,
            freeze_pos_embed=freeze_pos_embed,
            freeze_layer_norm=freeze_layer_norm,
            freeze_last_mlp_layer=freeze_last_mlp_layer,
        )

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(name='{self.model_name}')"
