from pathlib import Path

import torch
import torchvision.transforms as T
from histoprep import SlideReader
from histoprep.functional import get_mean_and_std_from_paths
from histoprep.utils import SlideReaderDataset, TileImageDataset
from torch.utils.data import DataLoader, Dataset

SLIDE_PATH = Path(__file__).parent / "data" / "slide.jpeg"
TILE_DIR = Path(__file__).parent / "data" / "slide" / "tiles"


class CustomDataset:
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.zeros(3, 224, 224) + index, "random_shit", 666, {"dog": "good_boi"}


def create_custom_loader(batch_size: int = 2) -> Dataset:
    return DataLoader(CustomDataset(), batch_size=batch_size)


def create_reader_loader(batch_size: int = 2, num_samples: int = 8) -> DataLoader:
    reader = SlideReader(SLIDE_PATH)
    __, tissue = reader.get_tissue_mask()
    tile_coords = reader.get_tile_coordinates(tissue, 224)[:num_samples]
    dataset = SlideReaderDataset(
        reader,
        tile_coords,
        transform=T.Compose(
            [T.ToTensor(), T.Normalize(*reader.get_mean_and_std(tile_coords))]
        ),
    )
    return DataLoader(dataset, batch_size=batch_size)


def create_tile_loader(batch_size: int = 2, num_samples: int = 8) -> DataLoader:
    paths = list(TILE_DIR.iterdir())[:num_samples]
    dataset = TileImageDataset(
        paths=paths,
        transform=T.Compose(
            [T.ToTensor(), T.Normalize(*get_mean_and_std_from_paths(paths))]
        ),
    )
    return DataLoader(dataset, batch_size=batch_size)
