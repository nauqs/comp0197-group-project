from functools import cache
from typing import Any, Tuple
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import math


# normalizing constants (mean, std) for datasets
# https://github.com/pytorch/vision/blob/b403bfc771e0caf31efd06d43860b09004f4ac61/torchvision/transforms/_presets.py#LL44C35-L44C56
DS_STATS = {
    "classification": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


class OxfordIIITPetCached(OxfordIIITPet):
    @cache
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return super().__getitem__(idx)


def load_dataset(root, split, size=224):
    """
    returns a Dataset with:
    - images as normalized float32 tensors
    - labels as uint 8 tensors

    the labels are:
    - 0: foreground
    - 1: background
    - 2: not classified

    Both images and labels are resized to [size, size].

    Note: we are using a pretrained ResNet with ImageNet weights as the backbone.
    The backbone assumes that input images are normalized to a specific
    mean and std (defined above as 'DS_STATS').
    """

    # define image and label transforms
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*DS_STATS["classification"]),
            torchvision.transforms.Resize([size, size]),
        ]
    )
    label_transform = torchvision.transforms.Compose(
        [
            lambda label: torch.from_numpy(np.array(label)),  # PIL -> uint8 tensor
            lambda label: label - 1,  # shift labels from [1, 2, 3] to [0, 1, 2]
            lambda label: TF.resize(label[None], [size, size])[0],
        ]
    )

    # load dataset
    ds = OxfordIIITPetCached(
        root=root,
        split=split,
        target_types="segmentation",
        download=True,
        transform=image_transform,
        target_transform=label_transform,
    )

    return ds


def create_datasets(root="/tmp/adl_data", valid_frac=0.2, labelled_frac=0.0625):
    """
    Returns:
    - train_all_ds: labeled and unlabeled training images
    - train_lab_ds: labeled training images
    - valid_ds: validation images
    """
    trainval_ds = load_dataset(root, "trainval")
    test_ds = load_dataset(root, "test")

    rng = torch.Generator()
    rng.manual_seed(0)
    train_all_ds, valid_ds = torch.utils.data.random_split(
        trainval_ds, (1 - valid_frac, valid_frac), generator=rng
    )
    train_unlab_ds, train_lab_ds = torch.utils.data.random_split(
        train_all_ds, (1 - labelled_frac, labelled_frac), generator=rng
    )

    return train_all_ds, train_lab_ds, train_unlab_ds, valid_ds, test_ds


def create_dataloaders(batch_size=8, image_size=224, *args, **kwargs):
    """
    Generates datasets using 'create_datasets' and crates a
    DataLoader for each dataset.
    """

    datasets = create_datasets(*args, **kwargs)
    dataloaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets
        ]

    return dataloaders

