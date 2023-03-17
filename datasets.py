import numpy as np
import torch
import torchvision
from torchvision.datasets import OxfordIIITPet


def load_dataset(root, split):
    """
    returns a Dataset with:
    - images as normalized float32 tensors
    - labels as uint 8 tensors
    
    the labels are:
    - 1: foreground
    - 2: background
    - 3: not classified
    """
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    label_transform = lambda label: torch.from_numpy(np.array(label))
    ds = OxfordIIITPet(root=root, split=split, target_types='segmentation', download=True, transform=image_transform, target_transform=label_transform)
    return ds


def create_datasets(root='/tmp/adl_data', valid_frac=0.2, labelled_frac=0.125):
    trainval_ds = load_dataset(root, 'trainval')
    test_ds = load_dataset(root, 'test')

    rng = torch.Generator()
    rng.manual_seed(0)
    train_all_ds, valid_ds = torch.utils.data.random_split(trainval_ds, (1-valid_frac, valid_frac), generator=rng)
    _, train_lab_ds = torch.utils.data.random_split(train_all_ds, (1-labelled_frac, labelled_frac), generator=rng)
    
    return train_all_ds, train_lab_ds, valid_ds