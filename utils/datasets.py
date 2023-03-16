import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


class ImageSegmentationDataset(Dataset):
    def __init__(self, filenames, image_size, labelled, root_dir="data"):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "annotations/trimaps")
        self.filenames = filenames
        self.image_size = image_size
        self.labelled = labelled
        self.transforms =  transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize((image_size, image_size)),])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        # Load image and mask
        image_path = os.path.join(self.image_dir, self.filenames[index]+".jpg")
        image = self.transforms(Image.open(image_path))

        # If unlabelled, return image only
        if not self.labelled:
            return image

        # Else, load mask as well and return image and mask
        mask_path = os.path.join(self.mask_dir, self.filenames[index]+".png")
        mask = self.transforms(Image.open(mask_path))

        return image, mask


def get_train_valid_data(root_dir, batch_size, image_size, labelled_fraction=.5, valid_fraction=.2):
    """
    Returns a DataLoader for the labelled-training,  unlabelled-training, and validation datasets.
    """

    print("Loading training and validation data...")

    columns = ["filename", "class_id", "species_id", "breed_id", "split"]
    
    # Load dataset metadata  
    train_val_df = pd.read_csv(os.path.join(root_dir, "annotations/trainval_split.txt"), sep=" ", header=None, names=columns)
    
    # Split in train and validation 80/20 using the train and valid columns
    train_filenames = train_val_df[train_val_df['split'] == 'train']['filename'].values.tolist()
    valid_filenames = train_val_df[train_val_df['split'] == 'valid']['filename'].values.tolist()

    # Split train in labelled and unlabelled
    torch.manual_seed(42)
    train_shuffled_indices = torch.randperm(len(train_filenames)).tolist()
    valid_shuffled_indices = torch.randperm(len(valid_filenames)).tolist()
    train_filenames = [train_filenames[i] for i in train_shuffled_indices]
    valid_filenames = [valid_filenames[i] for i in valid_shuffled_indices]

    train_labelled_filenames = train_filenames[:int(labelled_fraction * len(train_filenames))]
    train_unlabelled_filenames = train_filenames[int(labelled_fraction * len(train_filenames)):]

    # Create training and validation datasets
    train_dataset = ImageSegmentationDataset(train_labelled_filenames, image_size, labelled=True)
    unlabelled_dataset = ImageSegmentationDataset(train_unlabelled_filenames, image_size, labelled=False)
    valid_dataset = ImageSegmentationDataset(valid_filenames, image_size, labelled=True)

    # Create data loaders for each dataset
    labelled_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    unlabelled_train_loader = DataLoader(unlabelled_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return labelled_train_loader, unlabelled_train_loader, valid_loader

def get_test_data(root_dir, batch_size, image_size):
    """
    Returns a DataLoader for the test dataset.
    """

    columns = ["filename", "class_id", "species_id", "breed_id"]
    
    # Load dataset metadata  
    test_df = pd.read_csv(os.path.join(root_dir, "annotations/test.txt"), sep=" ", header=None, names=columns)
    test_filenames = test_df['filename'].values.tolist()

    # Create test dataset
    test_dataset = ImageSegmentationDataset(test_filenames, image_size, labelled=False)

    # Create data loader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return test_loader