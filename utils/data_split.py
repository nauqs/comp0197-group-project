import pandas as pd
import matplotlib.pyplot as plt
import torch
import os

columns = ["filename", "class_id", "species_id", "breed_id"]
data_dir = "../data/"

train_val_df = pd.read_csv(os.path.join(data_dir, "annotations/trainval.txt"), sep=" ", header=None, names=columns)
all_class = train_val_df['class_id'].unique()

# for each class, split the data into .8 train and .2 validation
torch.manual_seed(42)
train_filenames, valid_filenames = [], []
for c in all_class:
    class_df = train_val_df[train_val_df['class_id'] == c]
    filenames = class_df['filename'].values.tolist()
    shuffled_indices = torch.randperm(len(filenames))
    filenames = [filenames[i] for i in shuffled_indices]
    train_filenames += filenames[:int(.8 * len(filenames))]
    valid_filenames += filenames[int(.8 * len(filenames)):]

# Add a column to the dataframe to indicate whether the image is in the train or validation set
train_val_df['split'] = train_val_df['filename'].apply(lambda x: "train" if x in train_filenames else "valid")

# save the dataframe
train_val_df.to_csv(os.path.join(data_dir, "annotations/trainval_split.txt"), sep=" ", header=False, index=False)