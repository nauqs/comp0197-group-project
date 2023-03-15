import matplotlib.pyplot as plt
from utils.datasets import get_train_valid_data, get_test_data
from utils.load_config import load_config

# Load configuration
config = load_config("config.ini")
root_dir = config['DataSets']['Dir']
batch_size = int(config['DataSets']['BatchSize'])
image_size = int(config['DataSets']['ImageSize'])
labelled_fraction = float(config['DataSets']['LabelledFraction'])
valid_fraction = float(config['DataSets']['ValidFraction'])

# Dummy tests
labelled_train_loader, unlabelled_train_loader, valid_loader = get_train_valid_data(root_dir, batch_size, image_size, labelled_fraction, valid_fraction)
test_loader = get_test_data(root_dir, image_size, batch_size)

images, masks = next(iter(labelled_train_loader))

# Plot 10 pairs of images and masks in subplots
fig, ax = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    ax[0, i].imshow(images[i].permute(1, 2, 0))
    ax[0, i].axis("off")
    ax[1, i].imshow(masks[i].squeeze(), cmap="gray")
    ax[1, i].axis("off")
plt.show()