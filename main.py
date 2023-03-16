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

# Plot a 5x5 grid of images and masks
fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(images[i * 5 + j].permute(1, 2, 0))
        ax[i, j].imshow(masks[i * 5 + j].permute(1, 2, 0), alpha=0.5)
        ax[i, j].axis('off')
plt.show()