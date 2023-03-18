import torch
import torchvision
from pathlib import Path
import datasets


def unnormalize_images(images):
    mean, std = torch.tensor(datasets.DS_STATS['classification'])
    images = images * std[None, :, None, None] + mean[None, :, None, None]
    return images.clip(0, 1)


def visualize_predictions(images, logits, filename):

    # normalize images to [0, 1]
    images = unnormalize_images(images)

    # convert logits to class probabilites
    probs = 1 - torch.sigmoid(logits)
    masks = probs[:, None, :, :].repeat(1, 3, 1, 1)

    # create image grid of imput images and predictions
    N, C, H, W = images.shape
    image_grid = torch.cat([images, masks])
    image_grid = torchvision.utils.make_grid(image_grid, nrow=N)

    # ensure output directory exists
    Path(filename).parents[0].mkdir(parents=True, exist_ok=True)

    # save image to disk
    image_grid = (255 * image_grid).to(torch.uint8).cpu()
    torchvision.io.write_jpeg(image_grid, filename)
