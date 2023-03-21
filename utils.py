import torch
import torchvision
from pathlib import Path
import datasets
import numpy as np


def unnormalize_images(images):
    mean, std = torch.tensor(datasets.DS_STATS["classification"], device=images.device)
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


def cutmix(input, target, alpha=1.0):
    """generate the CutMix versions of the input and target data
    Paper: https://arxiv.org/pdf/1905.04899.pdf

    Args:
        input: the input data
        target: the target data
        alpha: the alpha value for the beta distribution

    Returns:
        input: the cutmix input data
        target_a: the target data A
        target_b: the target data B
        lam: the lambda value
    """

    # initialise lambda
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()

    # find the batch size from input data
    batch_size = len(target)

    # generate a random list of indices with size of the batch size
    rand_idx = torch.randperm(batch_size)

    # assign target_a and target_b
    target_a, target_b = target, target[rand_idx]

    # get the width and height of the input image
    W, H = input.shape[2], input.shape[3]

    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    r_w = torch.sqrt(1.0 - lam)
    r_h = torch.sqrt(1.0 - lam)
    cut_w = (W * r_w).int()
    cut_h = (H * r_h).int()

    x1 = torch.clamp(cx - cut_w // 2, 0, W)
    y1 = torch.clamp(cy - cut_h // 2, 0, H)
    x2 = torch.clamp(cx + cut_w // 2, 0, W)
    y2 = torch.clamp(cy + cut_w // 2, 0, H)

    input[:, :, x1:x2, y1:y2] = input[rand_idx][:, :, x1:x2, y1:y2]

    # adjust lambda to the exact area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    return input, target_a, target_b, lam


def cutmix_criterion(criterion, output, target_a, target_b, lam):
    """compute the cutmix loss

    Args:
        criterion: the loss function to use
        output: the model output
        target_a: the target data A (output of the cutmix function)
        target_b: the target data B (output of the cutmix function)
        lam: the lambda value (output of the cutmix function)

    Returns:
        loss: the cutmix loss
    """

    loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

    return loss
