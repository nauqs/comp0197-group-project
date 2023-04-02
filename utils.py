import torch
import torchvision
from pathlib import Path
import datasets
import numpy as np


def model_config_to_name(config):
    """
    Given a model configuration dictionary
    return a unique name for the model.
    """
    return '_'.join(f'{k}={v}' for k, v, in config.items())


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

def cutout(inputs, target=[], alpha=1.0):
    """generate the Cutout version of the input and target data
    Args
        input: the input data
        target: the target data
        alpha: the alpha value for the beta distribution
    Returns:
        cutout_input: the cutout input data
        target: the original target data
        mask: the mask used to cutout the input data
    """
    device = inputs.device
    # initialise lambda
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()

    # get the width and height of the input image
    W, H = inputs.shape[2], inputs.shape[3]

    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    r_w = torch.sqrt(1.0 - lam)
    r_h = torch.sqrt(1.0 - lam)
    cut_w = (W * r_w).int()
    cut_h = (H * r_h).int()

    x1 = torch.clamp(cx - cut_w // 2, 0, W).to(device)
    y1 = torch.clamp(cy - cut_h // 2, 0, H).to(device)
    x2 = torch.clamp(cx + cut_w // 2, 0, W).to(device)
    y2 = torch.clamp(cy + cut_w // 2, 0, H).to(device)

    # apply the mask to the input data and save to cutout_input
    cutout_input = inputs.clone().to(device)
    cutout_input[:, :, x1:x2, y1:y2] = 0.

    # adjust lambda to the exact area ratio
    # lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    # save the mask
    mask = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}

    # assign target_a and target_b
    if len(target) > 0:
        return cutout_input, inputs, target, mask

    else:
        return cutout_input, inputs, None, mask
    
def mixup(inputs, target=[], alpha=1.0):
    """generate the Mixup versionf of the input and target data
    Args
        input: the input data
        target: the target data
        alpha: the alpha value for the beta distribution
    Returns:
        mix_input: the mixup input data
        target: the original target data
        mask: the mask used to cutout the input data
    """
    device = inputs.device
    # initialise lambda
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()

    # find the batch size from input data
    batch_size = len(inputs)

    # generate a random list of indices with size of the batch size
    rand_idx = torch.randperm(batch_size)

    # mix up the input data and save to mix_input
    input_a = inputs
    input_b = inputs[rand_idx]
    mix_input = (lam * input_a + (1 - lam) * input_b).to(device)

    # assign target_a and target_b
    if len(target) > 0:
        return mix_input, inputs, target

    else:
        return mix_input, inputs, None

def cutmix(inputs, target=[], alpha=1.0):
    """generate the CutMix versions of the input and target data
    Paper: https://arxiv.org/pdf/1905.04899.pdf
    Args:
        input: the input data
        target: the target data
        alpha: the alpha value for the beta distribution
    Returns:
        mix_input: the cutmix input data
        target_a: the target data A
        target_b: the target data B
        mask: the mask used to cutmix the input data
    """
    device = inputs.device
    # initialise lambda
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()

    # find the batch size from input data
    batch_size = len(inputs)

    # generate a random list of indices with size of the batch size
    rand_idx = torch.randperm(batch_size)

    # get the width and height of the input image
    W, H = inputs.shape[2], inputs.shape[3]

    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    r_w = torch.sqrt(1.0 - lam)
    r_h = torch.sqrt(1.0 - lam)
    cut_w = (W * r_w).int()
    cut_h = (H * r_h).int()

    x1 = torch.clamp(cx - cut_w // 2, 0, W).to(device)
    y1 = torch.clamp(cy - cut_h // 2, 0, H).to(device)
    x2 = torch.clamp(cx + cut_w // 2, 0, W).to(device)
    y2 = torch.clamp(cy + cut_w // 2, 0, H).to(device)

    # apply the mask to the input data and save to mix_input
    input_a = inputs
    input_b = inputs[rand_idx]
    mix_input = inputs.clone().to(device)
    mix_input[:, :, x1:x2, y1:y2] = input_b[:, :, x1:x2, y1:y2]

    # adjust lambda to the exact area ratio
    # lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    # save the mask
    mask = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}

    # assign target_a and target_b
    if len(target) > 0:
        target_a, target_b = target, target[rand_idx]
        return mix_input, input_a, input_b, target_a, target_b, mask

    else:
        return mix_input, input_a, input_b, None, None, mask


def apply_cutmix_mask_to_output(output_a, output_b, mask):
    """apply the cutmix mask to the output data
    Args:
        output_a: the output A
        output_b: the output B
        mask: the mask to apply
    Returns:
        mix_output: the cutmix input data
    """

    x1, x2, y1, y2 = mask["x1"], mask["x2"], mask["y1"], mask["y2"]

    # apply the mask to the output and save to mix_output
    mix_output = output_a.clone()
    mix_output[:, x1:x2, y1:y2] = output_b[:, x1:x2, y1:y2]

    return mix_output
