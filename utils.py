import torch
import torchvision
from pathlib import Path
import datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import affine


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

def affine_transformation(inputs, target=[]):
    """generate the affine transformation version of the input and target data
    Args
        input: the input data
        target: the target data
        alpha: the alpha value for the beta distribution
    Returns:
        transformed_inputs
        transformed_outputs
    """
    # check the device of the inputs
    device = inputs.device
    

    # apply affine transformation to the input data
    angle = torch.randint(-10, 10, size=(1,)).item()
    translate = (torch.randint(-10, 10, size=(1,)).item(), torch.randint(-10, 10, size=(1,)).item())
    shear = torch.randint(-10, 10, size=(1,)).item()
    scale = torch.FloatTensor([torch.FloatTensor(1, ).uniform_(1, 1.2).item()])
    transformed_inputs = affine(inputs, angle, translate, scale, shear)
    transformed_targets = affine(target, angle, translate, scale, shear, fill=2)

    transformed_inputs = transformed_inputs.clone().to(device)
    transformed_targets = transformed_targets.clone().to(device)

    return transformed_inputs, transformed_targets    

def image_augmentation(method, inputs, target=[], alpha=0.3):
    """generate the image augmentation version of the inputs and targets
    Args
        method: the method to use for image augmentation (cutmix, mixup, cutout)
        input: the input image(s)
        target: the target data (optional)
    Returns:
        aug_input: the augmented input data
        input_a: the first input data
        input_b: the second input data (for mixup and cutmix)
        target_a: the target data for input_a
        target_b: the target data for input_b (for mixup and cutmix)
        augment: the mask (or lambda value for mixup) of the augmentation
    """
    
    # ensure the method specified is valid
    assert method in ["cutmix", "mixup", "cutout"]

    # check the device of the inputs
    device = inputs.device

    # initialise lambda for the image augmentation
    lam = torch.distributions.beta.Beta(alpha, alpha).sample()

    # initialise input_a and input_b
    input_a, input_b = inputs, None
    target_a, target_b = None, None
    if len(target) > 0:
        target_a = target

    # randomly select inputs_b from the input data (for mixup and cutmix)
    if method in ["mixup", "cutmix"]:
        batch_size = len(inputs)
        rand_idx = torch.randperm(batch_size)
        input_b = inputs[rand_idx]
        if len(target) > 0:
            target_b = target[rand_idx]
    else:
        if len(target) > 0:
            target_b = 2*torch.ones_like(target)

    if method in ["cutmix", "cutout"]:
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

        # restrict the cutout length, width to be at most 50% of the image
        if method == "cutout":
            width_ratio = (x2-x1) / W
            height_ratio = (y2-y1) / H
            if width_ratio > 0.5:
                width_shrink_ratio = torch.tensor(0.5) / width_ratio
                x2 = torch.clamp(x1+(x2-x1)*width_shrink_ratio, 0, W).to(device, dtype=torch.int)
            if height_ratio > 0.5:
                height_shrink_ratio = torch.tensor(0.5) / height_ratio
                y2 = torch.clamp(y1+(y2-y1)*height_shrink_ratio, 0, H).to(device, dtype=torch.int)
            
        # apply the augmentation to the input data and save to aug_input
        aug_input = inputs.clone().to(device)

        if method == "cutmix":
            aug_input[:, :, x1:x2, y1:y2] = input_b[:, :, x1:x2, y1:y2]
        else:
            aug_input[:, :, x1:x2, y1:y2] = 0.

        # save the augment
        augment = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}

    # implement mixup
    else:
        aug_input = (lam * input_a + (1 - lam) * input_b).to(device)
        augment = lam

    return aug_input, input_a, input_b, target_a, target_b, augment


def apply_image_aug_to_output(method, output_a, output_b, augment):
    """apply the image augmentation to output
    Args:
        method: the method to apply (cutmix, cutout, mixup)
        output_a: the output data A
        output_b: the output data B
        augment: the augmentation to apply
    Returns:
        aug_output: the data from the augmentation
    """

    assert method in ["cutmix", "cutout", "mixup"]
    # check the device of the output
    device = output_a.device


    # initialise the output
    aug_output = output_a.clone()

    if method in ["cutmix", "cutout"]:
        
        x1, x2, y1, y2 = augment["x1"], augment["x2"], augment["y1"], augment["y2"]

        if method == "cutmix":
            # apply the mask to the output
            aug_output[:, x1:x2, y1:y2] = output_b[:, x1:x2, y1:y2]
        else:
            # apply the mask to the output
            aug_output[:, x1:x2, y1:y2] = 2 

    else:
        aug_output = (augment * output_a + (1-augment) * output_b).to(device)
    
    return aug_output