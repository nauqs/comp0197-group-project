import torch
import torchvision
import torch.nn.functional as F


def visualize_predictions(images, logits, filename):
    
    # normalize images to [0, 1]
    images = images*0.5 + 0.5
    
    # convert logits to class probabilites
    probs = F.sigmoid(logits)
    masks = probs[:, None, :, :].repeat(1, 3, 1, 1)

    # create image grid of imput images and predictions
    N, C, H, W = images.shape
    image_grid = torch.cat([images, masks])
    image_grid = torchvision.utils.make_grid(image_grid, nrow=N)

    # save the image grid to disk
    image_grid = (255*image_grid).to(torch.uint8)
    torchvision.io.write_jpeg(image_grid, filename)
    