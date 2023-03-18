import time
import torch
import datasets
import models
import train
import utils
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # create dataloaders
    train_all_dl, train_lab_dl, valid_dl = datasets.create_dataloaders(batch_size=32)

    # initialize model
    model = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
    model = model.to(device)

    # train
    train.train(model, train_all_dl, train_lab_dl, valid_dl, epochs=10)

    # save model
    Path("weights").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "weights/supervised.pt")

    # visualize predictions
    images, labels = next(iter(valid_dl))
    images, labels = images[:8].to(device), labels[:8].to(device)
    logits = model(images)["out"][:, 0]
    utils.visualize_predictions(images, logits, "plots/predictions.jpg")
