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
    (
        train_all_dl,
        train_lab_dl,
        train_unlab_dl,
        valid_dl,
        test_dl,
    ) = datasets.create_dataloaders(batch_size=32)

    # TODO: add some cli arguments to control what to train, currently hardcoded
    train_supervised = False
    train_semi_supervised = True

    if train_supervised:
        # initialize model
        model = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
        model = model.to(device)
        # train
        train.train_supervised(model, train_lab_dl, valid_dl, epochs=10)

    elif train_semi_supervised:
        # initialize model
        model1 = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
        model2 = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
        model1 = model1.to(device)
        model2 = model2.to(device)
        # train
        train.train_semi_supervised(
            model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs=10
        )

    # save model
    Path("weights").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "weights/supervised.pt")

    # visualize predictions
    images, labels = next(iter(valid_dl))
    images, labels = images[:8].to(device), labels[:8].to(device)
    logits = model(images)["out"][:, 0]
    utils.visualize_predictions(images, logits, "plots/predictions.jpg")
