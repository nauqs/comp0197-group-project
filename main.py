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
    ) = datasets.create_dataloaders(batch_size=6)

    # TODO: add some cli arguments to control what to train, currently hardcoded
    train_supervised = False
    train_semi_supervised = False
    train_semi_supervised_cutmix = True

    if train_supervised:
        # initialize model
        print("Training supervised")
        print("Labeled data: ", len(train_lab_dl.dataset))
        model = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
        model = model.to(device)
        # train
        model = train.train_supervised(model, train_lab_dl, valid_dl, epochs=10)

        # save model
        Path("weights").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "weights/supervised.pt")
        # visualize predictions
        images, labels = next(iter(valid_dl))
        images, labels = images[:8].to(device), labels[:8].to(device)
        logits = model(images)["out"][:, 0]
        utils.visualize_predictions(images, logits, "plots/predictions.jpg")

    elif train_semi_supervised:
        # initialize model
        print("Training semi-supervised")
        print("Labeled data: ", len(train_lab_dl.dataset))
        model1 = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
        model2 = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
        model1 = model1.to(device)
        model2 = model2.to(device)
        # train
        model1, model2 = train.train_semi_supervised(
            model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs=10
        )

        # save model
        Path("weights").mkdir(parents=True, exist_ok=True)
        torch.save(model1.state_dict(), "weights/semi-supervised1.pt")
        torch.save(model2.state_dict(), "weights/semi-supervised2.pt")

        # visualize predictions
        images, labels = next(iter(valid_dl))
        images, labels = images[:8].to(device), labels[:8].to(device)
        logits1 = model1(images)["out"][:, 0]
        utils.visualize_predictions(images, logits1, "plots/semisup-predictions1.jpg")
        logits2 = model2(images)["out"][:, 0]
        utils.visualize_predictions(images, logits2, "plots/semisup-predictions2.jpg")

    elif train_semi_supervised_cutmix:
        # initialize model
        print("Training semi-supervised cutmix")
        print("Labeled data: ", len(train_lab_dl.dataset))
        model1 = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
        model2 = models.load_deeplab(use_imagenet_weights=True, large_resnet=False)
        model1 = model1.to(device)
        model2 = model2.to(device)
        # train
        model1, model2 = train.train_semi_supervised_cutmix(
            model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs=10
        )

        # save model
        Path("weights").mkdir(parents=True, exist_ok=True)
        torch.save(model1.state_dict(), "weights/semi-supervised-cutmix1.pt")
        torch.save(model2.state_dict(), "weights/semi-supervised-cutmix2.pt")

        # visualize predictions
        images, labels = next(iter(valid_dl))
        images, labels = images[:8].to(device), labels[:8].to(device)
        logits1 = model1(images)["out"][:, 0]
        utils.visualize_predictions(images, logits1, "plots/cutmix_predictions1.jpg")
        logits2 = model2(images)["out"][:, 0]
        utils.visualize_predictions(images, logits2, "plots/cutmix_predictions2.jpg")
