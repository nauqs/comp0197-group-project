from pathlib import Path
import datasets
import train
import wandb
import torch

def run(device, epochs, lr, supervised_only, lamb=None, aug_method=None):
    wandb.init(project="comp0197-group-project", entity="comp0197-group-project")

    (
        train_all_dl,
        train_lab_dl,
        valid_dl,
        test_dl,
    ) = datasets.create_dataloaders(batch_size=6)

    if supervised_only:
        train.train_supervised(device, train_lab_dl, valid_dl, test_dl, epochs, lr=lr)
    else:
        train.train_semi_supervised(device, train_all_dl, train_lab_dl, valid_dl, test_dl, epochs, lr=lr, lamb=lamb, aug_method=aug_method)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fixed hyperparameters
    lr = 1e-3
    epochs = 50

    # run supervised only
    run(device=device, epochs=epochs, lr=lr, supervised_only=True)

    # test different lambda values for each semi-supervised model
    for lamb in [0.1, 0.25, 0.5]:
        for aug_method in [None, 'affine', 'cutmix', 'cutout', 'mixup']:
            run(device=device, epochs=epochs, lr=lr, supervised_only=False, lamb=lamb, aug_method=aug_method)
