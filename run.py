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

    epochs = 50
    lamb = 0.5

    for lr in [1e-2, 1e-3, 1e-4]:
        # run supervised only
        run(device=device, epochs=epochs, lr=lr, supervised_only=True)

        # run augmented semi-supervised
        for aug_method in [None, 'affine', 'cutmix', 'cutout', 'mixup']:
            run(device=device, epochs=epochs, lr=lr, supervised_only=False, lamb=lamb, aug_method=aug_method)
