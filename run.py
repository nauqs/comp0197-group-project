import itertools
from pathlib import Path
import random
import datasets
import train
import wandb
import torch

aug_methods = [None, "affine", "cutmix", "cutout", "mixup"]


def run(device, epochs, lr, method: str, lamb=None, aug_method=None):
    assert method in ["supervised-full", "supervised", "semi-supervised"]
    assert aug_method in aug_methods
    if method == "supervised-full" or method == "supervised":
        aug_method = lamb = None

    wandb.init(
        project="comp0197-group-project",
        entity="comp0197-group-project",
        config=dict(
            epochs=epochs,
            lr=lr,
            method=method,
            lamb=lamb,
            aug_method=aug_method,
        ),
    )

    labelled_frac = 1.0 if method == "supervised-full" else 0.0625
    (
        train_all_dl,
        train_lab_dl,
        valid_dl,
        test_dl,
    ) = datasets.create_dataloaders(batch_size=6, labelled_frac=labelled_frac)

    if method == 'supervised-full' or method == 'supervised':
        train.train_supervised(device, train_lab_dl, valid_dl, test_dl, epochs, lr=lr)
    else:
        train.train_semi_supervised(
            device,
            train_all_dl,
            train_lab_dl,
            valid_dl,
            test_dl,
            epochs,
            lr=lr,
            lamb=lamb,
            aug_method=aug_method,
        )

    wandb.finish()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 50
    lr = 1e-3
    lamb = 0.25

    while True:
        configs = []
        # add self-supervised configs
        for aug_method in aug_methods:
            configs.append({"lamb": lamb, "aug_method": aug_method, 'method': 'semi-supervised'})
        # add supervised configs
        configs.append({'method': 'supervised-full'})
        configs.append({'method': 'supervised'})

        # shuffle so everyone runs them in a different order
        random.shuffle(configs)

        for c in configs:
            try:
                run(device=device, epochs=epochs, lr=lr, **c)
            except Exception as e:
                wandb.finish(exit_code=1)
                print(e)
            finally:
                try:
                    wandb.finish(exit_code=1)
                except: pass
