import torch
import datasets
import models
import train
import utils
from pathlib import Path
import argparse
import wandb


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('run_idx', type=int, default=0, help='trainnig run index')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # training hyperparameters
    epochs = 50

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

    # create output directory for weights
    Path("weights").mkdir(parents=True, exist_ok=True)

    # baseline: train will all labels
    print("Training with all labels...")
    model_config = {'training': 'fully-labeled'}
    model_name = utils.model_config_to_name(model_config)
    model = models.load_deeplab().to(device)
    with wandb.init(config=model_config, name=f'{model_name}_{args.run_idx}', group=model_name, project='comp0197-group-project') as run:
        model = train.train_supervised(model, train_all_dl, valid_dl, epochs)
    torch.save(model.state_dict(), f"weights/{model_name}_{args.run_idx}.pt")

    # train supervised
    print("Training supervised...")
    model_config = {'training': 'supervised'}
    model_name = utils.model_config_to_name(model_config)
    model = models.load_deeplab().to(device)
    with wandb.init(config=model_config, name=f'{model_name}_{args.run_idx}', group=model_name, project='comp0197-group-project') as run:
        model = train.train_supervised(model, train_lab_dl, valid_dl, epochs)
    torch.save(model.state_dict(), f"weights/{model_name}_{args.run_idx}.pt")

    # test multiple lambda values
    for lamb in [0.5, 1, 2]:

        # train semi-supervised
        print("Training semi-supervised...")
        model_config = {'training': 'semi-supervised', 'lamb': lamb}
        model_name = utils.model_config_to_name(model_config)
        model1 = models.load_deeplab().to(device)
        model2 = models.load_deeplab().to(device)
        with wandb.init(config=model_config, name=f'{model_name}_{args.run_idx}', group=model_name, project='comp0197-group-project') as run:
            model1, model2 = train.train_semi_supervised(
                model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs, lamb=lamb,
            )
        torch.save(model1.state_dict(), f"weights/{model_name}_{args.run_idx}.pt")

        # train cutmix
        print("Training semi-supervised cutmix...")
        model_config = {'training': 'cutmix', 'lamb': lamb}
        model_name = utils.model_config_to_name(model_config)
        model1 = models.load_deeplab().to(device)
        model2 = models.load_deeplab().to(device)
        with wandb.init(config=model_config, name=f'{model_name}_{args.run_idx}', group=model_name, project='comp0197-group-project') as run:
            model1, model2 = train.train_semi_supervised_cutmix(
                model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs, affine_transform= False, lamb=lamb,
            )
        torch.save(model1.state_dict(), f"weights/{model_name}_{args.run_idx}.pt")

        # train cutout
        print("Training semi-supervised cutout...")
        model_config = {'training': 'cutout', 'lamb': lamb}
        model_name = utils.model_config_to_name(model_config)
        model1 = models.load_deeplab().to(device)
        model2 = models.load_deeplab().to(device)
        with wandb.init(config=model_config, name=f'{model_name}_{args.run_idx}', group=model_name, project='comp0197-group-project') as run:
            model1, model2 = train.train_semi_supervised_cutout(
                model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs, affine_transform = False, lamb=lamb,
            )
        torch.save(model1.state_dict(), f"weights/{model_name}_{args.run_idx}.pt")
