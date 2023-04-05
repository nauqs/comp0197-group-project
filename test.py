import wandb
from train import eval_ensemble, eval
import os
import models
import torch
import datasets

def test_model_performance(
    model, test_lab_dl
):
    """
    tests a segmentation model.

    The dataloader 'test_lab_dl' should contain only labeled images
    and is only used for the supervised loss.

    """
    # print statistics
    test_loss, test_acc, test_iou, test_dice = eval(model, test_lab_dl)
    wandb.log(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_iou": test_iou,
            "test_dice": test_dice,
        }
    )

    return test_acc, test_iou, test_dice


def get_weights(path_to_folder):
    """
    returns the paths to the weights of the models in the folder
    """
    paths = []
    for file in os.listdir(path_to_folder):
        if file.endswith(".pt"):
            paths.append(os.path.join(path_to_folder, file))
    return paths

if __name__ == "__main__":
    weights = get_weights('weights')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.load_deeplab(
    use_imagenet_weights=False, large_resnet=False)
    _,_,_,_,test_dl= datasets.create_dataloaders(batch_size=6)

    for weight in weights:
        wandb.init(project="comp0197-group-project", entity="comp0197-group-project", group="test_set", name=weight.split('/')[-1], reinit=True)
        model.load_state_dict(torch.load(weight,map_location=device))
        model = model.to(device)
        test_model_performance(model, test_dl)