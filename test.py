import wandb
import torch
import torch.nn.functional as F
from time import time
import utils
from itertools import cycle


@torch.no_grad()
def iou_eval(probs, labels, labeled_pixels):
    smooth = 1e-10

    # foreground
    predicted_labels_1 = (probs[labeled_pixels] > 0.5) == 0
    true_labels_1 = labels[labeled_pixels] == 0

    # # background
    predicted_labels_2 = (probs[labeled_pixels] < 0.5) == 0
    true_labels_2 = labels[labeled_pixels] == 1

    iou_1 = ((predicted_labels_1 & true_labels_1).sum() + smooth) / (
        (predicted_labels_1 | true_labels_1).sum() + smooth
    )
    iou_2 = ((predicted_labels_2 & true_labels_2).sum() + smooth) / (
        (predicted_labels_2 | true_labels_2).sum() + smooth
    )

    # mean IoU
    iou = (iou_1 + iou_2) / 2

    return iou


@torch.no_grad()
def dice_eval(probs, labels, labeled_pixels):
    # foreground
    predicted_labels_1 = (probs[labeled_pixels] > 0.5) == 0
    true_labels_1 = labels[labeled_pixels] == 0

    # background
    predicted_labels_2 = (probs[labeled_pixels] < 0.5) == 0
    true_labels_2 = labels[labeled_pixels] == 1

    dice_1 = (
        2
        * (predicted_labels_1 & true_labels_1).sum()
        / (
            (predicted_labels_1 | true_labels_1).sum()
            + (predicted_labels_1 & true_labels_1).sum()
        )
    )
    dice_2 = (
        2
        * (predicted_labels_2 & true_labels_2).sum()
        / (
            (predicted_labels_2 | true_labels_2).sum()
            + (predicted_labels_2 & true_labels_2).sum()
        )
    )

    # mean IoU
    dice = (dice_1 + dice_2) / 2

    return dice


@torch.no_grad()
def eval(model, dataloader):
    """
    Evaluates model loss and accuracy over one epoch.
    """
    device = next(model.parameters()).device
    epoch_loss = 0
    epoch_acc = 0
    epoch_iou = 0
    epoch_dice = 0

    # iterate over batches
    n_batches = len(dataloader)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # get a mask of labeled pixels (foreground/background)
        labeled_pixels = labels != 2

        # get model predictions
        logits = model(images)["out"][:, 0]

        # compute loss, accuracy on labeled pixels
        loss = F.binary_cross_entropy_with_logits(
            logits[labeled_pixels], labels[labeled_pixels].float()
        )
        acc = ((logits[labeled_pixels] > 0) == labels[labeled_pixels]).float().mean()

        # compute iou and dice metrics
        iou = iou_eval(logits, labels, labeled_pixels)
        dice = dice_eval(logits, labels, labeled_pixels)

        epoch_loss += loss.item() / n_batches
        epoch_acc += acc.item() / n_batches
        epoch_iou += iou.item() / n_batches
        epoch_dice += dice.item() / n_batches

    return epoch_loss, epoch_acc, epoch_iou, epoch_dice

@torch.no_grad()
def eval_ensemble(model, dataloader):
    """
    Evaluates model1 and model2 ensemble metrics.
    """
    device = next(model.parameters()).device
    epoch_loss = 0
    epoch_acc = 0
    epoch_iou = 0
    epoch_dice = 0

    # iterate over batches
    n_batches = len(dataloader)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # get a mask of labeled pixels (foreground/background)
        labeled_pixels = labels != 2

        # get model predictions
        logits = model(images)["out"][:, 0]
        probs = torch.sigmoid(logits)

        # compute loss, accuracy on labeled pixels
        loss = F.binary_cross_entropy_with_logits(
            logits[labeled_pixels], labels[labeled_pixels].float()
        )
        acc = ((probs[labeled_pixels] > 0.5) == labels[labeled_pixels]).float().mean()

        # compute iou and dice metrics
        iou = iou_eval(probs, labels, labeled_pixels)
        dice = dice_eval(probs, labels, labeled_pixels)

        epoch_loss += loss.item() / n_batches
        epoch_acc += acc.item() / n_batches
        epoch_iou += iou.item() / n_batches
        epoch_dice += dice.item() / n_batches

    return epoch_loss, epoch_acc, epoch_iou, epoch_dice




def test_model_performance(
    model1, model2, test_lab_dl,lamb
):
    """
    tests a segmentation model.

    The dataloader 'test_lab_dl' should contain only labeled images
    and is only used for the supervised loss.

    """

    device = next(model1.parameters()).device
    wandb.config.update({"lr": lr})
    # set optimizer to optimise both models
    params = list(model1.parameters()) + list(model2.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # iterable for unlabeled images
    # we will need to sample a few with every labeled batch
    # if there are more labeled than unlabeled, we will need to loop around, hence the cycle
    # unlabeled_images_iter = cycle(test_unlab_dl)
    # print(
    #     f"Using {test_unlab_dl.batch_size} unlabeled images per labeled batch of {test_lab_dl.batch_size}."
    # )

    # iterate over epochs
    with torch.no_grad():

        # iterate over batches of labeled data
        n_batches = len(test_lab_dl)
        for images, labels in test_lab_dl:
            images, labels = images.to(device), labels.to(device)
            ## give next batch if we have more left
            ## don't even pick up labels
            #unlabeled_images, _ = next(unlabeled_images_iter)
            #unlabeled_images = unlabeled_images.to(device)

            # get a mask of labeled pixels (foreground/background)
            labeled_pixels = labels != 2
            ## zero gradients
            #optimizer.zero_grad()

            ## get model predictions for labeled data
            lab_logits1 = model1(images)["out"][:, 0]
            lab_preds1 = (lab_logits1 > 0).detach().clone()

        # print statistics
        test_loss1, test_acc1, test_iou1, test_dice1 = eval(model1, test_lab_dl)
        test_loss2, test_acc2, test_iou2, test_dice2 = eval(model2, test_lab_dl)
        test_loss, test_acc, test_iou, test_dice = eval_ensemble(model1, model2, test_lab_dl)
        wandb.log(
            {
                "test_loss1": test_loss1,
                "test_acc1": test_acc1,
                "test_iou1": test_iou1,
                "test_dice1": test_dice1,
                "test_loss2": test_loss2,
                "test_acc2": test_acc2,
                "test_iou2": test_iou2,
                "test_dice2": test_dice2,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_iou": test_iou,
                "test_dice": test_dice,
            }
        )

    return  test_acc1, test_acc2, test_iou1, test_iou2, test_dice1, test_dice2, test_acc, test_iou, test_dice

