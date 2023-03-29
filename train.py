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
def eval_ensemble(model1, model2, dataloader):
    """
    Evaluates model1 and model2 ensemble metrics.
    """
    device = next(model1.parameters()).device
    epoch_loss_ensemble = 0
    epoch_acc_ensemble = 0
    epoch_iou_ensemble = 0
    epoch_dice_ensemble = 0

    # iterate over batches
    n_batches = len(dataloader)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # get a mask of labeled pixels (foreground/background)
        labeled_pixels = labels != 2

        # get model predictions
        logits1 = model1(images)["out"][:, 0]
        logits2 = model2(images)["out"][:, 0]
        probs1 = torch.sigmoid(logits1)
        probs2 = torch.sigmoid(logits2)
        probs_ensemble = (probs1 + probs2) / 2

        # compute loss, accuracy on labeled pixels
        loss1 = F.binary_cross_entropy_with_logits(
            logits1[labeled_pixels], labels[labeled_pixels].float()
        )
        loss2 = F.binary_cross_entropy_with_logits(
            logits2[labeled_pixels], labels[labeled_pixels].float()
        )
        loss_ensemble = (loss1 + loss2)/2
        acc_ensemble = ((probs_ensemble[labeled_pixels] > 0.5) == labels[labeled_pixels]).float().mean()

        # compute iou and dice metrics
        iou_ensemble = iou_eval(probs_ensemble, labels, labeled_pixels)
        dice_ensemble = dice_eval(probs_ensemble, labels, labeled_pixels)

        epoch_loss_ensemble += loss_ensemble.item() / n_batches
        epoch_acc_ensemble += acc_ensemble.item() / n_batches
        epoch_iou_ensemble += iou_ensemble.item() / n_batches
        epoch_dice_ensemble += dice_ensemble.item() / n_batches

    return epoch_loss_ensemble, epoch_acc_ensemble, epoch_iou_ensemble, epoch_dice_ensemble


def train_supervised(model, train_lab_dl, valid_dl, epochs, lr=1e-3):
    """
    Trains a segmentation model.

    The dataloader 'train_lab_dl' should contain only labeled images
    and is only used for the supervised loss.
    """
    device = next(model.parameters()).device
    wandb.config.update({"lr": lr})
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # iterate over epochs
    for epoch in range(epochs):
        t = time()
        train_loss = 0
        train_acc = 0
        train_iou = 0
        train_dice = 0

        # iterate over batches
        n_batches = len(train_lab_dl)
        for images, labels in train_lab_dl:
            images, labels = images.to(device), labels.to(device)

            # get a mask of labeled pixels (foreground/background)
            labeled_pixels = labels != 2

            # zero gradients
            optimizer.zero_grad()

            # get model predictions
            logits = model(images)["out"][:, 0]

            # compute loss, accuracy on labeled pixels
            loss = F.binary_cross_entropy_with_logits(
                logits[labeled_pixels], labels[labeled_pixels].float()
            )
            probs = torch.sigmoid(logits)
            acc = (
                ((probs[labeled_pixels] > 0.5) == labels[labeled_pixels]).float().mean()
            )

            # compute iou and dice metrics
            iou = iou_eval(probs, labels, labeled_pixels)
            dice = dice_eval(probs, labels, labeled_pixels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() / n_batches
            train_acc += acc.item() / n_batches
            train_iou += iou.item() / n_batches
            train_dice += dice.item() / n_batches

        # print statistics
        val_loss, val_acc, val_iou, val_dice = eval(model, valid_dl)
        epoch_time = -t + (t := time())  # time per epoch
        wandb.log(
            {
                "epoch": epoch + 1,
                "time": epoch_time,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_iou": train_iou,
                "train_dice": train_dice,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_iou": val_iou,
                "val_dice": val_dice,
            }
        )
        print(
            f"epoch={epoch+1:2}, time={epoch_time:5.2f}, {train_acc=:4.2%},{train_iou=:4.2%}, {train_dice=:4.2%}, {val_acc=:4.2%}, {val_iou=:4.2%}, {val_dice=:4.2%}"
        )

    return model


def train_semi_supervised(
    model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs, lr=1e-3, lamb=1.5
):
    """
    Trains a segmentation model.

    The dataloader 'train_lab_dl' should contain only labeled images
    and is only used for the supervised loss
    The dataloader 'train_unlab_dl' should contain only unlabeled images
    and is only used for the cps loss.
    lamb is the weight of the cps loss.

    """

    device = next(model1.parameters()).device
    wandb.config.update({"lr": lr, "lamb": lamb})
    # set optimizer to optimise both models
    params = list(model1.parameters()) + list(model2.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # iterable for unlabeled images
    # we will need to sample a few with every labeled batch
    # if there are more labeled than unlabeled, we will need to loop around, hence the cycle
    unlabeled_images_iter = cycle(train_unlab_dl)
    print(
        f"Using {train_unlab_dl.batch_size} unlabeled images per labeled batch of {train_lab_dl.batch_size}."
    )

    # iterate over epochs
    for epoch in range(epochs):
        t = time()
        # loss is combined, accuracy is per model
        train_loss = 0
        train_acc1 = 0
        train_acc2 = 0
        train_acc_ensemble = 0
        train_iou1 = 0
        train_iou2 = 0
        train_iou_ensemble = 0
        train_dice1 = 0
        train_dice2 = 0
        train_dice_ensemble = 0

        # iterate over batches of labeled data
        n_batches = len(train_lab_dl)
        for images, labels in train_lab_dl:
            images, labels = images.to(device), labels.to(device)
            # give next batch if we have more left
            # don't even pick up labels
            unlabeled_images, _ = next(unlabeled_images_iter)
            unlabeled_images = unlabeled_images.to(device)

            # get a mask of labeled pixels (foreground/background)
            labeled_pixels = labels != 2
            # zero gradients
            optimizer.zero_grad()

            # get model predictions for labeled data
            lab_logits1 = model1(images)["out"][:, 0]
            lab_logits2 = model2(images)["out"][:, 0]
            # get model predictions for unlabeled data
            unlab_logits1 = model1(unlabeled_images)["out"][:, 0]
            unlab_logits2 = model2(unlabeled_images)["out"][:, 0]
            # stop gradient for predictions
            lab_preds1 = (lab_logits1 > 0).detach().clone()
            lab_preds2 = (lab_logits2 > 0).detach().clone()
            unlab_preds1 = (unlab_logits1 > 0).detach().clone()
            unlab_preds2 = (unlab_logits2 > 0).detach().clone()

            # compute losses
            loss_sup1 = F.binary_cross_entropy_with_logits(
                lab_logits1[labeled_pixels], labels[labeled_pixels].float()
            )
            loss_sup2 = F.binary_cross_entropy_with_logits(
                lab_logits2[labeled_pixels], labels[labeled_pixels].float()
            )

            loss_cps_lab1 = F.binary_cross_entropy_with_logits(
                lab_logits1, lab_preds2.float()
            )
            loss_cps_lab2 = F.binary_cross_entropy_with_logits(
                lab_logits2, lab_preds1.float()
            )

            loss_cps_unlab1 = F.binary_cross_entropy_with_logits(
                unlab_logits1, unlab_preds2.float()
            )
            loss_cps_unlab2 = F.binary_cross_entropy_with_logits(
                unlab_logits2, unlab_preds1.float()
            )
            # total loss = loss_sup + lamb * loss_cps
            loss = (
                loss_sup1
                + loss_sup2
                + lamb
                * (loss_cps_lab1 + loss_cps_lab2 + loss_cps_unlab1 + loss_cps_unlab2)
            )

            lab_probs1 =  torch.sigmoid(lab_logits1)
            lab_probs2 =  torch.sigmoid(lab_logits2)
            lab_prob_ensemble = (lab_probs1 + lab_probs2) / 2
            # accuracy is only computed on labeled pixels
            acc1 = (
                ((lab_probs1[labeled_pixels] > 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )
            acc2 = (
                ((lab_probs2[labeled_pixels] > 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )

            acc_ensemble = (
                ((lab_prob_ensemble[labeled_pixels]> 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )

            # compute IoU metric for both models
            iou1 = iou_eval(lab_probs1, labels, labeled_pixels)
            iou2 = iou_eval(lab_probs2, labels, labeled_pixels)
            iou_ensemble = iou_eval(lab_prob_ensemble, labels, labeled_pixels)

            # compute dice metric for both models
            dice1 = dice_eval(lab_probs1, labels, labeled_pixels)
            dice2 = dice_eval(lab_probs2, labels, labeled_pixels)
            dice_ensemble = dice_eval(lab_prob_ensemble, labels, labeled_pixels)

            # compute dice

            loss.backward()
            optimizer.step()
            train_loss += loss.item() / n_batches
            train_acc1 += acc1.item() / n_batches
            train_acc2 += acc2.item() / n_batches
            train_acc_ensemble += acc_ensemble.item() / n_batches
            train_iou1 += iou1.item() / n_batches
            train_iou2 += iou2.item() / n_batches
            train_iou_ensemble += iou_ensemble.item() / n_batches
            train_dice1 += dice1.item() / n_batches
            train_dice2 += dice2.item() / n_batches
            train_dice_ensemble += dice_ensemble.item() / n_batches

        # print statistics
        val_loss1, val_acc1, val_iou1, val_dice1 = eval(model1, valid_dl)
        val_loss2, val_acc2, val_iou2, val_dice2 = eval(model2, valid_dl)
        val_loss, val_acc, val_iou, val_dice = eval_ensemble(model1, model2, valid_dl)
        epoch_time = -t + (t := time())  # time per epoch
        wandb.log(
            {
                "epoch": epoch + 1,
                "time": epoch_time,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "train_acc2": train_acc2,
                "train_acc": train_acc_ensemble,
                "train_iou1": train_iou1,
                "train_iou2": train_iou2,
                "train_iou": train_iou_ensemble,
                "train_dice1": train_dice1,
                "train_dice2": train_dice2,
                "train_dice": train_dice_ensemble,
                "val_loss1": val_loss1,
                "val_acc1": val_acc1,
                "val_iou1": val_iou1,
                "val_dice1": val_dice1,
                "val_loss2": val_loss2,
                "val_acc2": val_acc2,
                "val_iou2": val_iou2,
                "val_dice2": val_dice2,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_iou": val_iou,
                "val_dice": val_dice,
            }
        )
        print(
            f"epoch={epoch+1:2}, time={epoch_time:5.2f}, {train_acc1=:4.2%},{train_acc2=:4.2%},{train_acc_ensemble=:4.2%},{train_iou1=:4.2%},{train_iou2=:4.2%}, {train_dice1=:4.2%}, {train_dice2=:4.2%} ,{val_acc1=:4.2%}, {val_acc2=:4.2%}, {val_iou1=:4.2%}, {val_iou2=:4.2%}, {val_dice1=:4.2%}, {val_dice2=:4.2%}, {val_acc=:4.2%}, {val_iou=:4.2%}, {val_dice=:4.2%}"
        )

    return model1, model2


def train_semi_supervised_cutmix(
    model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs, lr=1e-3, lamb=0.5
):
    """
    Trains a segmentation model.
    The dataloader 'train_lab_dl' should contain only labeled images
    and is only used for the supervised loss
    The dataloader 'train_unlab_dl' should contain only unlabeled images
    and is only used for the cps loss.
    lamb is the weight of the cps loss.
    """

    device = next(model1.parameters()).device
    wandb.config.update({"lr": lr, "lamb": lamb})
    # set optimizer to optimise both models
    params = list(model1.parameters()) + list(model2.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # iterable for unlabeled images
    # we will need to sample a few with every labeled batch
    # if there are more labeled than unlabeled, we will need to loop around, hence the cycle
    unlabeled_images_iter = cycle(train_unlab_dl)
    print(
        f"Using {train_unlab_dl.batch_size} unlabeled images per labeled batch of {train_lab_dl.batch_size}."
    )

    # iterate over epochs
    for epoch in range(epochs):
        t = time()
        # loss is combined, accuracy is per model
        train_loss = 0
        train_acc1 = 0
        train_acc2 = 0
        train_acc_ensemble = 0
        train_iou1 = 0
        train_iou2 = 0
        train_iou_ensemble = 0
        train_dice1 = 0
        train_dice2 = 0
        train_dice_ensemble = 0

        # iterate over batches of labeled data
        n_batches = len(train_lab_dl)
        for images, labels in train_lab_dl:
            images, labels = images.to(device), labels.to(device)

            unlabeled_images, _ = next(unlabeled_images_iter)
            unlabeled_images = unlabeled_images.to(device)

            # cutmix the unlabeled images
            (
                mix_unlab_images,
                unlab_images_a,
                unlab_images_b,
                _,
                _,
                unlab_mask,
            ) = utils.cutmix(unlabeled_images)

            # get a mask of labeled pixels (foreground/background)
            labeled_pixels = labels != 2
            # zero gradients
            optimizer.zero_grad()

            # get model predictions for labeled data
            lab_logits1 = model1(images)["out"][:, 0]
            lab_logits2 = model2(images)["out"][:, 0]

            # stop gradient for predictions
            lab_preds1 = (lab_logits1 > 0).detach().clone()
            lab_preds2 = (lab_logits2 > 0).detach().clone()

            # get model predictions for unlabeled data

            # Step 1: get predictions for images_a and images_b separately in both models, and cut-mix them
            unlab_logits1_images_a = model1(unlab_images_a)["out"][:, 0]
            unlab_logits1_images_b = model1(unlab_images_b)["out"][:, 0]
            unlab_logits2_images_a = model2(unlab_images_a)["out"][:, 0]
            unlab_logits2_images_b = model2(unlab_images_b)["out"][:, 0]

            unlab_target1_cutmix = utils.apply_cutmix_mask_to_output(
                unlab_logits1_images_a, unlab_logits1_images_b, unlab_mask
            )
            unlab_target2_cutmix = utils.apply_cutmix_mask_to_output(
                unlab_logits2_images_a, unlab_logits2_images_b, unlab_mask
            )

            # Step 2: get predicted labels for cut-mix images in both models
            unlab_logits1_cutmix = model1(mix_unlab_images)["out"][:, 0]
            unlab_logits2_cutmix = model2(mix_unlab_images)["out"][:, 0]
            unlab_preds1_cutmix = (unlab_logits1_cutmix > 0).detach().clone()
            unlab_preds2_cutmix = (unlab_logits2_cutmix > 0).detach().clone()

            # compute losses
            loss_sup1 = F.binary_cross_entropy_with_logits(
                lab_logits1[labeled_pixels], labels[labeled_pixels].float()
            )
            loss_sup2 = F.binary_cross_entropy_with_logits(
                lab_logits2[labeled_pixels], labels[labeled_pixels].float()
            )

            loss_cps_lab1 = F.binary_cross_entropy_with_logits(
                lab_logits1, lab_preds2.float()
            )
            loss_cps_lab2 = F.binary_cross_entropy_with_logits(
                lab_logits2, lab_preds1.float()
            )

            loss_cps_unlab1 = F.binary_cross_entropy_with_logits(
                unlab_target1_cutmix, unlab_preds2_cutmix.float()
            )
            loss_cps_unlab2 = F.binary_cross_entropy_with_logits(
                unlab_target2_cutmix, unlab_preds1_cutmix.float()
            )
            # total loss = loss_sup + lamb * loss_cps
            loss = (
                loss_sup1
                + loss_sup2
                + lamb
                * (loss_cps_lab1 + loss_cps_lab2 + loss_cps_unlab1 + loss_cps_unlab2)
            )
            lab_probs1 =  torch.sigmoid(lab_logits1)
            lab_probs2 =  torch.sigmoid(lab_logits2)
            lab_prob_ensemble = (lab_probs1 + lab_probs2) / 2
            # accuracy is only computed on labeled pixels
            acc1 = (
                ((lab_probs1[labeled_pixels] > 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )
            acc2 = (
                ((lab_probs2[labeled_pixels] > 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )
            acc_ensemble = (
                ((lab_prob_ensemble[labeled_pixels] > 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )

            # compute IoU metric for both models
            iou1 = iou_eval(lab_logits1, labels, labeled_pixels)
            iou2 = iou_eval(lab_logits2, labels, labeled_pixels)
            iou_ensemble = iou_eval(lab_prob_ensemble, labels, labeled_pixels)

            # compute the dice metric for both models
            dice1 = dice_eval(lab_logits1, labels, labeled_pixels)
            dice2 = dice_eval(lab_logits2, labels, labeled_pixels)
            dice_ensemble = dice_eval(lab_prob_ensemble, labels, labeled_pixels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() / n_batches
            train_acc1 += acc1.item() / n_batches
            train_acc2 += acc2.item() / n_batches
            train_acc_ensemble += acc_ensemble.item() / n_batches
            train_iou1 += iou1.item() / n_batches
            train_iou2 += iou2.item() / n_batches
            train_iou_ensemble += iou_ensemble.item() / n_batches
            train_dice1 += dice1.item() / n_batches
            train_dice2 += dice2.item() / n_batches
            train_dice_ensemble += dice_ensemble.item() / n_batches

        # print statistics
        val_loss1, val_acc1, val_iou1, val_dice1 = eval(model1, valid_dl)
        val_loss2, val_acc2, val_iou2, val_dice2 = eval(model2, valid_dl)
        val_loss, val_acc, val_iou, val_dice = eval_ensemble(model1, model2, valid_dl)
        epoch_time = -t + (t := time())  # time per epoch
        wandb.log(
            {
                "epoch": epoch + 1,
                "time": epoch_time,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "train_acc2": train_acc2,
                "train_acc": train_acc_ensemble,
                "train_iou1": train_iou1,
                "train_iou2": train_iou2,
                "train_iou": train_iou_ensemble,
                "train_dice1": train_dice1,
                "train_dice2": train_dice2,
                "train_dice": train_dice_ensemble,
                "val_loss1": val_loss1,
                "val_acc1": val_acc1,
                "val_iou1": val_iou1,
                "val_dice1": val_dice1,
                "val_loss2": val_loss2,
                "val_acc2": val_acc2,
                "val_iou2": val_iou2,
                "val_dice2": val_dice2,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_iou": val_iou,
                "val_dice": val_dice,
            }
        )
        print(
            f"epoch={epoch+1:2}, time={epoch_time:5.2f}, {train_acc1=:4.2%},{train_acc2=:4.2%},{train_acc_ensemble=:4.2%},{train_iou1=:4.2%},{train_iou2=:4.2%}, {train_dice1=:4.2%}, {train_dice2=:4.2%} ,{val_acc1=:4.2%}, {val_acc2=:4.2%}, {val_iou1=:4.2%}, {val_iou2=:4.2%}, {val_dice1=:4.2%}, {val_dice2=:4.2%}, {val_acc=:4.2%}, {val_iou=:4.2%}, {val_dice=:4.2%}"
        )

    return model1, model2
