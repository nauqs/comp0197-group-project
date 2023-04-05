import wandb
import torch
import torch.nn.functional as F
from time import time
import utils
import models


@torch.no_grad()
def acc_eval(probs, labels, labeled_pixels):
    acc = (
        ((probs[labeled_pixels] > 0.5) == labels[labeled_pixels])
        .float()
        .mean()
    )
    return acc


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

        # compute loss
        loss = F.binary_cross_entropy_with_logits(
            logits[labeled_pixels], labels[labeled_pixels].float()
        )

        # compute metrics
        probs = torch.sigmoid(logits)
        acc = acc_eval(probs, labels, labeled_pixels)
        iou = iou_eval(probs, labels, labeled_pixels)
        dice = dice_eval(probs, labels, labeled_pixels)

        epoch_loss += loss.item() / n_batches
        epoch_acc += acc.item() / n_batches
        epoch_iou += iou.item() / n_batches
        epoch_dice += dice.item() / n_batches

    return epoch_loss, epoch_acc, epoch_iou, epoch_dice


def train_supervised(device, train_lab_dl, valid_dl, test_dl, epochs, lr=1e-3, aug_method=None):
    """
    Trains a segmentation model.

    The dataloader 'train_lab_dl' should contain only labeled images
    and is only used for the supervised loss.
    """
    assert aug_method is None or aug_method == 'affine'
    model = models.load_deeplab().to(device)
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

            # apply data augmentation
            if aug_method == 'affine':
                images, labels = utils.affine_transformation(images, labels)

            # get a mask of labeled pixels (foreground/background)
            labeled_pixels = labels != 2

            # zero gradients
            optimizer.zero_grad()

            # get model predictions
            logits = model(images)["out"][:, 0]

            # compute loss
            loss = F.binary_cross_entropy_with_logits(
                logits[labeled_pixels], labels[labeled_pixels].float()
            )
            loss.backward()
            optimizer.step()
            
            # compute metrics
            probs = torch.sigmoid(logits)
            acc = acc_eval(probs, labels, labeled_pixels)
            iou = iou_eval(probs, labels, labeled_pixels)
            dice = dice_eval(probs, labels, labeled_pixels)
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
            f"epoch={epoch+1:2}, time={epoch_time:5.2f}, {train_acc=:4.2%}, {val_acc=:4.2%}"
        )

    # test metrics
    test_loss, test_acc, test_iou, test_dice = eval(model, test_dl)
    wandb.log(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_iou": test_iou,
            "test_dice": test_dice,
        }
    )

    return model


def train_semi_supervised(
    device, train_all_dl, train_lab_dl, valid_dl, test_dl, epochs, lr=1e-3, lamb=0.5, aug_method=None
    ):
    """
    Trains a segmentation model.

    The dataloader 'train_lab_dl' should contain only labeled images
    and is only used for the supervised loss
    The dataloader 'train_unlab_dl' should contain only unlabeled images
    and is only used for the cps loss.
    lamb is the weight of the cps loss.

    """

    # check inputs
    assert aug_method is None or aug_method in ['affine', 'cutmix', 'cutout', 'mixup']
    model1 = models.load_deeplab().to(device)
    model2 = models.load_deeplab().to(device)
    print(
        f"Using {train_all_dl.batch_size} unlabeled images per labeled batch of {train_all_dl.batch_size}."
    )
    
    # set optimizer to optimise both models
    params = list(model1.parameters()) + list(model2.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    wandb.config.update({"lr": lr})
    train_all_dl = iter(train_all_dl)

    # iterate over epochs
    for epoch in range(epochs):
        t = time()
        # loss is combined, accuracy is per model
        train_loss = 0
        train_acc = 0
        train_iou = 0
        train_dice = 0

        # iterate over batches of labeled data
        n_batches = len(train_lab_dl)
        for labeled_images, labels in train_lab_dl:
            labeled_images, labels = labeled_images.to(device), labels.to(device)

            # get batch of unlabeled images (without loading their labels)
            unlabeled_images, _ = next(train_all_dl)
            unlabeled_images = unlabeled_images.to(device)

            # apply data augmentation
            if aug_method == 'affine':
                labeled_images, labels = utils.affine_transformation(labeled_images, labels)
                unlabeled_images, _ = utils.affine_transformation(unlabeled_images)

            # zero gradients
            optimizer.zero_grad()

            ## supervised part ##

            # get a mask of labeled pixels (foreground/background)
            labeled_pixels = labels != 2

            # get model predictions for labeled data
            lab_logits1 = model1(labeled_images)["out"][:, 0]
            lab_logits2 = model2(labeled_images)["out"][:, 0]

            # compute supervised loss
            loss_sup = F.binary_cross_entropy_with_logits(
                lab_logits1[labeled_pixels], labels[labeled_pixels].float()
            ) + F.binary_cross_entropy_with_logits(
                lab_logits2[labeled_pixels], labels[labeled_pixels].float()
            )

            ## unsupervised part ##

            # cross-presudo supervision loss
            if aug_method is None or aug_method == 'affine':

                # get model predictions for unlabeled data
                unlab_logits1 = model1(unlabeled_images)["out"][:, 0]
                unlab_logits2 = model2(unlabeled_images)["out"][:, 0]

                # stop gradient for predictions
                unlab_preds1 = (unlab_logits1 > 0).detach().clone()
                unlab_preds2 = (unlab_logits2 > 0).detach().clone()

                # cross-presudo supervision loss
                loss_cps = F.binary_cross_entropy_with_logits(
                    unlab_logits1, unlab_preds2.float()
                ) + F.binary_cross_entropy_with_logits(
                    unlab_logits2, unlab_preds1.float()
                )

            # cutmix-style loss
            elif aug_method in ['cutmix', 'mixup']:

                # augment the unlabeled images
                (
                    mix_unlab_images,
                    unlab_images_a,
                    unlab_images_b,
                    _,
                    _,
                    unlab_mask,
                ) = utils.image_augmentation(aug_method, unlabeled_images)

                # get predictions for images_a and images_b separately in both models
                unlab_logits1_images_a = model1(unlab_images_a)["out"][:, 0]
                unlab_logits1_images_b = model1(unlab_images_b)["out"][:, 0]
                unlab_logits2_images_a = model2(unlab_images_a)["out"][:, 0]
                unlab_logits2_images_b = model2(unlab_images_b)["out"][:, 0]

                # apply augmentation to outputs
                unlab_target1_mix = utils.apply_image_aug_to_output(
                    aug_method, unlab_logits1_images_a, unlab_logits1_images_b, unlab_mask
                )
                unlab_target2_mix = utils.apply_image_aug_to_output(
                    aug_method, unlab_logits2_images_a, unlab_logits2_images_b, unlab_mask
                )

                # get predictions for augmented images
                unlab_logits1_mix = model1(mix_unlab_images)["out"][:, 0]
                unlab_logits2_mix = model2(mix_unlab_images)["out"][:, 0]

                # stop gradient
                unlab_preds1_mix = (unlab_logits1_mix > 0).detach().clone()
                unlab_preds2_mix = (unlab_logits2_mix > 0).detach().clone()

                # compute cutmix-style loss
                loss_cps = F.binary_cross_entropy_with_logits(
                    unlab_target1_mix, unlab_preds2_mix.float()
                ) + F.binary_cross_entropy_with_logits(
                    unlab_target2_mix, unlab_preds1_mix.float()
                )

            # cutout loss
            elif aug_method == 'cutout':

                # augment the unlabeled images
                (
                    masked_images,
                    _,
                    _,
                    _,
                    _,
                    mask_coordinates,
                ) = utils.image_augmentation(aug_method, unlabeled_images)

                # get predictions for original images
                unlab_logits1_orig = model1(unlabeled_images)["out"][:, 0]
                unlab_logits2_orig = model2(unlabeled_images)["out"][:, 0]

                # apply augmentation to outputs
                # the blacked out regions are set to 2
                mask = utils.apply_image_aug_to_output(
                    aug_method, unlab_logits1_orig, None, mask_coordinates
                ) != 2

                # get predictions for masked images
                unlab_logits1_masked = model1(masked_images)["out"][:, 0]
                unlab_logits2_masked = model2(masked_images)["out"][:, 0]

                # stop gradient
                unlab_preds1_masked = (unlab_logits1_masked > 0).detach().clone()
                unlab_preds2_masked = (unlab_logits2_masked > 0).detach().clone()

                # compute loss
                loss_cps = F.binary_cross_entropy_with_logits( 
                    unlab_logits1_orig[mask], unlab_preds2_masked[mask].float()
                ) + F.binary_cross_entropy_with_logits(
                    unlab_logits2_orig[mask], unlab_preds1_masked[mask].float()
                )

            # total loss = loss_sup + lamb * loss_cps
            loss = loss_sup + lamb * loss_cps
            loss.backward()
            optimizer.step()

            ## training metrics ##
            lab_probs1 = torch.sigmoid(lab_logits1)
            acc = acc_eval(lab_probs1, labels, labeled_pixels)
            iou = iou_eval(lab_probs1, labels, labeled_pixels)
            dice = dice_eval(lab_probs1, labels, labeled_pixels)
            train_loss += loss.item() / n_batches
            train_acc += acc.item() / n_batches
            train_iou += iou.item() / n_batches
            train_dice += dice.item() / n_batches

        # print statistics
        val_loss, val_acc, val_iou, val_dice = eval(model1, valid_dl)
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
            f"epoch={epoch+1:2}, time={epoch_time:5.2f}, {train_acc=:4.2%}, {val_acc=:4.2%}"
        )

    # test metrics
    test_loss, test_acc, test_iou, test_dice = eval(model1, test_dl)
    wandb.log(
        {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_iou": test_iou,
            "test_dice": test_dice,
        }
    )

    return model1