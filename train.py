import torch
import torch.nn.functional as F
from time import time


@torch.no_grad()
def eval(model, dataloader):
    """
    Evaluates model loss and accuracy over one epoch.
    """
    device = next(model.parameters()).device
    epoch_loss = 0
    epoch_acc = 0

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
        acc = ((logits[labeled_pixels] > 0.5) == labels[labeled_pixels]).float().mean()
        epoch_loss += loss.item() / n_batches
        epoch_acc += acc.item() / n_batches

    return epoch_loss, epoch_acc


def train_supervised(model, train_lab_dl, valid_dl, epochs, lr=1e-3):
    """
    Trains a segmentation model.

    The dataloader 'train_lab_dl' should contain only labeled images
    and is only used for the supervised loss.
    """
    device = next(model.parameters()).device

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # iterate over epochs
    for epoch in range(epochs):
        t = time()
        train_loss = 0
        train_acc = 0

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
            acc = (
                ((logits[labeled_pixels] > 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / n_batches
            train_acc += acc.item() / n_batches

        # print statistics
        val_loss, val_acc = eval(model, valid_dl)
        epoch_time = -t + (t := time())  # time per epoch
        print(
            f"epoch={epoch+1:2}, time={epoch_time:5.2f}, {train_acc=:4.2%}, {val_acc=:4.2%}"
        )

    return model


def train_semi_supervised(
    model1, model2, train_unlab_dl, train_lab_dl, valid_dl, epochs, lr=1e-3, lamb=6
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

    # set optimizer to optimise both models
    optimizer = torch.optim.Adam([*model1.parameters(), *model2.parameters()], lr=lr)

    # iterable for unlabeled images
    # we will need to sample a few with every labeled batch
    # if there are more labeled than unlabeled, we will need to loop around, hence the iter
    unlabeled_images_iter = iter(train_unlab_dl)

    # iterate over epochs
    for epoch in range(epochs):
        t = time()
        # loss is combined, accuracy is per model
        train_loss = 0
        train_acc1 = 0
        train_acc2 = 0

        # iterate over batches of labeled data
        n_batches = len(train_lab_dl)
        for images, labels in train_lab_dl:
            images, labels = images.to(device), labels.to(device)
            try:
                # give next batch if we have more left
                # don't pick up labels
                unlabeled_images, _ = next(unlabeled_images_iter)
                unlabeled_images = images.to(device)
            except StopIteration:
                # we ran out of unlabeled, restart the loop
                # this only happens if there are more labeled than unlabeled, or batches are unequal
                unlabeled_images_iter = iter(train_unlab_dl)
                unlabeled_images = next(unlabeled_images_iter).to(device)

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
            lab_preds1 = (lab_logits1 > 0.5).detach().clone()
            lab_preds2 = (lab_logits2 > 0.5).detach().clone()
            unlab_preds1 = (unlab_logits1 > 0.5).detach().clone()
            unlab_preds2 = (lab_logits2 > 0.5).detach().clone()

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
            # accuracy is only computed on labeled pixels
            acc1 = (
                ((lab_logits1[labeled_pixels] > 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )
            acc2 = (
                ((lab_logits2[labeled_pixels] > 0.5) == labels[labeled_pixels])
                .float()
                .mean()
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / n_batches
            train_acc1 += acc1.item() / n_batches
            train_acc2 += acc2.item() / n_batches

        # print statistics
        val_loss1, val_acc1 = eval(model1, valid_dl)
        val_loss2, val_acc2 = eval(model2, valid_dl)
        epoch_time = -t + (t := time())  # time per epoch
        print(
            f"epoch={epoch+1:2}, time={epoch_time:5.2f}, {train_acc1=:4.2%},{train_acc2=:4.2%}, {val_acc1=:4.2%}, {val_acc2=:4.2%}"
        )

    return model1, model2
