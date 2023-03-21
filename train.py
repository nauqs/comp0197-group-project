import torch
import torch.nn.functional as F
from time import time
import utils
from itertools import cycle


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
            lab_preds1 = (lab_logits1 > 0.5).detach().clone()
            lab_preds2 = (lab_logits2 > 0.5).detach().clone()
            unlab_preds1 = (unlab_logits1 > 0.5).detach().clone()
            unlab_preds2 = (unlab_logits2 > 0.5).detach().clone()

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

        # iterate over batches of labeled data
        n_batches = len(train_lab_dl)
        for images, labels in train_lab_dl:

            images, labels = images.to(device), labels.to(device)

            # mix_images, images_a, images_b, labels_a, labels_b, labeled_mask = utils.cutmix(images, labels)

            # #move cutmix_images, cutmix_labels_a, cutmix_labels_b, labeled_mask to device
            # mix_images = mix_images.to(device)
            # images_a = images_a.to(device)
            # images_b = images_b.to(device)
            # labels_a = labels_a.to(device)
            # labels_b = labels_b.to(device)
            # labeled_mask = labeled_mask.to(device)

            unlabeled_images, _ = next(unlabeled_images_iter)
            unlabeled_images = unlabeled_images.to(device)

            # cutmix the unlabeled images
            mix_unlab_images, unlab_images_a, unlab_images_b, _, _, unlab_mask = utils.cutmix(unlabeled_images)

            # get a mask of labeled pixels (foreground/background)
            labeled_pixels = labels != 2
            # zero gradients
            optimizer.zero_grad()

            # get model predictions for labeled data
            lab_logits1 = model1(images)["out"][:, 0]
            lab_logits2 = model2(images)["out"][:, 0]

            # stop gradient for predictions
            lab_preds1 = (lab_logits1 > 0.5).detach().clone()
            lab_preds2 = (lab_logits2 > 0.5).detach().clone()

            # get model predictions for unlabeled data

            # Step 1: get predictions for images_a and images_b separately in both models, and cut-mix them
            unlab_logits1_images_a = model1(unlab_images_a)["out"][:, 0]
            unlab_logits1_images_b = model1(unlab_images_b)["out"][:, 0]
            unlab_logits2_images_a = model2(unlab_images_a)["out"][:, 0]
            unlab_logits2_images_b = model2(unlab_images_b)["out"][:, 0]

            unlab_target1_cutmix = utils.apply_cutmix_mask_to_output(unlab_logits1_images_a, unlab_logits1_images_b, unlab_mask)
            unlab_target2_cutmix = utils.apply_cutmix_mask_to_output(unlab_logits2_images_a, unlab_logits2_images_b, unlab_mask)

            # Step 2: get predicted labels for cut-mix images in both models
            unlab_logits1_cutmix = model1(mix_unlab_images)["out"][:, 0]
            unlab_logits2_cutmix = model2(mix_unlab_images)["out"][:, 0]
            unlab_preds1_cutmix = (unlab_logits1_cutmix > 0.5).detach().clone()
            unlab_preds2_cutmix = (unlab_logits2_cutmix > 0.5).detach().clone()
            
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