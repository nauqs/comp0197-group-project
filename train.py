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
        labeled_pixels = labels != 3

        # get model predictions
        logits = model(images)['out'][:, 0]

        # compute loss, accuracy on labeled pixels
        loss = F.binary_cross_entropy_with_logits(logits[labeled_pixels], labels[labeled_pixels].float())
        acc = ((logits[labeled_pixels] > 0.5) == labels[labeled_pixels]).float().mean()
        epoch_loss += loss.item() / n_batches
        epoch_acc += acc.item() / n_batches
    
    return epoch_loss, epoch_acc


def train(model, train_all_dl, train_lab_dl, valid_dl, epochs=1, lr=1e-3):
    """
    Trains a segmentation model.
    
    The dataloader 'train_lab_dl' should contain only labeled images
    and is only used for the supervised loss.
    
    The dataloader 'train_all_dl' should contain *both* labeled and unlabeld
    images. It will be used for the cross-pseudo-label consistency loss.
    
    Currently, this function only performs supervised training. But
    we could adapt it to do either supervised or semisupervised training,
    based on input arguments?
    TODO: semi-supervised training
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
            labeled_pixels = labels != 3

            # zero gradients
            optimizer.zero_grad()
            
            # get model predictions
            logits = model(images)['out'][:, 0]

            # compute loss, accuracy on labeled pixels
            loss = F.binary_cross_entropy_with_logits(logits[labeled_pixels], labels[labeled_pixels].float())
            acc = ((logits[labeled_pixels] > 0.5) == labels[labeled_pixels]).float().mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / n_batches
            train_acc += acc.item() / n_batches
            print(loss.item())

        # print statistics
        val_loss, val_acc = eval(model, valid_dl)
        epoch_time = -t + (t := time()) # time per epoch
        print(f'epoch={epoch+1:2}, time={epoch_time:5.2f}, {train_acc=:4.2%}, {valid_acc=:4.2%}')

    return model
