import torch
import datasets, models, train, utils
from pathlib import Path


if __name__ == '__main__':
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')
    
    # create dataloaders
    train_all_dl, train_lab_dl, valid_dl = datasets.create_dataloaders(batch_size=32)
    
    # initialize model
    model = models.load_deeplab()
    model = model.to(device)
    
    # train
    train.train(model, train_all_dl, train_lab_dl, valid_dl, epochs=10)
    
    # save model
    Path('weights').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'weights/supervised.pt')
    
    # visualize predictions
    images, labels = next(iter(valid_dl))
    images, labels = images[:8].to(device), labels[:8].to(device)
    logits = model(images)['out'][:, 0]
    utils.visualize_predictions(images, logits, 'plots/predictions.jpg')