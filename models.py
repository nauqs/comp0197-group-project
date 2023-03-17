import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

def load_deeplab(num_classes=1, use_imagenet_weights=True):
    
    # load deeplab with random initialization
    deeplab = deeplabv3_resnet50(num_classes=num_classes)
    
    # optionally initialize the backbone with imagenet weights 
    if use_imagenet_weights:
        
        # load a resnet50 pretrained on imagenet
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # transfer the resnet's weights to the deeplab's backbone
        deeplab.backbone.load_state_dict(resnet.state_dict(), strict=False)

        # check that each layer in deeplab.backbone has the same weights as the resnet
        for name in deeplab.backbone.state_dict().keys():
            assert torch.allclose(resnet.state_dict()[name], deeplab.backbone.state_dict()[name])
    
    return deeplab