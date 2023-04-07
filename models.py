import torch
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


def load_deeplab(num_classes=1, use_imagenet_weights=False, large_resnet=False):
    """
    Initializes a DeepLabV3 segmentation model according to the paper
    'Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision'

    The model outputs a dictionary where the key 'out' contains
    the predicted mask. If the input images have shape [N, C, H, W], the
    predicted mask has shape [N, 1, H, W]. The '1' corresponds to the
    fact that we are only outputting a single logit, corresponding to
    foreground/background:
    >>> logits = model(images)['out'][:, 0]

    And to get probabilities from the logits, run:
    >>> probs = torch.sigmoid(logits)
    """

    # load deeplab with random initialization

    deeplab = (deeplabv3_resnet101 if large_resnet else deeplabv3_resnet50)(
        num_classes=num_classes
    )

    # optionally initialize the backbone with imagenet weights
    if use_imagenet_weights:
        # load a resnet50 pretrained on imagenet
        weights = (
            ResNet101_Weights.IMAGENET1K_V2
            if large_resnet
            else ResNet50_Weights.IMAGENET1K_V2
        )
        resnet = (resnet101 if large_resnet else resnet50)(weights=weights)

        # transfer the resnet's weights to the deeplab's backbone
        deeplab.backbone.load_state_dict(resnet.state_dict(), strict=False)

        # check that each layer in deeplab.backbone has the same weights as the resnet
        for name in deeplab.backbone.state_dict().keys():
            assert torch.allclose(
                resnet.state_dict()[name], deeplab.backbone.state_dict()[name]
            )

    return deeplab
