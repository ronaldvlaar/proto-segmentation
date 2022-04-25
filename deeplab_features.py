import torchvision
from torch import nn


class DeeplabV3_features(nn.Module):
    def __init__(self, model, layers, **kwargs):
        super(DeeplabV3_features, self).__init__()
        self.model = model
        self.layers = layers

        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

    def forward(self, *args, **kwargs):
        result = self.model.forward(*args, **kwargs)
        return result['out']

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        raise NotImplemented("TODO")


def deeplabv3_resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Coco
    """
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    model.classifier._modules = {k: model.classifier._modules[k] for k in list(model.classifier._modules.keys())[:-1]}

    return DeeplabV3_features(model, [3, 4, 6, 3], **kwargs)