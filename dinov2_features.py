import gin

from segmentation.utils import MSC
from deeplab_pytorch.libs.models.deeplabv2 import _ASPP

import torch
import torch.nn as nn

class DinoV2WithASPP(nn.Module):
    def __init__(self, dino_model, n_classes, atrous_rates):
        super(DinoV2WithASPP, self).__init__()
        self.dino_model = dino_model
        self.aspp = _ASPP(dino_model.num_features, n_classes, atrous_rates).to('cuda')

    def forward(self, x):
        print(x.shape, 'ishape')

        # for attr in list(dir(self.dino_model)):
        #     print(attr)
        x = self.dino_model.forward_features(x)['x_norm_patchtokens']
        batch_size, num_patches, num_features = x.size()
        height = width = int(num_patches ** 0.5)
        x = x.permute(0, 2, 1).contiguous().view(batch_size, num_features, height, width)
        print(x.shape, 'shape')
        x = self.aspp(x)
        print(x.shape, 'after aspp')

        return x

# @gin.configurable(allowlist=['deeplab_n_features', 'scales'])
def dinov2_features(pretrained=False, deeplab_n_features: int = gin.REQUIRED,
                                 scales=[1.0], **kwargs):
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to('cuda')

    return MSC(
        base=DinoV2WithASPP(dino_model, n_classes=deeplab_n_features, atrous_rates=[6, 12, 18, 24]
        ),
        scales=scales,
    )


if __name__ == '__main__':
    features = dinov2_features(pretrained=True, deeplab_n_features=64, scales=[0.5, 0.75])

    sample_input = torch.randn(2, 3, 336).to('cuda')  # Example input tensor with shape [batch_size, channels, height, width]
    
    for e in features(sample_input):
        print(e.shape)
