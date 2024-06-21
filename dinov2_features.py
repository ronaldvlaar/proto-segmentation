import gin

from segmentation.utils import MSC
from deeplab_pytorch.libs.models.deeplabv2 import _ASPP

import torch
import torch.nn as nn
import math

class DinoV2WithASPP(nn.Module):
    def __init__(self, dino_model, n_classes, atrous_rates):
        super(DinoV2WithASPP, self).__init__()
        self.dino_model = dino_model
        self.aspp = _ASPP(dino_model.num_features, n_classes, atrous_rates).to('cuda')

    def closest_factors(self, n):
        root = int(math.sqrt(n))
        for i in range(root, 0, -1):
            if n % i == 0:
                return i, n // i

    def crop_to_closest_patch_size(self, w, h, patch_size=14):
        w_new = (w // patch_size) * patch_size
        h_new = (h // patch_size) * patch_size
        
        return w_new, h_new

    def crop(self, image):
        _, _, w, h = image.size()
        w_new, h_new = self.crop_to_closest_patch_size(w, h, patch_size=self.dino_model.patch_size)
        
        return image[:, :, :w_new, :h_new]

    def freeze(self):
        for param in self.dino_model.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x = self.dino_model.forward_features(self.crop(x))['x_norm_patchtokens']

        batch_size, num_patches, num_features = x.size()
        height = width = int(num_patches ** 0.5)
        height, width = self.closest_factors(num_patches)
        x = x.permute(0, 2, 1).contiguous().view(batch_size, num_features, height, width)
        x = self.aspp(x)

        return x
    
    def freeze_bn(self):
        """
        Dinov2 does not have batch normalization layer. Instead it uses Layernorm which normalizes across the feature dimensions.
        The independence of batch size makes it suitable for small batch sizes like the batch size of 2 of the ProtoSeg baseline.
        This function serves as placeholder.
        """
        pass


def get_scales(size, scales=[0.5, 0.75], patch=14):
    """
    Rescales scales such the scaled samples become a multiple of the dinov2 patch_size.
    This is a requirement of the dinov2 model.
    """
    new_scales = []
    for scale in scales:
        scaled_size = scale*size
        while scaled_size % patch != 0:
            scaled_size += scale*patch
        new_scale = scaled_size/size
        new_scales.append(new_scale)

    return new_scales


@gin.configurable(allowlist=['deeplab_n_features', 'scales', 'size'])
def dinov2_features(pretrained=False, deeplab_n_features: int = gin.REQUIRED,
                                 scales=[1.0], size: int = gin.REQUIRED, **kwargs):
    # dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to('cuda')
    dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to('cuda')
    scales=get_scales(size, scales=scales, patch=dino_model.patch_size)

    base=DinoV2WithASPP(dino_model, n_classes=deeplab_n_features, atrous_rates=[6, 12, 18, 24])
    base.freeze()
    base.dino_model.eval()

    return MSC(
        base=base,
        scales=scales,
    )


if __name__ == '__main__':
    features = dinov2_features(pretrained=True, deeplab_n_features=64, scales=[0.5, 0.75], size=322)
    
    sample_input = torch.randn(2, 3, 321, 321).to('cuda')  # Example input tensor with shape [batch_size, channels, height, width]
    
    print(features.base.crop(sample_input).shape)
    import time
    features(sample_input)

    start = time.time()
    for _ in range(10):
        output = features(sample_input)
    end = time.time()
    
    forward_pass_time = end - start
    print(f"Forward pass time: {forward_pass_time:.2f} seconds")

    for e in features(sample_input):
        print(e.shape)
