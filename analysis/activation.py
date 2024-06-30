"""
python -m analysis.activation baseline-imnet-pascal pruned -p
"""

import json
import os
from collections import Counter

import argh
import gin
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image


from segmentation import train
from tqdm import tqdm
from segmentation.dataset import resize_label
from segmentation.constants import CITYSCAPES_CATEGORIES, CITYSCAPES_19_EVAL_CATEGORIES, \
    PASCAL_CATEGORIES, PASCAL_ID_MAPPING
from settings import data_path, log

from find_nearest import to_normalized_tensor
import cv2
import numpy as np
import json
from model import construct_PPNet

to_tensor = transforms.ToTensor()
    

def make_act_map_plots(img: Image,
                               ppnet,
                               savedir,
                               cls2protos,
                               cls2name,
                               proto_m,
                               im_title,
                               model_path,
                               prototype_activation_function_in_numpy=None):

    img_tensor = to_normalized_tensor(img).unsqueeze(0).cuda()
    conv_features = ppnet.conv_features(img_tensor)

    # save RAM
    del img_tensor

    logits, distances = ppnet.forward_from_conv_features(conv_features)

    proto_dist_ = distances[0].permute(1, 2, 0).detach().cpu().numpy()

    del conv_features, distances

    prototype_shape = ppnet.prototype_shape
    n_prototypes = prototype_shape[0]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    # get the whole image
    original_img_j = to_tensor(img).detach().cpu().numpy()
    original_img_j = np.transpose(original_img_j, (1, 2, 0))
    original_img_height = original_img_j.shape[0]
    original_img_width = original_img_j.shape[1]

    # # get segmentation map
    # logits = logits.permute(0, 3, 1, 2)
    # logits_inter = torch.nn.functional.interpolate(logits, size=original_img_j.shape[:2],
    #                                                mode='bilinear', align_corners=False)
    # logits_inter = logits_inter[0]
    # # pred = torch.argmax(logits_inter, dim=0).cpu().detach().numpy()

    protos = ppnet.prototype_vectors.squeeze()
    # print(protos, len(protos), protos[0].shape, protos[0].reshape((8,8)))

    for k in cls2protos.keys():
        classname = cls2name[k]
        protosidxs = cls2protos[k]
        if len(protosidxs) == 0:
            continue

        print('class', classname, ' protos', len(protosidxs), protosidxs, proto_m[protosidxs])

        fig, axs = plt.subplots(3, len(protosidxs), figsize=(5*len(protosidxs), 15))
        # fig.suptitle(classname) 
        for col_idx, j in enumerate(protosidxs):
            # TODO 
            # - Include rf calculated from protos
            # - Include original with box
            # - Create receptive field from bb-receptive_fieldNone.npy
            # - How to define the aggregate of activation maps?
            proto_dist_img_j = proto_dist_[:, :, j]
            if ppnet.prototype_activation_function == 'log':
                proto_act_img_j = np.log(
                    (proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))
            elif ppnet.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)

            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_width, original_img_height),
                                             interpolation=cv2.INTER_CUBIC)
            
            rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
            rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)

            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]

            overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap

            # Receptive field of the prototype processed by backbone. The coordinates correspond (see push.py)
            # and can be used to obtain the patch in the original image. Alternatively, the inverse of the backbone can be used
            # to obain the original patch from the feature map patch
            proto_rf = protos[j].cpu().reshape((8,8))
            # Invert resnet transform *std+mean
            axs[0][col_idx].imshow(overlayed_original_img_j)
            axs[0][col_idx].axis('off')
            # axs[1][col_idx].imshow(proto_rf)
            # axs[1][col_idx].set_title('original_proto_idx'+str(proto_m[j]))\
            
            proto_95_path = os.path.join(model_path, 'prototypes', classname, f'prototype-img_{proto_m[j]}.png')
            with open(proto_95_path, 'rb') as f:
                proto_95 = Image.open(f).convert('RGB')
            axs[1][col_idx].imshow(proto_95)
            # axs[1][col_idx].set_title(f'Prototype {proto_m[j]} - 95% activation')
            axs[1][col_idx].axis('off')

            original_with_bb_path = os.path.join(model_path, 'prototypes', classname, f'prototype-img_{proto_m[j]}-original_with_box.png')
            with open(original_with_bb_path, 'rb') as f:
                original_with_bb = Image.open(f).convert('RGB')
            axs[2][col_idx].imshow(original_with_bb)
            axs[2][col_idx].axis('off')
            # axs[2][col_idx].set_title(f'Protype {proto_m[j]} - patch origin')


        plt.tight_layout()
        # print('savepath', os.path.join(savedir, f'_{classname}-act_img_{im_title}.png'))
        plt.savefig(os.path.join(savedir, f'{im_title}_{classname}-act.png'))
        plt.close()


def run_analysis(model_name: str, training_phase: str, batch_size: int = 2, pascal: bool = False,
                   margin: int = 0):
    print(model_name, training_phase)
    model_path = os.path.join(os.environ['RESULTS_DIR'], model_name)
    config_path = os.path.join(model_path, 'config.gin')
    gin.parse_config_file(config_path)

    if training_phase == 'pruned':
        checkpoint_path = os.path.join(model_path, 'pruned/checkpoints/push_last.pth')
    else:
        checkpoint_path = os.path.join(model_path, f'checkpoints/{training_phase}_last.pth')

    log(f'Loading model from {checkpoint_path}')
    ppnet = construct_PPNet()
    ppnet = torch.load(checkpoint_path)  # , map_location=torch.device('cpu'))
    ppnet = ppnet.cuda()

    img_dir = os.path.join(data_path, f'img_with_margin_{margin}/val')

    if pascal:

        all_img_files = [p for p in os.listdir(img_dir) if p.endswith('.png')][:1]

        all_img_files.append('2009_003343.png')
        all_img_files.append('2009_004859.png')
        all_img_files.append('2009_000354.png')
    else:
        all_img_files = ['munster_000000_000019.npy']


    images = [os.path.join(img_dir, im) for im in all_img_files]

    RESULTS_DIR = os.path.join(model_path, f'analysis/{training_phase}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ID_MAPPING = PASCAL_ID_MAPPING if pascal else CITYSCAPES_19_EVAL_CATEGORIES
    CATEGORIES = PASCAL_CATEGORIES if pascal else CITYSCAPES_CATEGORIES

    cls2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
    if pascal:
        cls2name = {i: CATEGORIES[k + 1] for i, k in cls2name.items() if k < len(CATEGORIES) - 1}
    else:
        cls2name = {i: CATEGORIES[k] for i, k in cls2name.items()}


    if hasattr(ppnet, 'module'):
        ppnet = ppnet.module

    ppnet.eval()

    proto_ident = ppnet.prototype_class_identity.cpu().detach().numpy()

    proto2cls = {}
    cls2protos = {c: [] for c in range(ppnet.num_classes)}

    json_unique = os.path.join(model_path, 'prototypes/unique_prototypes.json')
    json_pruned = os.path.join(model_path, 'pruned/prototypes_to_keep.json')

    print(len(cls2name)*10)

    proto_m = np.array(range(len(cls2name)*10))
    um = []
    km = []
    with open(json_unique, 'r') as json_file:
        um = json.load(json_file)
    with open(json_pruned, 'r') as json_file:
        km = json.load(json_file)

    proto_m = proto_m[um]
    proto_m = proto_m[km]

    for proto_num in range(proto_ident.shape[0]):
        cls = np.argmax(proto_ident[proto_num])
        proto2cls[proto_num] = cls
        cls2protos[cls].append(proto_num)

    for img_path in images:
        if pascal:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        else:
            img = Image.fromarray(np.load(img_path)).convert('RGB')

        # remove margins which were used for training
        img = img.crop((margin, margin, img.width - margin, img.height - margin))

        im_title = img_path.split('/')[-1].replace('.png', '')

        print(im_title)
        with torch.no_grad():
            make_act_map_plots(img, ppnet, RESULTS_DIR, cls2protos, cls2name, proto_m, im_title, model_path,
                                       prototype_activation_function_in_numpy=log)
    
if __name__ == '__main__':
    argh.dispatch_command(run_analysis)
