"""
format
python -m analysis.odd_one_out baseline-imnet-pascal pruned -p
"""

import json
import os
import argh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from segmentation.constants import CITYSCAPES_CATEGORIES, CITYSCAPES_19_EVAL_CATEGORIES, \
    PASCAL_CATEGORIES, PASCAL_ID_MAPPING

def run(model_name: str, training_phase: str, batch_size: int = 2, pascal: bool = False,
                   margin: int = 0):

    model_path = os.path.join(os.environ['RESULTS_DIR'], model_name)
    json_unique = os.path.join(model_path, 'prototypes/unique_prototypes.json')
    json_pruned = os.path.join(model_path, 'pruned/prototypes_to_keep.json')

    ID_MAPPING = PASCAL_ID_MAPPING if pascal else CITYSCAPES_19_EVAL_CATEGORIES
    CATEGORIES = PASCAL_CATEGORIES if pascal else CITYSCAPES_CATEGORIES

    cls2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
    if pascal:
        cls2name = {i: CATEGORIES[k + 1] for i, k in cls2name.items() if k < len(CATEGORIES) - 1}
    else:
        cls2name = {i: CATEGORIES[k] for i, k in cls2name.items()}

    proto_m = np.array(range(len(cls2name)*10))
    um = []
    km = []
    with open(json_unique, 'r') as json_file:
        um = json.load(json_file)
    with open(json_pruned, 'r') as json_file:
        km = json.load(json_file)

    proto_project = proto_m[um]
    proto_m = proto_project[km]

    RESULTS_DIR = os.path.join(model_path, f'analysis/{training_phase}/patch_activations_for_protos')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    
    for proj_idx, proto_idx in zip(km, proto_m):
        proto_img= os.path.join(model_path, 'prototypes/'+cls2name[int(proto_idx/10)]+'/prototype-img_'+str(proto_idx)+'.png')
        
        patch_path = os.path.join(model_path, 'pruned/img/'+str(proj_idx))

        patches = os.listdir(patch_path)
        patches = list(filter(lambda x: '_original_with_heatmap_and_patch_' in x, patches))
        patches = list(map(lambda x : os.path.join(patch_path, x), patches))
        patches = patches[:-1]
        
        patch_class = list(map(lambda x: int(x.split('_')[-1][:-4]), patches))

        void = -1
        if void in patch_class:
            continue
        
        if patch_class.count(patch_class[0]) == len(patch_class):
            continue

        proto_class = int(proto_idx/10)
        print(patch_class)

        images = [mpimg.imread(path) for path in [proto_img]+patches]
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        # fig, axs = plt.subplots(1, 6, figsize=(18, 5))
        axs = axs.flatten()

        titles = ['Prototype ' +str(proto_idx) + ' - '+cls2name[proto_class]]
        for c in patch_class: 
            titles.append('Patch - '+cls2name[c])

        for ax, img, title in zip(axs, images, titles):
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title)
        # plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, cls2name[proto_class]+'-'+str(proto_idx)+'-'+model_name+'.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    argh.dispatch_command(run)
