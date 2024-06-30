"""
format
python -m analysis.tau_over_k baseline-imnet-pascal pruned -p
"""

import json
import os
import argh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

    # RESULTS_DIR = os.path.join(model_path, f'analysis/{training_phase}/patch_activations_for_protos')
    # os.makedirs(RESULTS_DIR, exist_ok=True)

    tau_k_list = []
    for proj_idx, proto_idx in zip(km, proto_m):
        proto_img= os.path.join(model_path, 'prototypes/'+cls2name[int(proto_idx/10)]+'/prototype-img_'+str(proto_idx)+'.png')
        
        patch_path = os.path.join(model_path, 'pruned/img/'+str(proj_idx))

        patches = os.listdir(patch_path)
        patches = list(filter(lambda x: '_original_with_heatmap_and_patch_' in x, patches))
        patches = list(map(lambda x : os.path.join(patch_path, x), patches))
        patches = patches[:-1]
        
        patch_class = list(map(lambda x: int(x.split('_')[-1][:-4]), patches))
        proto_class = int(proto_idx/10)

        tau_k = patch_class.count(proto_class)/len(patch_class)
        # print(tau_k, len(patch_class))
        tau_k_list.append(tau_k)

    print(len(tau_k_list), np.mean(tau_k_list), np.std(tau_k_list))

if __name__ == '__main__':
    argh.dispatch_command(run)
