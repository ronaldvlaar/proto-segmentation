"""
format
python -m analysis.pruned_len baseline-imnet-pascal pruned -p
"""

import json
import os
import argh
import numpy as np
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

    print('protos before pruning', len(cls2name)*10)

    proto_m = np.array(range(len(cls2name)*10))
    um = []
    km = []
    with open(json_unique, 'r') as json_file:
        um = json.load(json_file)
    with open(json_pruned, 'r') as json_file:
        km = json.load(json_file)

    proto_m = proto_m[um]
    proto_m = proto_m[km]

    print('protos after pruning', len(proto_m))



if __name__ == '__main__':
    argh.dispatch_command(run)