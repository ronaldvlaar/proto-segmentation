import os
import json
import numpy as np
from collections import deque
from segmentation.constants import CITYSCAPES_CATEGORIES, CITYSCAPES_19_EVAL_CATEGORIES, \
    PASCAL_CATEGORIES, PASCAL_ID_MAPPING


def to_native(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, deque):
        return list(data)
    elif isinstance(data, dict):
        return {key: to_native(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [to_native(element) for element in data]
    else:
        return data


def log_json(data, path):
    data = to_native(data)
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def flatten_dict(d, cls2name):
    flattened_dict = {}
    for k, v in d.items():
        if isinstance(v, list):
            for index, item in enumerate(v):
                new_key = f"{k}_{index}" if len(cls2name) == 0 else f"{k}_{cls2name[index]}"
                flattened_dict[new_key] = item
        else:
            flattened_dict[k] = v

    return flattened_dict


def get_cls2name(pascal : bool):
    ID_MAPPING = PASCAL_ID_MAPPING if pascal else CITYSCAPES_19_EVAL_CATEGORIES
    CATEGORIES = PASCAL_CATEGORIES if pascal else CITYSCAPES_CATEGORIES

    cls2name = {k - 1: i for i, k in ID_MAPPING.items() if k > 0}
    if pascal:
        cls2name = {i: CATEGORIES[k + 1] for i, k in cls2name.items() if k < len(CATEGORIES) - 1}
    else:
        cls2name = {i: CATEGORIES[k] for i, k in cls2name.items()}

    return cls2name
