"""
Obtain the class seperability and compactness for a model emperically

python -m analysis.separability_stats baseline-imnet-pascal-relu pruned -p

"""

import os
from collections import deque

import argh
import gin
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
import pickle

from tqdm import tqdm
from segmentation import train
from segmentation.constants import CITYSCAPES_19_EVAL_CATEGORIES, PASCAL_ID_MAPPING
from settings import data_path, log
from helpfunc import get_cls2name, log_json
import pandas as pd
from gsoftmax import compactness, separability
from model import construct_PPNet


def save_pkl(cls_logits, path, downsample=1000000):
    sample = {}
    for k in cls_logits:
        total = len(cls_logits[k])
        downsample = total if downsample > total else downsample
        indices = np.random.choice(total, size=downsample, replace=False) 
        sample[k] = np.array(cls_logits[k])[indices]
    
    with open(path, 'wb') as f:
            pickle.dump(sample, f)


def run_evaluation(model_name: str, training_phase: str, batch_size: int = 2, pascal: bool = False,
                   margin: int = 0):
    print(model_name, training_phase)
    cls2name = get_cls2name(pascal)
    model_path = os.path.join(os.environ['RESULTS_DIR'], model_name)
    config_path = os.path.join(model_path, 'config.gin')
    gin.parse_config_file(config_path)

    print('phase', training_phase)

    if training_phase == 'pruned':
        checkpoint_path = os.path.join(model_path, 'pruned/checkpoints/push_last.pth')
    else:
        checkpoint_path = os.path.join(model_path, f'checkpoints/{training_phase}_last.pth')

    log(f'Loading model from {checkpoint_path}')
    ppnet = construct_PPNet()
    ppnet = torch.load(checkpoint_path)
    ppnet = ppnet.cuda()
    ppnet.eval()

    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    img_dir = os.path.join(data_path, f'img_with_margin_{margin}/val')

    all_img_files = [p for p in os.listdir(img_dir) if p.endswith('.npy')]

    ann_dir = os.path.join(data_path, 'annotations/val')

    ID_MAPPING = PASCAL_ID_MAPPING if pascal else CITYSCAPES_19_EVAL_CATEGORIES
    RESULTS_DIR = os.path.join(model_path, f'emperical/{training_phase}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    CLS_CONVERT = np.vectorize(ID_MAPPING.get)

    np.random.shuffle(all_img_files)

    n_batches = int(np.ceil(len(all_img_files) / batch_size))
    batched_img_files = np.array_split(all_img_files, n_batches)

    np.seterr(divide='ignore', invalid='ignore')

    logits_per_class = {}
    nonlogits_per_class = {}
    for cls_i in range(ppnet.num_classes):
        logits_per_class[cls2name[cls_i]] = deque(np.array([]))
        nonlogits_per_class[cls2name[cls_i]] = deque(np.array([]))

    with torch.no_grad():
        for batch_img_files in tqdm(batched_img_files, desc='evaluating'):
            img_tensors = []
            anns = []
            for img_file in batch_img_files:
                img = np.load(os.path.join(img_dir, img_file)).astype(np.uint8)
                ann = np.load(os.path.join(ann_dir, img_file))
                ann = CLS_CONVERT(ann)

                if margin != 0:
                    img = img[margin:-margin, margin:-margin]

                if pascal:
                    img_shape = (513, 513)
                else:
                    img_shape = ann.shape

                img_tensor = transform(img)
                if pascal:
                    img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=img_shape, mode='bilinear', align_corners=False)[0]
                anns.append(ann)
                img_tensors.append(img_tensor)

            img_tensors = torch.stack(img_tensors, dim=0).cuda()
            batch_logits, batch_distances = ppnet.forward(img_tensors, inference_activation=False)
            batch_logits = batch_logits.permute(0, 3, 1, 2)
            

            for sample_i in range(len(batch_img_files)):
                ann = anns[sample_i]
                logits = torch.unsqueeze(batch_logits[sample_i], 0)
                distances = torch.unsqueeze(batch_distances[sample_i], 0)

                logits = F.interpolate(logits, size=ann.shape, mode='bilinear', align_corners=False)[0]
                distances = F.interpolate(distances, size=ann.shape, mode='bilinear', align_corners=False)[0]
                distances = distances.cpu().detach().numpy()
                
                for cls_i in range(ppnet.num_classes):
                    #mask
                    gt = ann == cls_i + 1
                    class_logits = logits.cpu().detach().numpy()
                    class_logits_gt = class_logits[cls_i][gt]
                    logits_per_class[cls2name[cls_i]].extend(class_logits_gt)
                    nonclass_logits = deque(np.array([]))
                    for cls_o in range(ppnet.num_classes):
                        if cls_o == cls_i:
                            continue
                        nonclass_logits.extend(class_logits[cls_o][gt])
                    total_logits = len(nonclass_logits)
                    halving = int(len(nonclass_logits)//16)
                    min_sample = 5000
                    downsample =  halving if halving > min_sample else min_sample if min_sample < total_logits else total_logits
                    indices = np.random.choice(total_logits, size=downsample, replace=False)
                    # nonlogits_per_class[cls2name[cls_i]].extend(nonclass_logits[indices])
                    nonlogits_per_class[cls2name[cls_i]].extend(np.array(nonclass_logits)[indices])
                    # nonlogits_per_class[cls2name[cls_i]].extend(nonclass_logits)
                    del nonclass_logits
    mus = []
    sigmas = []
    classes = list(logits_per_class.keys())
    for k in classes:
        vals = logits_per_class[k]
        mean = np.mean(vals) if len(vals) > 0 else 'nan'
        std = np.std(vals) if len(vals) > 0 else 'nan'
        mus.append(mean)
        sigmas.append(std)

    print('first for fin')
    save_pkl(logits_per_class, os.path.join(RESULTS_DIR, 'logits_per_class.pkl'), downsample=1000000)
    del logits_per_class
    
    compact = compactness(np.array(sigmas))
    sep = separability(np.array(mus), np.array(sigmas), ppnet.num_classes)
    ratio = compact*sep

    class_sep = []
    nonclass_mus = []
    nonclass_sigmas = []
    for idx, k in enumerate(classes):
        # vals_gt = logits_per_class[k]
        vals_nonclass = nonlogits_per_class[k]
    
        mean_nonclass = np.mean(vals_nonclass)
        # mean_gt = np.mean(vals_gt)
        mean_gt = mus[idx]
        sigma_nonclass = np.std(vals_nonclass)
        # sigma_gt = np.std(vals_gt)
        sigma_gt = sigmas[idx]

        nonclass_mus.append(mean_nonclass)
        nonclass_sigmas.append(sigma_nonclass)
        class_sep.append(separability([mean_nonclass, mean_gt], [sigma_nonclass, sigma_gt], 2)[0])

        # del logits_per_class[k]
        # del nonlogits_per_class[k]

    print('second for fin')
    save_pkl(nonlogits_per_class, os.path.join(RESULTS_DIR, 'nonlogits_per_class.pkl'), downsample=1000000)
    del nonlogits_per_class
        
    nonclass_compact = compactness(np.array(nonclass_sigmas))
    class_ratio = list(compact * np.array(class_sep))

    data={'class' : classes, 
          'mu' : mus, 
          'sigma' : sigmas, 
          'nonclass_mu' : nonclass_mus,
          'nonclass_sigma' : nonclass_sigmas,
          'compactness' : compact,
          'nonclass_compactness' : nonclass_compact,
          'separability' : sep,
          'class_seperability' : class_sep,
          'class_ratio' : class_ratio,
          'ratio' : ratio,
          'avg_compactness' : np.mean(compact),
          'avg_separability' : np.mean(sep),
          'avg_ratio' : np.mean(ratio),
          'avg_class_seperability' : np.mean(class_sep),
          'avg_class_ratio' : np.mean(class_ratio)
          }
    
    print('reached data')
    
    # pd.DataFrame(data).to_json(os.path.join(RESULTS_DIR, 'res3.json'))

    log_json(data, os.path.join(RESULTS_DIR, 'res.json'))

    print('done')

    # with open(os.path.join(RESULTS_DIR, 'logits_per_class.pkl'), 'wb') as f:
    #     pickle.dump(logits_per_class, f)

    # with open(os.path.join(RESULTS_DIR, 'nonlogits_per_class.pkl'), 'wb') as f:
    #     pickle.dump(nonlogits_per_class, f)


if __name__ == '__main__':
    argh.dispatch_command(run_evaluation)
