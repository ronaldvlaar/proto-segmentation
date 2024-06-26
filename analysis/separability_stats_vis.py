"""
visualize the class seperability and compactness

python -m analysis.separability_stats_vis

"""

import os
import pickle as pkl
import json

import matplotlib.pyplot as plt
import numpy as np


m1 = 'baseline-imnet-cityscapes'
m2 = 'baseline-imnet-cityscapes-leakyReLU'
m3 = 'gsoft-imnet-cityscapes-leakyReLU-full'
m4 = 'dino-imnet-cityscapes-leakyReLU'

p = False
pascal = 'Pascal' if p else 'Cityscapes' 

resdir = './analysis/img/'+m1+'--'+m2+'--'+m3+'--'+m4
os.makedirs(resdir, exist_ok=True)


def load_model_data(m, src='/emperical/pruned'):
    m_path = os.path.join(os.environ['RESULTS_DIR'], m+src)
    with open(os.path.join(m_path, 'logits_per_class.pkl'), 'rb') as f:
        m_logits = pkl.load(f)
    with open(os.path.join(m_path, 'nonlogits_per_class.pkl'), 'rb') as f:
        m_nonlogits = pkl.load(f)
    with open(os.path.join(m_path, 'res.json'), 'r') as f:
        m_dict = json.load(f)

    return m_path, m_logits, m_nonlogits, m_dict

m1_path, m1_logits, m1_nonlogits, m1_dict = load_model_data(m1, src='/emperical/pruned')
m2_path, m2_logits, m2_nonlogits, m2_dict = load_model_data(m2, src='/emperical/pruned')
m3_path, m3_logits, m3_nonlogits, m3_dict = load_model_data(m3, src='/emperical/pruned')
m4_path, m4_logits, m4_nonlogits, m4_dict = load_model_data(m4, src='/emperical/pruned')

# print(np.mean(m1_dict['mu']), np.mean(m2_dict['mu']))
# print(np.mean(m1_dict['mu'])-np.mean(m1_dict['nonclass_mu']))
# print(np.mean(m1_dict['nonclass_mu']), np.mean(m2_dict['nonclass_mu']))
# print(np.mean(m2_dict['mu'])-np.mean(m2_dict['nonclass_mu']))


def barplot(m1_dict, m2_dict, m3_dict, keys=['avg_compactness', 'avg_class_seperability', 'avg_class_ratio'], categories=['compactness', 'separability', 'ratio']):
    m1_stats = [m1_dict[k] for k in keys]
    m2_stats = [m2_dict[k] for k in keys]
    m3_stats = [m3_dict[k] for k in keys]
    m4_stats = [m4_dict[k] for k in keys]

    print(m4_stats)

    x = np.arange(len(categories)) 

    width=0.20

    fig, ax = plt.subplots()
    bars1 = ax.bar(x-1.5*width, m1_stats, width, label='Sigmoid', color='green')
    bars2 = ax.bar(x-0.5*width, m2_stats, width, label='leakyReLU', color='yellow')
    bars3 = ax.bar(x+0.5*width, m3_stats, width, label='G-softmax', color='orange')
    bars4 = ax.bar(x+1.5*width, m4_stats, width, label='DINOv2', color='red')

    ax.set_ylabel('value')
    ax.set_title('ProtoSeg '+pascal + ' - feature compactness and separability')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    autolabel(bars4)

    fig.tight_layout()
    plt.savefig(resdir+'/compactness_and_separability.png', dpi=300)

barplot(m1_dict, m2_dict, m3_dict)
