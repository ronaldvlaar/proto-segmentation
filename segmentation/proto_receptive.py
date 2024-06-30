"""
Example run:

python -m segmentation.proto_receptive pascal_kld_imnet temp
"""
import os
import shutil
from typing import Optional

import argh
import torch
import neptune.new as neptune
import torchvision
from pytorch_lightning import Trainer, seed_everything
import gin
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, CSVLogger

from segmentation.data_module import PatchClassificationDataModule
from segmentation.dataset import PatchClassificationDataset
from segmentation.module import PatchClassificationModule
from segmentation.config import get_operative_config_json
from model import construct_PPNet
from segmentation.push import push_prototypes
from settings import log
from deeplab_features import torchvision_resnet_weight_key_to_deeplab2


Trainer = gin.external_configurable(Trainer)

@gin.configurable(denylist=['config_path', 'experiment_name', 'neptune_experiment', 'pruned'])
def train(
        config_path: str,
        experiment_name: str,
        neptune_experiment: Optional[str] = None,
        pruned: bool = False,
        start_checkpoint: str = '',
        random_seed: int = gin.REQUIRED,
        early_stopping_patience_last_layer: int = gin.REQUIRED,
        warmup_steps: int = gin.REQUIRED,
        joint_steps: int = gin.REQUIRED,
        finetune_steps: int = gin.REQUIRED,
        warmup_batch_size: int = gin.REQUIRED,
        joint_batch_size: int = gin.REQUIRED,
        load_coco: bool = False
):

    results_dir = os.path.join(os.environ['RESULTS_DIR'], experiment_name)

    ppnet = torch.load(os.path.join(results_dir, f'checkpoints/push_last.pth'))
    ppnet = ppnet.cuda()

    module = PatchClassificationModule(
        model_dir=results_dir,
        ppnet=ppnet,
        training_phase=1,
        max_steps=joint_steps
    )

    module.eval()
    torch.set_grad_enabled(False)

    push_dataset = PatchClassificationDataset(
        split_key='train',
        is_eval=True,
        push_prototypes=True
    )
    
    push_prototypes(
        push_dataset,
        prototype_network_parallel=ppnet,
        prototype_layer_stride=1,
        root_dir_for_saving_prototypes=module.prototypes_dir,
        prototype_img_filename_prefix='prototype-img',
        prototype_self_act_filename_prefix='prototype-self-act',
        proto_bound_boxes_filename_prefix='bb',
        save_prototype_class_identity=True,
        pascal=not push_dataset.only_19_from_cityscapes,
        log=log
    )

def load_config_and_train(
        config_path: str,
        experiment_name: str,
        neptune_experiment: Optional[str] = None,
        pruned: bool = False,
        start_checkpoint: str = ''
):
    gin.parse_config_file(f'segmentation/configs/{config_path}.gin')
    train(
        config_path=config_path,
        experiment_name=experiment_name,
        pruned=pruned,
        neptune_experiment=neptune_experiment,
        start_checkpoint=start_checkpoint
    )

if __name__ == '__main__':
    argh.dispatch_command(load_config_and_train)
