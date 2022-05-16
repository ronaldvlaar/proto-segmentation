"""
Training prototype model for image classification.

Example run:

python -m classification.train cityscapes 2022_05_15_coco
"""
import os
import shutil
from typing import Optional

import argh
import torch
import neptune.new as neptune
from pytorch_lightning import Trainer, seed_everything
import gin
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, CSVLogger

from classification.data_module import ImageClassificationDataModule
from preprocess import preprocess
from classification.module import ImageClassificationModule
from segmentation.config import get_operative_config_json
from model import construct_PPNet
from push import push_prototypes
from settings import log

Trainer = gin.external_configurable(Trainer)


@gin.configurable(allowlist=['model_image_size', 'random_seed',
                             'early_stopping_patience_main', 'early_stopping_patience_last_layer',
                             'start_checkpoint'])
def train_cls(
        config_path: str,
        experiment_name: str,
        neptune_experiment: Optional[str] = None,
        pruned: bool = False,
        model_image_size: int = gin.REQUIRED,
        random_seed: int = gin.REQUIRED,
        early_stopping_patience_main: int = gin.REQUIRED,
        early_stopping_patience_last_layer: int = gin.REQUIRED,
        start_checkpoint: str = '',
        start_epoch: int = 0
):
    seed_everything(random_seed)

    results_dir = os.path.join(os.environ['RESULTS_DIR'], experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    log(f'Starting experiment in "{results_dir}" from config {config_path}')

    last_checkpoint = os.path.join(results_dir, 'checkpoints', 'nopush_best.pth')

    if start_checkpoint:
        log(f'Loading checkpoint from {start_checkpoint}')
        ppnet = torch.load(start_checkpoint)
    elif neptune_experiment is not None and os.path.exists(last_checkpoint):
        log(f'Loading last model from {last_checkpoint}')
        ppnet = torch.load(last_checkpoint)
    else:
        ppnet = construct_PPNet(img_size=model_image_size)

    data_module = ImageClassificationDataModule(
        model_image_size=model_image_size,
    )

    logs_dir = os.path.join(results_dir, 'logs')
    os.makedirs(os.path.join(logs_dir, 'tb'), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, 'csv'), exist_ok=True)

    tb_logger = TensorBoardLogger(logs_dir, name='tb')
    csv_logger = CSVLogger(logs_dir, name='csv')
    loggers = [tb_logger, csv_logger]

    json_gin_config = get_operative_config_json()

    tb_logger.log_hyperparams(json_gin_config)
    csv_logger.log_hyperparams(json_gin_config)

    if not pruned:
        use_neptune = bool(int(os.environ['USE_NEPTUNE']))
        if use_neptune:
            if neptune_experiment is not None:
                neptune_run = neptune.init(
                    project="mikolajsacha/protobased-research",
                    run=neptune_experiment
                )
                neptune_logger = NeptuneLogger(
                    run=neptune_run
                )
            else:
                neptune_logger = NeptuneLogger(
                    project="mikolajsacha/protobased-research",
                    tags=[config_path, 'object_classification', 'protopnet'],
                    name=experiment_name
                )
                loggers.append(neptune_logger)

            neptune_run = neptune_logger.run
            neptune_run['config_file'].upload(f'classification/configs/{config_path}.gin')
            neptune_run['config'] = json_gin_config

        shutil.copy(f'classification/configs/{config_path}.gin', os.path.join(results_dir, 'config.gin'))

        log('MAIN TRAINING')
        callbacks = [
            EarlyStopping(monitor='val/loss', patience=early_stopping_patience_main, mode='min')
        ]

        module = ImageClassificationModule(
            model_dir=results_dir,
            model_image_size=model_image_size,
            ppnet=ppnet,
            last_layer_only=False
        )

        trainer = Trainer(logger=loggers, callbacks=callbacks, checkpoint_callback=None,
                          enable_progress_bar=False)
        if start_epoch != 0:
            trainer.fit_loop.current_epoch = start_epoch

        trainer.fit(model=module, datamodule=data_module)
        
        best_checkpoint = os.path.join(results_dir, 'checkpoints', 'nopush_best.pth')
        log(f'Loading best model from {best_checkpoint}')
        ppnet = torch.load(best_checkpoint)

        ppnet = ppnet.cuda()

        log('SAVING PROTOTYPES')
        module.eval()
        torch.set_grad_enabled(False)

        def preprocess_push_input(x):
            return preprocess(x, mean=data_module.norm_mean, std=data_module.norm_std)

        push_prototypes(
            data_module.train_push_dataloader(),
            prototype_network_parallel=ppnet,
            prototype_layer_stride=1,
            class_specific=True,
            preprocess_input_function=preprocess_push_input,
            root_dir_for_saving_prototypes=module.prototypes_dir,
            epoch_number=module.current_epoch,
            prototype_img_filename_prefix='prototype-img',
            prototype_self_act_filename_prefix='prototype-self-act',
            proto_bound_boxes_filename_prefix='bb',
            save_prototype_class_identity=True,
            log=log
        )
    else:
        best_checkpoint = os.path.join(results_dir, 'pruned/pruned.pth')
        log(f'Loading pruned model from {best_checkpoint}')
        ppnet = torch.load(best_checkpoint)
        ppnet = ppnet.cuda()
        trainer = None

        use_neptune = bool(int(os.environ['USE_NEPTUNE']))
        if use_neptune:
            neptune_logger = NeptuneLogger(
                project="mikolajsacha/protobased-research",
                tags=[config_path, 'image_classification', 'protopnet', 'pruned'],
                name=f'{experiment_name}_pruned' if pruned else experiment_name
            )
            loggers.append(neptune_logger)

            neptune_run = neptune_logger.run
            neptune_run['config_file'].upload(f'classification/configs/{config_path}.gin')
            neptune_run['config'] = json_gin_config

    log('LAST LAYER FINE-TUNING')
    callbacks = [
        EarlyStopping(monitor='val/loss', patience=early_stopping_patience_last_layer, mode='min')
    ]

    module = ImageClassificationModule(
        model_dir=os.path.join(results_dir, 'pruned') if pruned else results_dir,
        model_image_size=model_image_size,
        ppnet=ppnet,
        last_layer_only=True
    )

    if trainer is not None:
        current_epoch = trainer.current_epoch
    else:
        current_epoch = start_epoch

    trainer = Trainer(logger=loggers, callbacks=callbacks, checkpoint_callback=None,
                      enable_progress_bar=False)
    if start_epoch != 0:
        trainer.fit_loop.current_epoch = start_epoch
    else:
        trainer.fit_loop.current_epoch = current_epoch + 1
    trainer.fit(model=module, datamodule=data_module)


def load_config_and_train(
        config_path: str,
        experiment_name: str,
        neptune_experiment: Optional[str] = None,
        pruned: bool = False,
        start_epoch: int = 0
):
    gin.parse_config_file(f'classification/configs/{config_path}.gin')
    train_cls(config_path, experiment_name, pruned=pruned, start_epoch=start_epoch,
              neptune_experiment=neptune_experiment)


if __name__ == '__main__':
    argh.dispatch_command(load_config_and_train)