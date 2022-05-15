"""
Pytorch Lightning Module for training prototype segmentation model on Cityscapes and SUN datasets
"""
import os
from typing import Dict, Optional

import gin
import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from helpers import list_of_distances
from model import PPNet
from save import save_model_w_condition
from settings import log
from train_and_test import warm_only, joint, last_only

ReduceLROnPlateau = gin.external_configurable(ReduceLROnPlateau)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def reset_metrics() -> Dict:
    return {
        'n_examples': 0,
        'n_correct': 0,
        'pos_scores': [],
        'neg_scores': [],
        'tp': 0,
        'tn': 0,
        'fp': 0,
        'fn': 0,
        'n_batches': 0,
        'n_patches': 0,
        'cross_entropy': 0,
        # 'cluster_cost': 0,
        # 'separation': 0,
        # 'separation_higher': 0,
        # 'contrastive_loss': 0,
        # 'orthogonal_loss': 0,
        # 'object_dist_loss': 0,
        # 'prototype_relevance_loss': 0,
        'loss': 0
    }


def update_lr_warmup(optimizer, current_step, warmup_batches):
    if warmup_batches > 0:
        if current_step < warmup_batches:
            lr_scale = min(1., float(current_step + 1) / warmup_batches)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * optimizer.defaults['lr']


# noinspection PyAbstractClass
@gin.configurable(denylist=['model_dir', 'model_image_size', 'ppnet', 'last_layer_only'])
class PatchClassificationModule(LightningModule):
    def __init__(
            self,
            model_dir: str,
            model_image_size: int,
            ppnet: PPNet,
            last_layer_only: bool,
            num_warm_epochs: int = gin.REQUIRED,
            target_tau: float = 1.0,
            tau_decrease_r: float = 0.0,
            update_tau_every_n: int = 0,
            loss_weight_crs_ent: float = gin.REQUIRED,
            loss_weight_contrastive: float = gin.REQUIRED,
            loss_weight_object: float = 0.0,
            loss_weight_orthogonal: float = 0.0,
            loss_weight_clst: float = gin.REQUIRED,
            loss_weight_sep: float = gin.REQUIRED,
            loss_weight_proto_rel: float = 0.0,
            loss_weight_l1: float = gin.REQUIRED,
            joint_optimizer_lr_features: float = gin.REQUIRED,
            joint_optimizer_lr_add_on_layers: float = gin.REQUIRED,
            joint_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
            joint_optimizer_weight_decay: float = gin.REQUIRED,
            warm_optimizer_lr_add_on_layers: float = gin.REQUIRED,
            warm_optimizer_lr_prototype_vectors: float = gin.REQUIRED,
            warm_optimizer_weight_decay: float = gin.REQUIRED,
            last_layer_optimizer_lr: float = gin.REQUIRED,
            warmup_batches: int = gin.REQUIRED,
            decrease_lr_after_batches: int = 0,
            gradient_clipping: Optional[float] = gin.REQUIRED,
            ignore_void_class: bool = False
    ):
        super().__init__()
        self.model_dir = model_dir
        self.prototypes_dir = os.path.join(model_dir, 'prototypes')
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        self.model_image_size = model_image_size
        self.ppnet = ppnet
        self.last_layer_only = last_layer_only
        self.num_warm_epochs = num_warm_epochs
        self.target_tau = target_tau
        self.tau_decrease_r = tau_decrease_r
        self.update_tau_every_n = update_tau_every_n
        self.loss_weight_crs_ent = loss_weight_crs_ent
        self.loss_weight_contrastive = loss_weight_contrastive
        self.loss_weight_object = loss_weight_object
        self.loss_weight_orthogonal = loss_weight_orthogonal
        self.loss_weight_clst = loss_weight_clst
        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_proto_rel = loss_weight_proto_rel
        self.loss_weight_l1 = loss_weight_l1
        self.joint_optimizer_lr_features = joint_optimizer_lr_features
        self.joint_optimizer_lr_add_on_layers = joint_optimizer_lr_add_on_layers
        self.joint_optimizer_lr_prototype_vectors = joint_optimizer_lr_prototype_vectors
        self.joint_optimizer_weight_decay = joint_optimizer_weight_decay
        self.warm_optimizer_lr_add_on_layers = warm_optimizer_lr_add_on_layers
        self.warm_optimizer_lr_prototype_vectors = warm_optimizer_lr_prototype_vectors
        self.warm_optimizer_weight_decay = warm_optimizer_weight_decay
        self.last_layer_optimizer_lr = last_layer_optimizer_lr
        self.warmup_batches = warmup_batches
        self.decrease_lr_after_batches = decrease_lr_after_batches
        self.gradient_clipping = gradient_clipping
        self.ignore_void_class = ignore_void_class

        os.makedirs(self.prototypes_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # initialize variables for computing metrics
        self.metrics = {}
        for split_key in ['train', 'val', 'test', 'train_last_layer']:
            self.metrics[split_key] = reset_metrics()

        # initialize configure_optimizers()
        self.lr_scheduler = None

        # we use optimizers manually
        self.automatic_optimization = False

        self.best_loss = 10e6

        # use this for distribution prediction
        # self.loss = torch.nn.KLDivLoss(reduction='batchmean')

        # self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x):
        return self.ppnet(x)

    def _step(self, split_key: str, batch, batch_idx):
        metrics = self.metrics[split_key]

        if len(batch) == 2:
            image, target = batch
            object_mask = None
        else:
            image, target, object_mask = batch

            # make object IDs unique within batch
            # sample_idx = torch.arange(object_mask.shape[0], device=object_mask.device).unsqueeze(-1).unsqueeze(-1)
            # max_obj_num = torch.max(object_mask)
            # object_mask = (sample_idx * max_obj_num) + object_mask

        image = image.to(self.device)
        target = target.to(self.device).to(torch.float32)
        # if object_mask is not None:
            # object_mask = object_mask.to(self.device).to(torch.float32)

        output, patch_distances, patch_features = self.ppnet.forward_with_features(image)
        del patch_features

        # treat each patch as a separate sample in calculating loss
        # log_output_flat = torch.nn.functional.log_softmax(output.reshape(-1, output.shape[-1]), dim=-1)
        # target_flat = target.reshape(-1, target.shape[-1])

        output_flat = output.reshape(-1, output.shape[-1])
        # features_flat = patch_features.permute(0, 2, 3, 1).reshape(-1, patch_features.shape[1])
        dist_flat = patch_distances.permute(0, 2, 3, 1).reshape(-1, patch_distances.shape[1])

        if target.ndim > 3:
            if object_mask is not None:
                raise NotImplementedError('TODO')
            target_flat = target.reshape(-1, target.shape[-1])

            # make positive targets have the same weights as negatives in each patch
            n_neg = torch.sum(1 - target_flat, dim=-1).unsqueeze(-1)
            n_pos = torch.sum(target_flat, dim=-1).unsqueeze(-1)

            pos_weight = n_neg / n_pos
            neg_weight = n_pos / n_neg

            weight = pos_weight * target_flat + neg_weight * (1 - target_flat)

            cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
                output_flat,
                target_flat,
                weight=weight
            )

            output_sigmoid = torch.sigmoid(output_flat)
            output_class = (output_sigmoid > 0.5).to(torch.float)
            is_correct = output_class == target_flat

            target_mask = target_flat.bool()

            metrics['pos_scores'] += list(torch.masked_select(output_sigmoid, target_mask).cpu().detach().numpy())
            metrics['neg_scores'] += list(torch.masked_select(output_sigmoid, ~target_mask).cpu().detach().numpy())

            metrics['tp'] += torch.sum(is_correct * target_flat).item()
            metrics['tn'] += torch.sum(is_correct * (1 - target_flat)).item()
            metrics['fp'] += torch.sum(~is_correct * (1 - target_flat)).item()
            metrics['fn'] += torch.sum(~is_correct * target_flat).item()

            # target_oh = target_flat
        else:
            target_flat = target.flatten()

            # if object_mask is not None:
                # object_mask_flat = object_mask.flatten()

            if self.ignore_void_class:
                target_not_void = (target_flat != 0).nonzero().squeeze()
                target_flat = target_flat[target_not_void] - 1
                output_flat = output_flat[target_not_void]
                # features_flat = features_flat[target_not_void]
                # dist_flat = dist_flat[target_not_void]
                # if object_mask is not None:
                    # object_mask_flat = object_mask_flat[target_not_void]

            cross_entropy = torch.nn.functional.cross_entropy(
                output_flat,
                target_flat.long(),
            )

            output_sigmoid = torch.softmax(output_flat, dim=-1)
            output_class = torch.argmax(output_sigmoid, dim=-1)
            is_correct = output_class == target_flat

            # target_oh = torch.nn.functional.one_hot(target_flat.long(), num_classes=self.ppnet.num_classes)

        l1_mask = 1 - torch.t(self.ppnet.prototype_class_identity).to(self.device)
        l1 = (self.ppnet.last_layer.weight * l1_mask).norm(p=1)

        metrics['n_batches'] += 1

        # separation, cluster_cost = [], []

        # if self.ignore_void_class:
            # n_p_per_class = self.ppnet.num_prototypes // self.ppnet.num_classes
        # else:
            # n_p_per_class = self.ppnet.num_prototypes // (self.ppnet.num_classes - 1)



        # max_dist = (self.ppnet.prototype_shape[1]
                    # * self.ppnet.prototype_shape[2]
                    # * self.ppnet.prototype_shape[3])

        # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
        # calculate cluster cost
        # prototypes_of_correct_class = torch.t(torch.index_select(
            # self.ppnet.prototype_class_identity.to(self.device),
            # dim=-1,
            # index=target_flat.long()
        # )).to(self.device)

        # inverted_distances, _ = torch.max((max_dist - dist_flat) * prototypes_of_correct_class, dim=1)
        # cluster_cost = max_dist - inverted_distances

        # calculate separation cost
        # prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        # inverted_distances_to_nontarget_prototypes, _ = \
            # torch.max((max_dist - dist_flat) * prototypes_of_wrong_class, dim=1)

        # separation = max_dist - inverted_distances_to_nontarget_prototypes

        # separation_higher = torch.sum(separation > cluster_cost).item()
        # cluster_cost = torch.mean(cluster_cost)
        # separation = torch.mean(separation)



        # flat_proto_vectors = self.ppnet.prototype_vectors.view(self.ppnet.num_prototypes, -1)

        # TODO maybe we can do it even smarter without the loop
        # for cls_i in range(target_oh.shape[1]):
            # if cls_i == 0 and not self.ignore_void_class:
                # # ignore 'void' class in loop
                # continue

            # if self.ignore_void_class:
                # cls_dists = dist_flat[:, cls_i * n_p_per_class:(cls_i + 1) * n_p_per_class]
                # cls_proto_vectors = flat_proto_vectors[cls_i * n_p_per_class:(cls_i + 1) * n_p_per_class]
            # else:
                # cls_dists = dist_flat[:, (cls_i - 1) * n_p_per_class:cls_i * n_p_per_class]
                # cls_proto_vectors = flat_proto_vectors[(cls_i - 1) * n_p_per_class:cls_i * n_p_per_class]

            # min_cls_dists, _ = torch.min(cls_dists, dim=-1)
            # target_cls = target_oh[:, cls_i]

            # we want to minimize cluster_cost and maximize separation
            # separation.append((1 - target_cls) * min_cls_dists + 512 * target_cls)
            # cluster_cost.append(min_cls_dists * target_cls)

            # proto_dist = torch.cdist(cls_proto_vectors, cls_proto_vectors)
            # proto_dist = proto_dist + 10e6 * torch.triu(torch.ones_like(proto_dist, device=proto_dist.device))
            # orthogonal_loss.append(torch.min(proto_dist))

        # prototype relevance loss:
        # for each prototype, we want it to have at least 1 close point in the batch
        # proto_identity.shape = (num_prototypes, num_classes)
        # dist_flat.shape = (batch_size, num_prototypes)
        # target_oh.shape = (batch_size, num_classes)

        # (batch_size, num_classes) x (num_classes, num_prototypes) = (batch_size, num_prototypes)
        # target_proto_identity.shape = (batch_size, num_prototypes)
        # target_proto_identity = torch.matmul(target_oh.float(),
                                             # self.ppnet.prototype_class_identity.to(target_oh.device).float().T)

        # proto_dist_correct_class.shape = (num_prototypes, batch_size)
        # proto_dist_correct_class = (dist_flat + 10e6 * (target_proto_identity != 1)).T

        # proto_rel_loss.shape = (num_prototypes, )
        # proto_rel_loss, _ = torch.min(proto_dist_correct_class, dim=-1)

        # is_cls_present = (proto_rel_loss < 10e6).nonzero().squeeze()
        # proto_rel_loss = proto_rel_loss[is_cls_present]
        # if len(proto_rel_loss) > 0:
            # proto_rel_loss = torch.mean(proto_rel_loss)
        # else:
            # proto_rel_loss = 0.0

        # separation cost = minimum over distances to all classes that have score==0.0
        # separation = torch.stack(separation, dim=-1)
        # separation, _ = torch.min(separation, dim=-1)

        # cluster cost = maximum over distances to all classes that have score!=0.0
        # cluster_cost = torch.stack(cluster_cost, dim=-1)
        # cluster_cost, _ = torch.max(cluster_cost, dim=-1)

        # try contrastive loss formulation (we want higher 'logits' for separation than for cluster cost)
        # contrastive_input = torch.stack((cluster_cost, separation), dim=-1)
        # contrastive_target = torch.ones(contrastive_input.shape[0], device=contrastive_input.device, dtype=torch.long)
        # contrastive_loss = torch.nn.functional.cross_entropy(contrastive_input, contrastive_target)

        # orthogonal loss - prototypes of same class should be away from each other
        # orthogonal_loss = torch.mean(torch.stack(orthogonal_loss))

        # if object_mask is None:
            # object_dist_loss = 0.0
        # else:
            # object_dist_loss = []
            # # TODO maybe we can do this smarter without the loop
            # for i in torch.unique(object_mask_flat):
                # same_object = (object_mask_flat == i).nonzero().squeeze()
                # if same_object.ndim > 0 and len(same_object) > 1:
                    # obj_features = features_flat[same_object]
                    # mean_feature = torch.mean(obj_features, dim=-1)
                    # mean_dist_to_mean = torch.mean(torch.cdist(mean_feature.unsqueeze(0), obj_features.T))
                    # object_dist_loss.append(mean_dist_to_mean)
#
            # if len(object_dist_loss) > 0:
                # object_dist_loss = torch.mean(torch.stack(object_dist_loss))
            # else:
                # object_dist_loss = 0.0

        # loss = (self.loss_weight_crs_ent * cross_entropy +
                # self.loss_weight_contrastive * contrastive_loss +
                # self.loss_weight_clst * cluster_cost +
                # self.loss_weight_sep * separation +
                # self.loss_weight_object * object_dist_loss +
                # self.loss_weight_orthogonal * orthogonal_loss +
                # self.loss_weight_proto_rel * proto_rel_loss +
                # self.loss_weight_l1 * l1)

        loss = cross_entropy

        loss_value = loss.item()
        # print(loss_value, cluster_cost.item(), separation.item())
        # print(loss_value)

        if not np.isnan(loss_value):
            metrics['loss'] += loss_value
            metrics['cross_entropy'] += cross_entropy.item()
            # metrics['contrastive_loss'] += contrastive_loss.item()
            # metrics['cluster_cost'] += cluster_cost.item()
            # if object_mask is not None:
                # metrics['object_dist_loss'] += object_dist_loss if isinstance(object_dist_loss, float) \
                    # else object_dist_loss.item()
            # metrics['separation'] += separation.item()
            # metrics['separation_higher'] += separation_higher
            # metrics['orthogonal_loss'] += orthogonal_loss.item()
            # metrics['prototype_relevance_loss'] += proto_rel_loss.item()
            metrics['n_examples'] += target_flat.size(0)
            metrics['n_correct'] += torch.sum(is_correct)
            n_patches = output_flat.shape[0]
            metrics['n_patches'] += n_patches

        if split_key == 'train':
            warm_optim, main_optim = self.optimizers()

            if self.last_layer_only:
                optimizer = main_optim
            else:
                if self.current_epoch < self.num_warm_epochs:
                    optimizer = warm_optim
                else:
                    optimizer = main_optim

            # TODO: This are temporary changes to fit deeplabv3 schedule, they need to be coded nicely
            # update_lr_warmup(optimizer, self.trainer.global_step, self.warmup_batches)

            # # if 0 < self.decrease_lr_after_batches == self.trainer.global_step:
                # for pg in optimizer.param_groups:
                    # pg['lr'] = 0.1 * optimizer.defaults['lr']

            # "poly" learning rate from deeplabv3 paper
            power = 0.9
            max_step = 400 * 750
            optimizer.defaults['lr'] = 0.001

            for pg in optimizer.param_groups:
                pg['lr'] = max(optimizer.defaults['lr'] * ((1 - self.trainer.global_step/max_step) ** power), 0.000001)

            optimizer.zero_grad()
            self.manual_backward(loss)

            # if self.gradient_clipping is not None:
                # torch.nn.utils.clip_grad_norm_(self.ppnet.parameters(), self.gradient_clipping)

            optimizer.step()
            self.log('train_loss_step', loss_value, on_step=True, prog_bar=True)

            lr = get_lr(optimizer)
            self.log('lr', lr, on_step=True)

            if self.ppnet.argmax_only:
                tau_val = self.ppnet.gumbel_softmax_tau.item()
                self.log('gumbel_softmax_tau', tau_val, on_step=True)
                if tau_val > self.target_tau:
                    if self.trainer.global_step % self.update_tau_every_n == 0:
                        new_tau_val = max(self.target_tau, np.exp(-self.tau_decrease_r * self.trainer.global_step))
                        self.ppnet.gumbel_softmax_tau = nn.Parameter(torch.tensor(new_tau_val), requires_grad=False)

    def training_step(self, batch, batch_idx):
        return self._step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step('test', batch, batch_idx)

    def on_train_epoch_start(self):
        if self.last_layer_only:
            last_only(model=self.ppnet, log=log)
        else:
            if self.current_epoch == 0:
                log('WARM-UP START.')

            if self.current_epoch < self.num_warm_epochs:
                warm_only(model=self.ppnet, log=log)
            else:
                joint(model=self.ppnet, log=log)
                if self.current_epoch == self.num_warm_epochs:
                    log('WARM-UP END.')

        # reset metrics
        for split_key in self.metrics.keys():
            self.metrics[split_key] = reset_metrics()

    def on_validation_epoch_end(self):
        val_loss = self.metrics['val']['loss'] / self.metrics['val']['n_batches']

        if self.last_layer_only:
            self.log('training_stage', 2.0)
            stage_key = 'push'
            # self.lr_scheduler.step(val_loss)
        else:
            if self.current_epoch < self.num_warm_epochs:
                # noinspection PyUnresolvedReferences
                self.log('training_stage', 0.0)
                stage_key = 'warmup'
            else:
                # noinspection PyUnresolvedReferences
                self.log('training_stage', 1.0)
                stage_key = 'nopush'
                # self.lr_scheduler.step(val_loss)

        if val_loss < self.best_loss:
            log(f'Saving best model, loss: ' + str(val_loss))
            self.best_loss = val_loss
            save_model_w_condition(
                model=self.ppnet,
                model_dir=self.checkpoints_dir,
                model_name=f'{stage_key}_best',
                accu=val_loss,
                target_accu=0.0,
                log=log
            )
        save_model_w_condition(
            model=self.ppnet,
            model_dir=self.checkpoints_dir,
            model_name=f'{stage_key}_last',
            accu=val_loss,
            target_accu=0.0,
            log=log
        )

    def _epoch_end(self, split_key: str):
        metrics = self.metrics[split_key]
        n_batches = metrics['n_batches']

        for key in ['loss', 'cross_entropy']:
            self.log(f'{split_key}/{key}', metrics[key] / n_batches)

        if len(metrics['pos_scores']) > 0 or len(metrics['neg_scores']) > 0:
            pred_scores = np.concatenate((np.array(metrics['pos_scores']), np.asarray(metrics['neg_scores'])))
            true_scores = np.concatenate((np.ones(len(metrics['pos_scores'])), np.zeros(len(metrics['neg_scores']))))
            true_scores = true_scores.astype(int)

            auroc = roc_auc_score(true_scores, pred_scores)

            self.log(f'{split_key}/auroc', auroc)

            pred_pos = metrics['tp'] + metrics['fp']
            precision = metrics['tp'] / pred_pos if pred_pos != 0 else 1.0
            self.log(f'{split_key}/precision', precision)

            true_pos = metrics['tp'] + metrics['fn']
            recall = metrics['tp'] / true_pos if true_pos != 0 else 1.0
            self.log(f'{split_key}/recall', recall)

            f1 = 2 * (precision * recall) / (precision + recall)
            self.log(f'{split_key}/f1_score', f1)
            self.log(f'{split_key}/accuracy', metrics['n_correct'] / (metrics['n_patches'] * self.ppnet.num_classes))
        else:
            self.log(f'{split_key}/accuracy', metrics['n_correct'] / metrics['n_patches'])

        # self.log(f'{split_key}/separation_higher', metrics['separation_higher'] / metrics['n_patches'])
        self.log('l1', self.ppnet.last_layer.weight.norm(p=1).item())

    def training_epoch_end(self, step_outputs):
        return self._epoch_end('train')

    def validation_epoch_end(self, step_outputs):
        p = self.ppnet.prototype_vectors.view(self.ppnet.num_prototypes, -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        self.log('p dist pair', p_avg_pair_dist.item())

        return self._epoch_end('val')

    def test_epoch_end(self, step_outputs):
        return self._epoch_end('test')

    def configure_optimizers(self):
        warm_optimizer_specs = \
            [
                {
                    'params': self.ppnet.add_on_layers.parameters(),
                    'lr': self.warm_optimizer_lr_add_on_layers,
                    'weight_decay': self.warm_optimizer_weight_decay
                },
                {
                    'params': self.ppnet.prototype_vectors,
                    'lr': self.warm_optimizer_lr_prototype_vectors
                }
            ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        if self.last_layer_only:
            main_optimizer_specs = [
                {
                    'params': self.ppnet.last_layer.parameters(),
                    'lr': self.last_layer_optimizer_lr
                }
            ]
        else:
            main_optimizer_specs = \
                [
                    {
                        'params': self.ppnet.features.parameters(),
                        'lr': self.joint_optimizer_lr_features,
                        'weight_decay': self.joint_optimizer_weight_decay
                    },
                    # bias are now also being regularized
                    {
                        'params': self.ppnet.add_on_layers.parameters(),
                        'lr': self.joint_optimizer_lr_add_on_layers,
                        'weight_decay': self.joint_optimizer_weight_decay
                    },
                    {
                        'params': self.ppnet.prototype_vectors,
                        'lr': self.joint_optimizer_lr_prototype_vectors
                    }
                ]
        main_optimizer = torch.optim.Adam(main_optimizer_specs)
        # self.lr_scheduler = ReduceLROnPlateau(main_optimizer)

        return warm_optimizer, main_optimizer
