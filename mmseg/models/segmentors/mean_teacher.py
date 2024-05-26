from typing import Dict, Optional, Tuple, Union, List
from copy import deepcopy
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix, ForwardResults)

from mmengine.model import MMDistributedDataParallel
from mmengine.optim import OptimWrapper

from .base import BaseSegmentor
from mmseg.models import build_segmentor
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

from mmseg.models.utils.strong_transform import (denorm, get_mean_std, get_cut_masks, strong_transform)


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@MODELS.register_module()
class MeanTeacher(BaseSegmentor):

    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__(data_preprocessor=cfg['data_preprocessor'])

        self.local_iter = 0
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.model = MODELS.build(deepcopy(cfg['model']))
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = MODELS.build(ema_cfg)

        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']

    def get_model(self):
        return get_module(self.model)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def forward_dummy(self, img):
        return self.get_model().forward_dummy(img)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        return self.get_model()._forward(inputs, data_samples)

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        return self.get_model().predict(inputs, data_samples)

    def generete_pseudo_weight(self, ema_logits, dev):
        # ema_softmax = torch.softmax(ema_logits.detach(), dim=0)
        mask = (ema_logits < 0.5).all(dim=0)
        pseudo_prob, pseudo_label = torch.max(ema_logits.detach(), dim=0)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu())) - torch.sum(mask).item()
        if ps_size != 0:
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        else:
            pseudo_weight = 0
        if pseudo_weight == 0.:
            pseudo_weight = 1e-6

        return torch.tensor(pseudo_weight).to(dev)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # Init/update ema model

        batch_size = inputs.shape[0] // 2
        source_img = inputs[:batch_size]
        target_img = inputs[batch_size:]
        source_data_samples = data_samples[:batch_size]
        target_data_samples = data_samples[batch_size:]
        losses = dict()
        if self.local_iter == 20000:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 20000:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        self.local_iter += 1

        gt_losses = self.get_model().loss(source_img, source_data_samples)
        losses.update(gt_losses)
        # return losses
        if self.local_iter < 20000:
            return losses

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        batch_img_metas = [
            data_sample.metainfo for data_sample in target_data_samples
        ]
        with torch.no_grad():
            ema_logits = self.get_ema_model().encode_decode(
                target_img, batch_img_metas)

        data_samples = []

        B, C, H, W = ema_logits.shape
        dev = inputs.device

        means, stds = get_mean_std(self.data_preprocessor)
        strong_parameters = {
            'mix': get_cut_masks(inputs.shape, device=dev),
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means.unsqueeze(0),  # assume same normalization
            'std': stds.unsqueeze(0)
        }

        # mix_masks = get_cut_masks(inputs.shape, device=dev)

        augmented_inputs, ema_logits = strong_transform(strong_parameters, data=inputs, target=ema_logits)
        # generate pseudo labels
        for i in range(B):
            m_v = torch.ones((H, W), device=dev, dtype=torch.long) * 255
            for c in range(C):
                m_v[ema_logits[i][c] > 0.5] = c
            data_sample = SegDataSample()
            data_sample.set_metainfo(batch_img_metas[i])
            data_sample.gt_sem_seg = PixelData(data=m_v)
            data_samples.append(data_sample)

        semi_losses = self.get_model().loss(augmented_inputs, data_samples)
        for key in semi_losses:
            if 'loss' in key:
                semi_losses[key] *= 0.05
        losses.update(add_prefix(semi_losses, 'semi'))

        return losses

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.get_model().decode_head.loss(inputs, augmented_inputs, data_samples,
                                                        self.model.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)
