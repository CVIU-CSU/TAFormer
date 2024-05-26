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
class MultiTaskMeanTeacher_ori(BaseSegmentor):

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

    # def _run_forward(self, data: Union[dict, tuple, list],
    #                  mode: str, optim_wrapper: OptimWrapper) -> Union[Dict[str, torch.Tensor], list]:
    #     print(optim_wrapper)
    #     if isinstance(data, dict):
    #         results = self(**data, mode=mode, optim_wrapper=optim_wrapper)
    #     elif isinstance(data, (list, tuple)):
    #         results = self(*data, mode=mode, optim_wrapper=optim_wrapper)
    #     else:
    #         raise TypeError('Output of `data_preprocessor` should be '
    #                         f'list, tuple or dict, but got {type(data)}')
    #     return results

    # def forward(self,
    #             inputs: Tensor,
    #             data_samples: OptSampleList = None,
    #             mode: str = 'tensor',
    #             optim_wrapper: OptimWrapper = None) -> ForwardResults:
    #     print(optim_wrapper)
    #     if mode == 'loss':
    #         return self.loss(inputs, data_samples, optim_wrapper)
    #     elif mode == 'predict':
    #         return self.predict(inputs, data_samples)
    #     elif mode == 'tensor':
    #         return self._forward(inputs, data_samples)
    #     else:
    #         raise RuntimeError(f'Invalid mode "{mode}". '
    #                            'Only supports loss, predict and tensor mode')

    # def train_step(self, data: Union[dict, tuple, list],
    #                optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    #     """Implements the default model training process including
    #     preprocessing, model forward propagation, loss calculation,
    #     optimization, and back-propagation.

    #     During non-distributed training. If subclasses do not override the
    #     :meth:`train_step`, :class:`EpochBasedTrainLoop` or
    #     :class:`IterBasedTrainLoop` will call this method to update model
    #     parameters. The default parameter update process is as follows:

    #     1. Calls ``self.data_processor(data, training=False)`` to collect
    #        batch_inputs and corresponding data_samples(labels).
    #     2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
    #        loss
    #     3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
    #        backward and dict of loss tensor used to log messages.
    #     4. Calls ``optim_wrapper.update_params(loss)`` to update model.

    #     Args:
    #         data (dict or tuple or list): Data sampled from dataset.
    #         optim_wrapper (OptimWrapper): OptimWrapper instance
    #             used to update model parameters.

    #     Returns:
    #         Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
    #     """
    #     # Enable automatic mixed precision training context.
    #     with optim_wrapper.optim_context(self):
    #         data = self.data_preprocessor(data, True)
    #         log_vars = self._run_forward(data, mode='loss', optim_wrapper=optim_wrapper)  # type: ignore
    #     # parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
    #     optim_wrapper._inner_count += 1
    #     if optim_wrapper.should_update():
    #         optim_wrapper.step()
    #         optim_wrapper.zero_grad()
    #     # optim_wrapper.update_params(parsed_losses)
    #     return log_vars

    def generete_pseudo_weight(self, ema_logits, dev):
        # ema_softmax = torch.softmax(ema_logits.detach(), dim=0)
        mask = (ema_logits < 0.5).all(dim=1).cpu()

        pseudo_prob, pseudo_label = torch.max(ema_logits.detach(), dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu())) - torch.sum(mask).item()
        if ps_size == 0:
            pseudo_weight = 1.0
        else:
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        if pseudo_weight == 0:
            pseudo_weight = 1e-6
        # pseudo_weight = 1.0

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

        # b, c, h, w = inputs.shape
        # inputs = inputs.reshape(b*2, c//2, h, w)
        # strong_inputs = inputs[1::2]
        # inputs = inputs[::2]
        losses = dict()
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        self.local_iter += 1

        self.model.decode_head.semi = False
        gt_losses = self.get_model().loss(inputs, data_samples)
        losses.update(gt_losses)
        # return losses
        # if self.local_iter < 20000:
        return losses

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        batch_img_metas = [
            data_sample.metainfo for data_sample in data_samples
        ]
        with torch.no_grad():
            ema_model = self.get_ema_model()
            x = ema_model.extract_feat(inputs)
            ema_logits = ema_model.decode_head.predict(x, batch_img_metas,
                                                  None, True)

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

        mask_vessel = torch.ones((B, H, W), device=dev, dtype=torch.long) * 255
        mask_vessel[ema_logits[:, 0] > 0.5] = 0
        mask_vessel_weight = self.generete_pseudo_weight(ema_logits[:, 0:1], dev)

        mask_odoc = torch.ones((B, H, W), device=dev, dtype=torch.long) * 255
        mask_odoc[ema_logits[:, 1] > 0.5] = 1
        mask_odoc[ema_logits[:, 2] > 0.5] = 2
        mask_odoc_weight = self.generete_pseudo_weight(ema_logits[:, 1:3], dev)

        mask_lesion = torch.ones((B, H, W), device=dev, dtype=torch.long) * 255
        mask_lesion[ema_logits[:, 3] > 0.5] = 3
        mask_lesion[ema_logits[:, 4] > 0.5] = 4
        mask_lesion[ema_logits[:, 5] > 0.5] = 5
        mask_lesion[ema_logits[:, 6] > 0.5] = 6
        mask_lesion_weight = self.generete_pseudo_weight(ema_logits[:, 3:7], dev)

        data_sample = SegDataSample()
        data_sample.set_metainfo(batch_img_metas[0])
        data_sample.gt_sem_seg = PixelData(data=mask_odoc[0])
        data_sample.pseudo_weight = mask_odoc_weight
        data_samples.append(data_sample)

        data_sample = SegDataSample()
        data_sample.set_metainfo(batch_img_metas[0])
        data_sample.gt_sem_seg = PixelData(data=mask_lesion[0])
        data_sample.pseudo_weight = mask_lesion_weight
        data_samples.append(data_sample)

        data_sample = SegDataSample()
        data_sample.set_metainfo(batch_img_metas[1])
        data_sample.gt_sem_seg = PixelData(data=mask_vessel[1])
        data_sample.pseudo_weight = mask_vessel_weight
        data_samples.append(data_sample)

        data_sample = SegDataSample()
        data_sample.set_metainfo(batch_img_metas[1])
        data_sample.gt_sem_seg = PixelData(data=mask_lesion[1])
        # data_sample.gt_sem_seg = PixelData(data=ema_logits.new_full(ema_logits.shape[-2:], 255, dtype=torch.long))
        data_sample.pseudo_weight = mask_lesion_weight
        # data_sample.pseudo_weight = 0
        data_samples.append(data_sample)

        data_sample = SegDataSample()
        data_sample.set_metainfo(batch_img_metas[2])
        data_sample.gt_sem_seg = PixelData(data=mask_vessel[2])
        data_sample.pseudo_weight = mask_vessel_weight
        data_samples.append(data_sample)

        data_sample = SegDataSample()
        data_sample.set_metainfo(batch_img_metas[2])
        data_sample.gt_sem_seg = PixelData(data=mask_odoc[2])
        data_sample.pseudo_weight = mask_odoc_weight
        data_samples.append(data_sample)

        self.model.decode_head.semi = True
        semi_losses = self.get_model().loss(augmented_inputs, data_samples, True)
        # for key in semi_losses:
        #     if 'loss' in key:
        #         semi_losses[key] *= 0.1
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
