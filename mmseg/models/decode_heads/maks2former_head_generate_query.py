# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet.models.dense_heads import \
        Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = None

from mmengine.structures import InstanceData
from mmengine.model import ModuleList
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList
from mmcv.ops import point_sample
from mmdet.models.utils import get_uncertain_point_coords_with_randomness, multi_apply
from mmdet.utils import reduce_mean, InstanceList


@MODELS.register_module()
class MultiTaskMask2FormerHeadGenerateQuery(MMDET_Mask2FormerHead):
    """Implements the Mask2Former head.

    See `Mask2Former: Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/abs/2112.01527>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 num_classes,
                 align_corners=False,
                 ignore_index=255,
                 two_stage=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index
        self.threshold = 0.5

        feat_channels = kwargs['feat_channels']
        self.hidden_dim = feat_channels
        self.mask_embed = ModuleList()
        for i in range(3):
            self.mask_embed.append(nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, kwargs['out_channels'])))
        self.cls_embed = ModuleList()
        for i in range(3):
            self.cls_embed.append(nn.Linear(feat_channels, self.num_classes + 1))
        # self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.semi = False

        self.two_stage = two_stage
        # learnable query features
        if two_stage:
            # del self.query_feat
            self.enc_output = nn.Linear(feat_channels, feat_channels)
            self.enc_output_norm = nn.LayerNorm(feat_channels)

            self.query_proj = nn.Linear(feat_channels * 2, feat_channels)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_pseudo_weight = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
            if 'pseudo_weight' in data_sample:
                batch_pseudo_weight.append(data_sample.pseudo_weight)
            else:
                batch_pseudo_weight.append(torch.tensor(1.0, device=gt_sem_seg.device))
        return batch_gt_instances, batch_img_metas, batch_pseudo_weight

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas, batch_pseudo_weight = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples[:x[0].shape[0]])
        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas, batch_pseudo_weight)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType, return_query_mask: bool = False) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        # cs = cls_score.squeeze(0)
        # for i in range(cs.shape[0]):
        #     print(i, cs[i])
        # vis_mask = mask_pred.cpu().numpy()
        # vis_mask[vis_mask>0.5] = 255
        # vis_mask[vis_mask<0.5] = 0
        # vis_mask = vis_mask.squeeze(0).astype(np.uint8)
        # print(vis_mask.shape)
        # for i in range(vis_mask.shape[0]):
        #     cv2.imwrite(f"mask/{i+1}.png", vis_mask[i])
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        if return_query_mask:
            b, q, c = cls_score.shape
            _, _, h, w = mask_pred.shape
            index = torch.argmax(cls_score, dim=1).reshape(-1)  # b * c
            i = np.arange(b).repeat(c)
            mask = mask_pred[[i, index]].reshape(b, c, h, w)
            # logits, label = torch.max(cls_score, dim=1)
            # mask[logits<0.5] = 0
            return mask
        else:
            return seg_logits

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        mask_preds_list: List[Tensor],
        task_idx_list: List[int],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_pseudo_weight: List[Tensor],
        return_sampling_results: bool = False
    ) -> Tuple[List[Union[Tensor, int]]]:
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - avg_factor (int): Average factor that is used to average\
                    the loss. When using sampling method, avg_factor is
                    usually the sum of positive and negative priors. When
                    using `MaskPseudoSampler`, `avg_factor` is usually equal
                    to the number of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end.
        """
        results = multi_apply(self._get_targets_single, cls_scores_list,
                              mask_preds_list, task_idx_list, batch_gt_instances,
                              batch_img_metas, batch_pseudo_weight)
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        res = (labels_list, label_weights_list, mask_targets_list,
               mask_weights_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list)

        return res + tuple(rest_results)

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor, task_idx: int,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            pseudo_weight: Tensor) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_inds = pos_inds + task_idx

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # label_weights = gt_labels.new_ones((self.num_queries, ))
        label_weights = gt_labels.new_zeros((self.num_queries, ))
        label_weights[task_idx:task_idx+self.num_queries//3] = 1

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = pseudo_weight

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict],
                     batch_pseudo_weight: List[Tensor]) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        batch_pseudo_weight_list = [batch_pseudo_weight for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list, batch_pseudo_weight_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict],
                             batch_pseudo_weight: List[Tensor]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = []
        mask_preds_list = []
        task_idx_list = []
        num_imgs_per_task = num_imgs // 3
        num_query_per_task = self.num_queries // 3
        if not self.semi:
            for i in range(num_imgs):
                idx = i // num_imgs_per_task
                cls_scores_list.append(cls_scores[i][idx*num_query_per_task:(idx+1)*num_query_per_task])
                mask_preds_list.append(mask_preds[i][idx * num_query_per_task:(idx + 1) * num_query_per_task])
                task_idx_list.append(idx * num_query_per_task)
            (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
            avg_factor) = self.get_targets(cls_scores_list, mask_preds_list, task_idx_list,
                                        batch_gt_instances, batch_img_metas, batch_pseudo_weight)
            labels = torch.stack(labels_list, dim=0)
            # shape (batch_size, num_queries)
            label_weights = torch.stack(label_weights_list, dim=0)
            # shape (num_total_gts, h, w)
            mask_targets = torch.cat(mask_targets_list, dim=0)
            # shape (batch_size, num_queries)
            mask_weights = torch.stack(mask_weights_list, dim=0)
        else:
            for idx in [1, 2]:
                cls_scores_list.append(cls_scores[0][idx*num_query_per_task:(idx+1)*num_query_per_task])
                mask_preds_list.append(mask_preds[0][idx * num_query_per_task:(idx + 1) * num_query_per_task])
                task_idx_list.append(idx * num_query_per_task)
            for idx in [0, 2]:
                cls_scores_list.append(cls_scores[1][idx*num_query_per_task:(idx+1)*num_query_per_task])
                mask_preds_list.append(mask_preds[1][idx * num_query_per_task:(idx + 1) * num_query_per_task])
                task_idx_list.append(idx * num_query_per_task)
            for idx in [0, 1]:
                cls_scores_list.append(cls_scores[2][idx*num_query_per_task:(idx+1)*num_query_per_task])
                mask_preds_list.append(mask_preds[2][idx * num_query_per_task:(idx + 1) * num_query_per_task])
                task_idx_list.append(idx * num_query_per_task)
            (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
             avg_factor) = self.get_targets(cls_scores_list, mask_preds_list, task_idx_list,
                                            batch_gt_instances, batch_img_metas, batch_pseudo_weight)
            # print(len(mask_targets_list), mask_targets_list[0].shape)
            #shape (batch_size, num_queries)
            labels_list[0][60:90] = labels_list[1][60:90]
            label_weights_list[0][60:90] = label_weights_list[1][60:90]
            mask_weights_list[0][60:90] = mask_weights_list[1][60:90]

            labels_list[2][60:90] = labels_list[3][60:90]
            label_weights_list[2][60:90] = label_weights_list[3][60:90]
            mask_weights_list[2][60:90] = mask_weights_list[3][60:90]

            labels_list[4][30:60] = labels_list[5][30:60]
            label_weights_list[4][30:60] = label_weights_list[5][30:60]
            mask_weights_list[4][30:60] = mask_weights_list[5][30:60]

            labels = torch.stack(labels_list[::2], dim=0)
            # shape (batch_size, num_queries)
            label_weights = torch.stack(label_weights_list[::2], dim=0)
            # shape (num_total_gts, h, w)
            mask_targets = torch.cat(mask_targets_list, dim=0)
            # shape (batch_size, num_queries)
            mask_weights = torch.stack(mask_weights_list[::2], dim=0)
        # print('labels:', labels)
        # print("label_weights:", label_weights)
            # print(mask_targets.shape)
        # print("mask_weights:", mask_weights)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        class_weight = class_weight[labels]
        class_weight[(label_weights.flatten())==0] = 0

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight.sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_weights = mask_weights[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, weight=mask_weights, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_weights, _ = torch.broadcast_tensors(mask_weights.unsqueeze(1), mask_point_preds)
        mask_weights = mask_weights.reshape(-1)
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            weight=mask_weights,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice


    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c)
        num_query_per_task = self.num_queries // 3
        cls_pred = []
        for i in range(3):
            cls_pred.append(self.cls_embed[i](decoder_out[:, i*num_query_per_task:(i+1)*num_query_per_task]))
        cls_pred = torch.cat(cls_pred, dim=1)
        # cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = []
        for i in range(3):
            mask_embed.append(self.mask_embed[i](decoder_out[:, i*num_query_per_task:(i+1)*num_query_per_task]))
        mask_embed = torch.cat(mask_embed, dim=1)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        cls_pred_list = []
        mask_pred_list = []
        if not self.two_stage:
            # shape (num_queries, c) -> (num_queries, batch_size, c)
            query_feat = self.query_feat.weight.unsqueeze(1).repeat(
                (1, batch_size, 1))
        else:
            # src_flatten = torch.cat(decoder_inputs, dim=0).transpose(0, 1) # bs, sum{hxw}, c
            src_flatten = mask_features.flatten(2).transpose(1, 2)
            # src_flatten = torch.cat([src_flatten, mask_feature_flatten], dim=1)
            output_memory = self.enc_output_norm(self.enc_output(src_flatten))
            topk = self.num_queries // 3
            generate_query_feat = []
            for i in range(3):
                enc_outputs_class_unselected = self.cls_embed[i](output_memory)
                topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

                tgt_undetach = torch.gather(output_memory, 1,
                                            topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid
                generate_query_feat.append(tgt_undetach)
            generate_query_feat = torch.cat(generate_query_feat, dim=1).transpose(0, 1) # (num_queries, batch_size, c)
            # cls_pred, mask_pred, attn_mask = self._forward_head(
            #     query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
            # cls_pred_list.append(cls_pred)
            # mask_pred_list.append(mask_pred)
            generate_query_feat = query_feat.detach()

        learn_query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        query_feat = self.query_proj(torch.cat([learn_query_feat, generate_query_feat], dim=-1))

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        self_attn_mask = torch.zeros((batch_size * self.num_heads, self.num_queries, self.num_queries), device=attn_mask.device)
        num_query_per_task = self.num_queries // 3
        for i in range(3):
            self_attn_mask[:, i*num_query_per_task:(i+1)*num_query_per_task, i*num_query_per_task:(i+1)*num_query_per_task] = 1
        self_attn_mask = self_attn_mask < 0.5

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list