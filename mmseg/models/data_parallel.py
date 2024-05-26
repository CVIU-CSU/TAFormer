# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Union

import torch

from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS

from mmengine.utils.dl_utils.parrots_wrapper import TORCH_VERSION
from mmengine.utils.version_utils import digit_version
from mmengine.model import MMDistributedDataParallel
from mmengine.logging import MMLogger, print_log

@MODEL_WRAPPERS.register_module()
class ModelMMDistributedDataParallel(MMDistributedDataParallel):

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
          call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        # with optim_wrapper.optim_context(self):
        log_vars = {}
        data = self.module.data_preprocessor(data, training=True)

        losses = self._run_forward(data, mode='loss')
        gt_loss, gt_log_vars = self.module.parse_losses(losses)
        log_vars.update(gt_log_vars)
        gt_loss.backward()

        semi_losses = self._run_forward(data, mode='semi_loss')
        semi_loss, semi_log_vars = self.module.parse_losses(semi_losses)
        log_vars.update(semi_log_vars)
        semi_loss.backward()

        # logger: MMLogger = MMLogger.get_current_instance()
        # print_log(self.module.get_model().decode_head.mask_embed[1][4].weight.grad[0, 0], logger)
        # optim_wrapper.update_params()
        optim_wrapper._inner_count += 1
        if optim_wrapper.should_update():
            optim_wrapper.step()
            optim_wrapper.zero_grad()
        if self.detect_anomalous_params:
            detect_anomalous_params(parsed_loss, model=self)
        return log_vars
