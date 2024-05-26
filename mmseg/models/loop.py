# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals
from mmengine.runner.loops import _InfiniteDataloaderIterator


@LOOPS.register_module()
class MultiDatasetIterBasedTrainLoop(BaseLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: List,
            max_iters: int,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        self._runner = runner
        self.dataloader = []
        diff_rank_seed = runner._randomness_cfg.get(
            'diff_rank_seed', False)
        for dl in dataloader:
            self.dataloader.append(runner.build_dataloader(
                dl, seed=runner.seed, diff_rank_seed=diff_rank_seed))

        self._max_iters = int(max_iters)
        assert self._max_iters == max_iters, \
            f'`max_iters` should be a integer number, but get {max_iters}'
        self._max_epochs = 1  # for compatibility with EpochBasedTrainLoop
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # get the iterator of the dataloader
        self.dataloader_iterator = []
        for dl in self.dataloader:
            self.dataloader_iterator.append(_InfiniteDataloaderIterator(dl))

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        while self._iter < self._max_iters:
            self.runner.model.train()

            data_batch = defaultdict(list)
            for dlit in self.dataloader_iterator:
                db = next(dlit)
                for key in db:
                    data_batch[key].extend(db[key])
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self._iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


@LOOPS.register_module()
class MultiDatasetValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: List,
                 evaluator: List,
                 fp16: bool = False) -> None:
        self._runner = runner
        self.dataloader = []
        diff_rank_seed = runner._randomness_cfg.get(
            'diff_rank_seed', False)
        for dl in dataloader:
            self.dataloader.append(runner.build_dataloader(
                dl, seed=runner.seed, diff_rank_seed=diff_rank_seed))
        self.evaluator = []
        for eva in evaluator:
            self.evaluator.append(runner.build_evaluator(eva))
        for i, dl in enumerate(self.dataloader):
            self.evaluator[i].dataset_meta = dl.dataset.metainfo
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        total_metrics = dict()
        for i, dl in enumerate(self.dataloader):
            for idx, data_batch in enumerate(dl):
                self.run_iter(idx, data_batch, i)

            # compute metrics
            metrics = self.evaluator[i].evaluate(len(dl.dataset))
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
                else:
                    total_metrics[k] = v
        self.runner.call_hook('after_val_epoch', metrics=total_metrics)
        self.runner.call_hook('after_val')
        return total_metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], dataset_idx):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator[dataset_idx].process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
