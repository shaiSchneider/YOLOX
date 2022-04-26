#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import (
    YoloFormatDataset,
    TrainTransform,
    YoloBatchSampler,
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
    worker_init_reset_seed,
)
from yolox.utils import wait_for_the_master, get_local_rank


class Exp(MyExp):

    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.num_classes = 1
        self.max_epoch = 50

        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 100
        # epoch number used for warmup
        self.warmup_epochs = 0


        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 0.0
        # prob of applying mixup aug
        self.mixup_prob = 0.0
        self.enable_mixup = False
        self.perspective = 0.0
        self.data_dir = r"/home/shai/Desktop/lifeguard_data/sanity_check"

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = YoloFormatDataset(
                data_dir=self.data_dir,
                anno_file="train.txt",
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset, mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = YoloFormatDataset(
                data_dir=self.data_dir,
                anno_file="val.txt",
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=0.0,
                    hsv_prob=0.0),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset, mosaic=no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=0.0,
                hsv_prob=0.0),
            degrees=0.0,
            translate=0.0,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=0.0,
            enable_mixup=self.enable_mixup,
            mosaic_prob=0.0,
            mixup_prob=0.0,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        val_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return val_loader

    # def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
    #     from yolox.evaluators import VOCEvaluator

    #     val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
    #     evaluator = VOCEvaluator(
    #         dataloader=val_loader,
    #         img_size=self.test_size,
    #         confthre=self.test_conf,
    #         nmsthre=self.nmsthre,
    #         num_classes=self.num_classes,
    #     )
    #     return evaluator