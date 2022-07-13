# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
from typing import Iterable
import math
import os
import sys

import torch

from datasets.coco_eval import CocoEvaluator
from util.utils import slprint, to_device
import util.misc as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, lr_scheduler,
                    max_norm: float = 0, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()

    cur_iteration = 0
    epoch_size = len(data_loader)
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, mask_dict = model(samples, dn_args=(targets, args.scalar, args.label_noise_scale,
                                                             args.box_noise_scale, args.num_patterns))
                loss_dict = criterion(outputs, targets, mask_dict)
            else:
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        total_iter = epoch * epoch_size + cur_iteration
        lr_scheduler.step(total_iter)
        cur_iteration += 1

        if logger:
            # Record log per iteration
            logger.write_logger(iteration=total_iter,
                                lr=optimizer.param_groups[0]["lr"],
                                loss=loss_value,
                                **loss_dict_reduced_scaled,
                                **loss_dict_reduced_unscaled,
                                class_error=loss_dict_reduced['class_error'])

    return {'lr': optimizer.param_groups[0]["lr"],
            'loss': loss_value,
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
            'class_error': loss_dict_reduced['class_error']}
