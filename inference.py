# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict
from typing import Optional
from pathlib import Path
import argparse
import datetime
import cv2
import json
import os
import numpy as np
import random
import sys
import time

from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist

from datasets import build_dataset, voc_eval, draw
from engine import train_one_epoch
from models import build_dab_deformable_detr
from util.logger import setup_logger
from util.utils import clean_state_dict
import datasets
import util.misc as utils


def get_args_parser():
    parser = argparse.ArgumentParser('DN-DETR', add_help=False)

    # about dn args
    parser.add_argument('--use_dn', action="store_true",
                        help="use denoising training.")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int,
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int,
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str,
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'],
                        help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str,
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true',
                        help="Using pre-norm in the Transformer blocks.")
    parser.add_argument('--num_select', default=300, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--random_refpoints_xy', action='store_true',
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR  # sjhong default=4 -> 5
    parser.add_argument('--two_stage', default=False, action='store_true',
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=5, type=int,
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=5, type=int,
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=5, type=int,
                        help="number of deformable attention sampling points in encoder layers")

    # about lr
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='learning rate for backbone')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=3, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_kfl_bbox', default=5, type=float,
                        help="KFL box coefficient in the matching cost")

    # dataset parameters
    parser.add_argument('--num_cls', default=1, type=int,
                        help='Number of object classes in dataset')
    parser.add_argument('--img_size', default=768, type=int,
                        help='Image size')

    # Inference utils
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--model', default='', help='Load model weights from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)

    parser.add_argument('--save_results', action='store_true',
                        help="For eval only. Save the outputs for all images.")

    return parser


def get_args_parser():
    parser = argparse.ArgumentParser('DN-DETR', add_help=False)

    # about dn args
    parser.add_argument('--use_dn', action="store_true",
                        help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int,
                        help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")

    # about lr
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='learning rate for backbone')

    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--drop_lr_now', action="store_true", help="load checkpoint and drop for 12epoch setting")
    parser.add_argument('--save_checkpoint_interval', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int,
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int,
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str,
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'],
                        help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str,
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true',
                        help="Using pre-norm in the Transformer blocks.")
    parser.add_argument('--num_select', default=300, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int,
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true',
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR  # sjhong default=4 -> 5
    parser.add_argument('--two_stage', default=False, action='store_true',
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=5, type=int,
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=5, type=int,
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=5, type=int,
                        help="number of deformable attention sampling points in encoder layers")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=3, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_kfl_bbox', default=5, type=float,
                        help="KFL box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float,
                        help="loss coefficient for cls")
    parser.add_argument('--mask_loss_coef', default=1, type=float,
                        help="loss coefficient for mask")
    parser.add_argument('--dice_loss_coef', default=1, type=float,
                        help="loss coefficient for dice")
    parser.add_argument('--bbox_loss_coef', default=3, type=float,
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--bbox_loss_kfl_coef', default=5, type=float,
                        help="loss coefficient for bbox KFL loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help="alpha for focal loss")

    # dataset parameters
    parser.add_argument('--num_cls', default=1, type=int,
                        help='Number of object classes in dataset')
    parser.add_argument('--img_size', default=768, type=int,
                        help='Image size')

    # Traing utils
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='', help='Load model weights from checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+',
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help="eval only. w/o Training.")
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true',
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true',
                        help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    return parser


def main(args):
    # setup logger
    output_dir = os.path.join(args.output_dir, 'predict')
    os.makedirs(output_dir, exist_ok=True)
    os.environ['output_dir'] = args.output_dir
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = 42 + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, _, postprocessors = build_dab_deformable_detr(args)
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.model, map_location='cpu')['model']
    _tmp_st = OrderedDict({k: v for k, v in clean_state_dict(checkpoint).items()})
    model.load_state_dict(_tmp_st, strict=False)
    del _tmp_st

    dataset_test = build_dataset(image_set='test', img_size=args.img_size)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    print("Start Inference")
    start_time = time.time()
    name = 0
    for samples, _ in data_loader_test:
        with torch.no_grad():
            outputs = postprocessors(model(samples.to(device))[0], target_size=768)   # [{'scores': s, 'labels': l, 'boxes': b}]

        img, _ = samples.decompose()
        img = img.squeeze(0).permute(1, 2, 0).numpy()
        drawn_img = draw(img, outputs, thr=0.1)
        name += 1
        cv2.imwrite(os.path.join(output_dir, str(name)+'.png'), drawn_img)

    total_time = time.time() - start_time
    print("time: {} sec".format(str(round(total_time))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
