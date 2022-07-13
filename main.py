# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pathlib import Path
from typing import Optional
import argparse
import datetime
import json
import numpy as np
import random
import os
import time
import sys

from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist

from datasets import build_dataset, voc_eval
from engine import train_one_epoch
from models import build_dab_deformable_detr
from util.logger import setup_logger
from util.lr_scheduler import CosineAnnealingWarmUpRestarts
from util.misc import Logger
from util.utils import clean_state_dict
import datasets
import util.misc as utils


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
    parser.add_argument('--lr', default=1e-5, type=float,
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
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
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
    utils.init_distributed_mode(args)
    torch.autograd.set_detect_anomaly(True)

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['output_dir'] = args.output_dir
    tb_logger = Logger(log_path=args.output_dir)
    command_logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'),
                                  distributed_rank=args.rank,
                                  color=False,
                                  name="DN-DETR")
    command_logger.info("Command: " + ' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        command_logger.info("Full config saved to {}".format(save_json_path))
    command_logger.info('world size: {}'.format(args.world_size))
    command_logger.info('rank: {}'.format(args.rank))
    command_logger.info('local_rank: {}'.format(args.local_rank))
    command_logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_dab_deformable_detr(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    dataset_train = build_dataset(image_set='train', img_size=args.img_size)
    dataset_test = build_dataset(image_set='test', img_size=args.img_size)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer,
                                                 T_0=len(dataset_train),
                                                 T_mult=2,
                                                 eta_max=0.001,
                                                 T_warmup=len(dataset_train),
                                                 gamma=0.9
                                                 )

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

            if args.drop_lr_now:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

    if not args.resume and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        command_logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict(
            {k: v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        del _tmp_st
        command_logger.info(str(_load_output))

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()

        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_status = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                       lr_scheduler=lr_scheduler, max_norm=args.clip_max_norm, args=args)

        tb_logger.write_logger(epoch=epoch,
                               **train_status)

        epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - epoch_start_time)))
        print('Epoch[{}]: {}'.format(epoch, epoch_time_str))

        if epoch > 100 and epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint{epoch:04}.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            test_stats = voc_eval(model,
                                  postprocessors,
                                  data_loader_test,
                                  device,
                                  args.num_patterns,
                                  args.img_size,
                                  iou_thr=0.3)
            print('Eval: ', test_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Now time: {}".format(str(datetime.datetime.now())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
