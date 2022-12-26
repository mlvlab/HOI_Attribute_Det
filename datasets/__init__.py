# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .hico import build as build_hico
from .vcoco import build as build_vcoco
from .vaw import build as build_vaw
from .cocoatt import build as build_cocoatt
from torch.utils.data.dataset import ConcatDataset

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.mtl:
        data=[]
        if 'vcoco' in args.mtl_data:
            data.append(build_vcoco(image_set, args))
        if 'hico' in args.mtl_data:
            data.append(build_hico(image_set, args))
        if 'vaw' in args.mtl_data:
            data.append(build_vaw(image_set, args))  

        # return ConcatDataset(data)
        return data
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    if args.dataset_file == 'vaw':
        return build_vaw(image_set, args)
    if args.dataset_file == 'cocoatt':
        return build_cocoatt(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
