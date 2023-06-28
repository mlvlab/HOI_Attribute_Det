# ------------------------------------------------------------------------
# HOTR official code : hotr/data/datasets/hico.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from QPIC (https://github.com/hitachi-rd-cv/qpic)
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np
from skimage.draw import polygon2mask
import torch
import torch.utils.data
import torchvision

# from hotr.data.datasets import builtin_meta
import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy
from .coco import convert_coco_poly_to_mask
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class VAWDetection(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, attribute_index, transforms, num_queries):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        with open(attribute_index,'r') as f:
            self.attribute_names = json.load(f)
            self._valid_att_names= [k for k,v in self.attribute_names.items()]
            self._valid_att_ids = [v for k,v in self.attribute_names.items()]
                
        self._transforms = transforms
        self.num_queries = num_queries
        # self.get_metadata()
        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                            82, 84, 85, 86, 87, 88, 89, 90)
        
    ############################################################################
    # Number Method
    ############################################################################
    # def get_metadata(self):
    #     meta = builtin_meta._get_coco_instances_meta()
    #     self.COCO_CLASSES = meta['coco_classes']
    #     self._valid_obj_ids = [id for id in meta['thing_dataset_id_to_contiguous_id'].keys()]
    #     self._valid_att_names= [k for k,v in self.attribute_names.items()]
    #     self._valid_att_ids = [v for k,v in self.attribute_names.items()]
    

    # def get_valid_obj_ids(self):
    #     return self._valid_obj_ids

    # def get_attributes(self):
    #     return self._valid_att_names

    # def num_category(self):
    #     return len(self.COCO_CLASSES)

    def num_attributes(self):
        return len(self._valid_att_ids)
    ############################################################################

    def convert_bbox(self,bbox): #annotation bbox (c_x,c_y,w,h)-> (x1,y1,x2,y2) for roi align
        x1, y1, w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        x2, y2 = x1+w, y1+h  
        return [x1,y1,x2,y2]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_anno = self.annotations[idx]
        file_dir = img_anno['file_name'].split('/')[-2]+'/'+img_anno['file_name'].split('/')[-1]
        img = Image.open(self.img_folder / file_dir).convert('RGB')
        w, h = img.size

        # guard against no boxes via resizing
        boxes = torch.as_tensor(img_anno['boxes'], dtype=torch.float32).reshape(-1, 4)
        # if self.img_set == 'train':


        # box_list = []
        # for bbox in img_anno['boxes']:
        #     converted_bbox = self.convert_bbox(bbox)
        #     box_list.append(converted_bbox)
        # boxes = torch.tensor(box_list)
        # boxes[:,0::2] = boxes[:,0::2]/w
        # boxes[:,1::2] = boxes[:,1::2]/h


        # else:
        #     boxes = img_anno['boxes']
        # masks = [] 
        # for polygon in img_anno['instance_polygon']:
        #     if polygon is None:
        #         mask = torch.ones((h, w)).unsqueeze(0)
        #     else:
        #         mask = convert_coco_poly_to_mask(np.array(polygon), h, w)
        #     masks.append(mask)
        # masks = torch.cat(masks)

        mask_list = []
        for polygon in img_anno['instance_polygon']:
            if polygon is not None:
                mask = torch.from_numpy(polygon2mask((w,h),polygon[0]).transpose()).unsqueeze(0)
                mask_list.append(mask)
            else:
                mask = torch.ones((h, w)).unsqueeze(0)
                mask_list.append(mask)
        masks = torch.cat(mask_list)

        polygons = img_anno['instance_polygon']
        img_id = str(img_anno['image_id'])
        # for polygon in polygons:
        #     if polygon is None:
        #         torch.ones

        object_names = img_anno['object_name']
        #clip_embeds = torch.tensor(img_anno['clip_em'])

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            obj_classes = [(i, self._valid_obj_ids.index(obj)) for i, obj in enumerate(img_anno['category_id'])]
            # pos_att_classes = [(i, pos_att) for i, pos_att in enumerate(img_anno['pos_att_id'])]
            # neg_att_classes = [(i, neg_att) for i, neg_att in enumerate(img_anno['neg_att_id'])]
        else:
            obj_classes = [self._valid_obj_ids.index(obj) for obj in img_anno['category_id']]
            # pos_att_classes = [(i, pos_att) for i, pos_att in enumerate(img_anno['pos_att_id'])]
        obj_classes = torch.tensor(obj_classes, dtype=torch.int64)
        # pos_att_classes = torch.tensor(pos_att_classes, dtype=torch.int64)
        # neg_att_classes = torch.tensor(neg_att_classes, dtype=torch.int64) if self.img_set =='train' else None
        num_boxes = len(obj_classes)
        num_attributes = self.num_attributes()
        pos_att_classes = torch.zeros((num_boxes,num_attributes),dtype=torch.float32)
        for b, pos_id in zip(pos_att_classes,img_anno['pos_att_id']):
            b[pos_id]=1
        
        if self.img_set == 'train':
            neg_att_classes = torch.zeros((num_boxes,num_attributes),dtype=torch.float32)
            for b, neg_id in zip(neg_att_classes,img_anno['neg_att_id']):
                b[neg_id]=1

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        
        boxes[:,2:]+=boxes[:,:2]

        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            obj_classes = obj_classes[keep]
            pos_att_classes = pos_att_classes[keep]
            neg_att_classes = neg_att_classes[keep]
            target['masks'] = masks[keep]
            #target['polygons'] =  polygons
            target['object_name'] = object_names
            #target['clip_em'] = clip_embeds[keep]
            target['labels'] = obj_classes
            target['pos_att_classes'] = pos_att_classes
            target['neg_att_classes'] = neg_att_classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            target['type'] = 'att'
            target['img_id'] = img_id
            target['boxes'] = boxes
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            #target['masks'] =  masks[keep]
            #target['boxes'] = boxes
            #target['clip_em'] = clip_embeds[keep]
            target['labels'] = target['labels'][:, 1]
            # target['pos_att_classes'] = target['pos_att_classes'][:, 1]
            target['dataset'] = 'vaw'

        else:
            target['masks'] =  masks
            target['polygons'] =  polygons
            # target['object_name'] = object_names
            #target['clip_em'] = clip_embeds
            target['labels'] = obj_classes
            target['pos_att_classes'] = pos_att_classes
            target['id'] = idx
            target['type'] = 'att'
            target['dataset'] = 'vaw'
            target['img_id'] = img_id

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            target['boxes'] = boxes

        return img, target

    def set_rare_attrs(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        self.rare_triplets = []
        self.non_rare_triplets = []
        for i in annotations:
            if i['instance_count'] > 0:
                if i['instance_count'] < 10:
                    self.rare_triplets.append((i['id']))
                else:
                    self.non_rare_triplets.append((i['id']))
    
    def set_class_imb_attrs(self,anno_file):
        with open('data/vaw/annotations/head_tail.json','r') as f:
            head_tail = json.load(f)
        with open(anno_file,'r') as f:

            annotations= json.load(f)
        self.head,self.medium,self.tail = [],[],[]
        for i in head_tail['head']:
            self.head.append(annotations[i])
        for i in head_tail['medium']:
            self.medium.append(annotations[i])
        for i in head_tail['tail']:
            self.tail.append(annotations[i])
    
    def _valid_att_idxs(self,anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)
        num_attributes = self.num_attributes()
        self.valid_masks = np.zeros(num_attributes)

        for i in annotations:
            if i['instance_count'] > 0:
                self.valid_masks[i['id']]=1

# Add color jitter to coco transforms
def make_vaw_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            # T.RandomSelect(
            # T.RandomResize(scales, max_size=1333)
            #     T.Compose([
            #         T.RandomResize([400, 500, 600]),
            #         T.RandomSizeCrop(384, 600),
            #         T.RandomResize(scales, max_size=1333),
            #     ])
            # )
            T.RandomResize(scales, max_size=1333)
            ,
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    # root = Path(args.data_path)
    root = Path('data/vaw')
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' , root / 'annotations' / 'vaw_train.json'),
        'val': (root / 'images' , root / 'annotations' / 'vaw_test.json'),
        'test': (root / 'images' , root / 'annotations' / 'vaw_test.json')
    }

    # CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'
    # attribute_freq = root / 'annotations' / 'vaw_orig_train_cat_info.json'
    attribute_index = root / 'annotations' / 'attribute_index.json'    

    img_folder, anno_file = PATHS[image_set]
    dataset = VAWDetection(image_set, img_folder, anno_file,attribute_index, transforms=make_vaw_transforms(image_set),
                            num_queries=args.num_queries)
    #if image_set == 'val' or image_set == 'test':
            
        #dataset.set_rare_attrs(attribute_freq)
        #dataset._valid_att_idxs(attribute_freq)
        #dataset.set_class_imb_attrs(attribute_index)
        
    return dataset

