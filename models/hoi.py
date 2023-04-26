# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from scipy.optimize import linear_sum_assignment
from skimage.draw import polygon2mask
from pycocotools import mask as coco_mask

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from util.fed import load_class_freq, get_fed_loss_inds                       
from .layers.roi_align import ROIAlign
from typing import List
import numpy as np
import cv2

from datasets.coco import convert_coco_poly_to_mask

class DETRHOI(nn.Module):

    def __init__(self, backbone, transformer, num_obj_classes, num_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.mtl = args.mtl
        if args.mtl:
            if args.freeze_hoi:
                self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)                
                self.vcoco_verb_class_embed = nn.Linear(hidden_dim, num_classes['vcoco'])
                self.hico_verb_class_embed = nn.Linear(hidden_dim, num_classes['hico'])
                self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
                if args.br:
                    if args.fc_version:
                        self.attribute_fc1 = nn.Linear(backbone.num_channels, backbone.num_channels)
                        self.attribute_fc2 = nn.Linear(backbone.num_channels, backbone.num_channels)
                        self.attribute_class_embed = nn.Linear(backbone.num_channels, num_classes['att'])
                    else:
                        self.attribute_input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)                
                        self.attribute_class_embed = nn.Linear(hidden_dim, num_classes['att'])
                        self.attribute_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
                else:
                    self.attribute_input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)                
                    self.attribute_class_embed = nn.Linear(hidden_dim, num_classes['att'])
                    self.attribute_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

                self.attribute_roi_align = ROIAlign(output_size=(2,2), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
                self.attribute_avgpool = nn.AdaptiveAvgPool2d((1, 1))


            else:
                if 'hico' in args.mtl_data or 'vcoco' in args.mtl_data:
                    self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)                
                    if 'vcoco' in args.mtl_data:
                        self.vcoco_verb_class_embed = nn.Linear(hidden_dim, num_classes['vcoco'])
                    if 'hico' in args.mtl_data:
                        self.hico_verb_class_embed = nn.Linear(hidden_dim, num_classes['hico'])

                    self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

                if 'vaw' in args.mtl_data:

                    if args.br:
                        self.attribute_input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
                    self.attribute_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
                    self.attribute_roi_align = ROIAlign(output_size=(7,7), spatial_scale=1.0, sampling_ratio=0, aligned=True)
                    self.attribute_class_embed = nn.Linear(hidden_dim, num_classes['att'])
                    self.attribute_avgpool = nn.AdaptiveAvgPool2d((1, 1))


            
        else:
            if args.hoi:
                self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
                self.verb_class_embed = nn.Linear(hidden_dim, num_classes['hoi'])

            elif args.att_det:
                self.att_class_embed = nn.Linear(hidden_dim, num_classes['att'])
            self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)

        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.show_vid = args.show_vid
        if args.object_embedding:
            self.obj_em_relu = F.relu
            self.obj_em_w1 = nn.Linear(1024,256)
            #self.obj_em_w2 = nn.Linear(512,256)
            

        if args.predict_mask:
            n_class = 1 #binary classifier
            self.relu = F.relu
            self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn1     = nn.BatchNorm2d(1024)
            self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn2     = nn.BatchNorm2d(512)
            self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn3     = nn.BatchNorm2d(256)
            # self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            # self.bn4     = nn.BatchNorm2d(128)
            # self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            # self.bn5     = nn.BatchNorm2d(64)
            # self.deconv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            # self.bn6     = nn.BatchNorm2d(32)
            self.mask_classifier = nn.Conv2d(256, n_class, kernel_size=1)

    def forward(self, samples: NestedTensor, targets=None, dtype: str='', dataset:str='',args=None, eval=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples) #args.masks True for mask prediction
        src, mask = features[-1].decompose()        
        assert mask is not None

        #feature shape excluding padding size (H,W)
        masking_shape = [] 
        for img_mask in mask:
            max_h, max_w = np.where(img_mask.cpu().numpy() == False)[0].max(), np.where(img_mask.cpu().numpy() == False)[1].max()
            masking_shape.append(img_mask[:max_h,:max_w].shape)

        if dtype=='att':           
            if not self.training:
                #for video inference
                if self.show_vid: 
                    # import pdb;pdb.set_trace()
                    # box_tensors = torch.Tensor([int(0)] + targets.tolist()).unsqueeze(0) # [1,5] : 1 frame 
                    box_tensors = torch.cat([torch.zeros((len(targets),1)),targets],dim=-1)
                    encoder_src = self.input_proj(src)
                    B,C,H,W = encoder_src.shape 
                    encoder_src = encoder_src.flatten(2).permute(2, 0, 1)
                    pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
                    mask = mask.flatten(1)
                    memory = self.transformer.encoder(encoder_src, src_key_padding_mask=mask, pos=pos_embed)
                    encoder_output = memory.permute(1, 2, 0) 
                    encoder_output = encoder_output.view([B,C,H,W]) 
                    feature_H, feature_W = encoder_output.shape[2], encoder_output.shape[3]
                
                    box_tensors[...,1], box_tensors[...,3] = feature_W*box_tensors[...,1]/samples.tensors.shape[-1], feature_W*box_tensors[...,3]/samples.tensors.shape[-1] 
                    box_tensors[...,2], box_tensors[...,4] = feature_H*box_tensors[...,2]/samples.tensors.shape[-2], feature_H*box_tensors[...,4]/samples.tensors.shape[-2]
                                        
                    pooled_feature = self.attribute_roi_align(input = encoder_output, rois = box_tensors.cuda()) 
                    x = self.attribute_conv(pooled_feature) 
                    x = self.attribute_avgpool(x) 
                    x = torch.flatten(x, 1)
                    outputs_class = self.attribute_class_embed(x)
                    # import pdb;pdb.set_trace()
                    # out = {'pred_logits':outputs_class,'type': dtype,'dataset':dataset}
                    out = outputs_class.sigmoid()
                    return out
            
                object_boxes = torch.cat([torch.stack([target['boxes'][...,0]/target['orig_size'][1],target['boxes'][...,1]/target['orig_size'][0],target['boxes'][...,2]/target['orig_size'][1],target['boxes'][...,3]/target['orig_size'][0]],axis=1) for target in targets])
                batch_index = torch.cat([torch.Tensor([int(i)]) for i, target in enumerate(targets) for _ in target['boxes']])
                box_tensors = torch.cat([batch_index.unsqueeze(1).cuda(),object_boxes.cuda()], axis=1) #[K,5]

                encoder_src = self.input_proj(src)
                B,C,H,W = encoder_src.shape 
                encoder_src = encoder_src.flatten(2).permute(2, 0, 1)
                pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
                mask = mask.flatten(1)
                memory = self.transformer.encoder(encoder_src, src_key_padding_mask=mask, pos=pos_embed)
                encoder_output = memory.permute(1, 2, 0) 
                encoder_output = encoder_output.view([B,C,H,W]) 
                feature_H, feature_W = encoder_output.shape[2], encoder_output.shape[3]

                encoder_output = F.relu(encoder_output)

                #normalize box for feature size (exclude padding size) 
                for box_tensor in box_tensors: 
                    h_size, w_size = masking_shape[int(box_tensor[0].item())][0], masking_shape[int(box_tensor[0].item())][1]
                    box_tensor[1], box_tensor[3] = w_size*box_tensor[1], w_size*box_tensor[3] 
                    box_tensor[2], box_tensor[4] = h_size*box_tensor[2], h_size*box_tensor[4] 

                #img level binary mask -> normalize to feature size(excluding padding) -> add padding to normalized mask
                if args.input_masking:
                    binary_masks = []
                    for target in targets:
                        for mask in target['masks']:
                            binary_masks.append(mask)

                    assert len(binary_masks) == len(object_boxes)

                    pooled_features = []
                    feature_H, feature_W = encoder_output.shape[2], encoder_output.shape[3]
                    for mask, object_box in zip(binary_masks,object_boxes):
                        batch_index = int(object_box[0].item())
                        tmp_tensor = [torch.tensor(int(0)).unsqueeze(0).cuda()]
                        tmp_tensor.extend([box_tensor[1:]])
                        tmp_tensor = torch.cat(tmp_tensor).unsqueeze(0)
                        if mask is not None and mask.sum() != 0: 
                            size_H, size_W = masking_shape[batch_index][0], masking_shape[batch_index][1] 
                            resized_mask = F.interpolate(torch.tensor(mask).float().unsqueeze(0).unsqueeze(0), size=(size_H, size_W), mode='bilinear')
                            resized_mask = resized_mask.squeeze(0).squeeze(0)
                            padded_mask = torch.from_numpy(np.pad(resized_mask, ((0,feature_H-resized_mask.shape[0]),(0,feature_W-resized_mask.shape[1])), 'constant', constant_values=False)).cuda()
                            masked_feature = (encoder_output[batch_index]*padded_mask).unsqueeze(0)
                            pooled_feature = self.attribute_roi_align(input = masked_feature, rois = tmp_tensor.cuda())
                            if pooled_feature.sum() == 0:
                                orig_pooled_feature = self.attribute_roi_align(input = encoder_output[batch_index].unsqueeze(0), rois = tmp_tensor.cuda())
                                pooled_features.append(orig_pooled_feature)
                            else:
                                pooled_features.append(pooled_feature)

                        #binary_mask is None
                        else: 
                            pooled_feature = self.attribute_roi_align(input = encoder_output[batch_index].unsqueeze(0), rois = tmp_tensor.cuda())
                            pooled_features.append(pooled_feature)

                    x = self.attribute_conv(torch.cat(pooled_features)) 
                    x = self.attribute_avgpool(x) 
                    x = torch.flatten(x, 1)
                    outputs_class = self.attribute_class_embed(x)
                        
                elif args.output_masking: #roi aligned feature masking          
                    binary_masks = []
                    for target in targets:
                        for mask in target['masks']:
                            binary_masks.append(mask)
                    
                    assert len(binary_masks) == len(object_boxes)
                    output_masks = []
                    for mask, object_box in zip(binary_masks,object_boxes):
                        if mask is not None and mask.sum() != 0: 
                            orig_size = mask.shape
                            h,w = orig_size[0], orig_size[1]
                            unnorm_x1, unnorm_x2 = w*object_box[0], w*object_box[2]  #object_box (batch_idx,x1,y1,x2,y2)
                            unnorm_y1, unnorm_y2 = h*object_box[1], h*object_box[3]
                            unnorm_box = (unnorm_x1, unnorm_y1, unnorm_x2, unnorm_y2)
                            cropped_mask_tensor = self.crop2box(unnorm_box, mask)    
                            if cropped_mask_tensor.sum() == 0: 
                                import pdb; pdb.set_trace()
                                output_mask = torch.ones_like(torch.empty(7,7)).float().unsqueeze(0).unsqueeze(0)
                                output_masks.append(output_mask)
                            else:                        
                                output_mask = F.interpolate(torch.tensor(cropped_mask_tensor).float().unsqueeze(0).unsqueeze(0), size=(7, 7), mode='bilinear')
                                output_masks.append(output_mask)
                        
                        else: #binary_mask is None
                            output_mask = torch.ones_like(torch.empty(7,7)).float().unsqueeze(0).unsqueeze(0)
                            output_masks.append(output_mask) 

                    pooled_feature = self.attribute_roi_align(input = encoder_output, rois = box_tensors.cuda()) 
                    x = self.attribute_conv(pooled_feature*torch.cat(output_masks).cuda()) 
                    x = self.attribute_avgpool(x) 
                    x = torch.flatten(x, 1)
                    outputs_class = self.attribute_class_embed(x)                           

                #NO MASK EVALUATION
                else:
                    pooled_feature = self.attribute_roi_align(input = encoder_output, rois = box_tensors.cuda()) 
                    if args.object_embedding:
                        import pdb; pdb.set_trace()
                        object_embedding = [target['clip_em'] for target in targets]
                        gated_feature = self.gating_function(pooled_feature, object_embedding)
                        x = self.attribute_conv(gated_feature) 
                        x = self.attribute_avgpool(x) 
                        x = torch.flatten(x, 1)
                        outputs_class = self.attribute_class_embed(x)

                    elif args.predict_mask:
                        x_256 = features[0].tensors #torch.Size([8, 256, 253, 276])
                        x_512 = features[1].tensors #torch.Size([8, 512, 127, 138])
                        x_1024 = features[2].tensors #torch.Size([8, 1024, 64, 69])
                        x_2048 = features[3].tensors #torch.Size([8, 2048, 32, 35])
                        x = self.relu(self.deconv1(x_2048)) #torch.Size([8, 1024, 64, 70])               
                        if x_1024.shape[2:] != x.shape[2:]:
                            H = x.shape[2:][0] - x_1024.shape[2:][0]
                            W = x.shape[2:][1] - x_1024.shape[2:][1]
                            x_1024 = F.pad(x_1024,(0,W,0,H),"constant",0)
                        x = self.bn1(x_1024 + x) #torch.Size([8, 1024, 64, 70])                    
                        x = self.relu(self.deconv2(x)) #torch.Size([8, 512, 128, 140])
                        if x_512.shape[2:] != x.shape[2:]:
                            H = x.shape[2:][0] - x_512.shape[2:][0]
                            W = x.shape[2:][1] - x_512.shape[2:][1]
                            x_512 = F.pad(x_512,(0,W,0,H),"constant",0)
                        x = self.bn2(x_512 + x) #torch.Size([8, 512, 128, 140])            
                        x = self.relu(self.deconv3(x)) #torch.Size([8, 256, 256, 280])           
                        if x_256.shape[2:] != x.shape[2:]:
                            H = x.shape[2:][0] - x_256.shape[2:][0]
                            W = x.shape[2:][1] - x_256.shape[2:][1]
                            x_256 = F.pad(x_256,(0,W,0,H),"constant",0)
                        x = self.bn3(x_256 + x) #torch.Size([8, 256, 256, 280])                    
                        mask_prediction = self.mask_classifier(x) 

                        mask_preds = []
                        #gt_masks = []
                        for i,target in enumerate(targets):
                            batch_index = i
                            mask_pred = mask_prediction[i].repeat((len(target['masks']),1,1))
                            mask_preds.append(mask_pred)

                        mask_preds = torch.cat(mask_preds).sigmoid()

                        #attribute prediciton                    
                        pooled_feature = self.attribute_roi_align(input = encoder_output, rois = box_tensors.cuda()) 
                        resized_mask = F.interpolate(mask_preds.unsqueeze(1), size=(pooled_feature.shape[-2], pooled_feature.shape[-1]), mode='bilinear')
                        x = self.attribute_conv(pooled_feature*resized_mask) 
                        x = self.attribute_avgpool(x) 
                        x = torch.flatten(x, 1)
                        outputs_class = self.attribute_class_embed(x)


                    else:
                        x = self.attribute_conv(pooled_feature) 
                        x = self.attribute_avgpool(x) 
                        x = torch.flatten(x, 1)
                        outputs_class = self.attribute_class_embed(x)
            
            #training
            else:
                object_boxes = [torch.Tensor([int(i)]+self.convert_bbox(box.tolist())) for i, target in enumerate(targets) for box in target['boxes']]
                box_tensors = torch.stack(object_boxes,0) #[K,5] , K: box annotation length in mini-batch                
                encoder_src = self.input_proj(src)
                B,C,H,W = encoder_src.shape
                encoder_src = encoder_src.flatten(2).permute(2, 0, 1)
                pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
                mask = mask.flatten(1)
                memory = self.transformer.encoder(encoder_src, src_key_padding_mask=mask, pos=pos_embed)
                encoder_output = memory.permute(1, 2, 0) 
                encoder_output = encoder_output.view([B,C,H,W]) 
                encoder_output = F.relu(encoder_output)

                for box_tensor in box_tensors:
                    batch_idx = int(box_tensor[0])
                    non_padded_shape = masking_shape[batch_idx] #(H,W)
                    box_tensor[1], box_tensor[3] = non_padded_shape[1]*box_tensor[1], non_padded_shape[1]*box_tensor[3] 
                    box_tensor[2], box_tensor[4] = non_padded_shape[0]*box_tensor[2], non_padded_shape[0]*box_tensor[4] 

                if args.predict_mask: #args.masking = True로
                    x_256 = features[0].tensors #torch.Size([8, 256, 253, 276])
                    x_512 = features[1].tensors #torch.Size([8, 512, 127, 138])
                    x_1024 = features[2].tensors #torch.Size([8, 1024, 64, 69])
                    x_2048 = features[3].tensors #torch.Size([8, 2048, 32, 35])
                    x = self.relu(self.deconv1(x_2048)) #torch.Size([8, 1024, 64, 70])               
                    if x_1024.shape[2:] != x.shape[2:]:
                        H = x.shape[2:][0] - x_1024.shape[2:][0]
                        W = x.shape[2:][1] - x_1024.shape[2:][1]
                        x_1024 = F.pad(x_1024,(0,W,0,H),"constant",0)
                    x = self.bn1(x_1024 + x) #torch.Size([8, 1024, 64, 70])                    
                    x = self.relu(self.deconv2(x)) #torch.Size([8, 512, 128, 140])
                    if x_512.shape[2:] != x.shape[2:]:
                        H = x.shape[2:][0] - x_512.shape[2:][0]
                        W = x.shape[2:][1] - x_512.shape[2:][1]
                        x_512 = F.pad(x_512,(0,W,0,H),"constant",0)
                    x = self.bn2(x_512 + x) #torch.Size([8, 512, 128, 140])            
                    x = self.relu(self.deconv3(x)) #torch.Size([8, 256, 256, 280])           
                    if x_256.shape[2:] != x.shape[2:]:
                        H = x.shape[2:][0] - x_256.shape[2:][0]
                        W = x.shape[2:][1] - x_256.shape[2:][1]
                        x_256 = F.pad(x_256,(0,W,0,H),"constant",0)
                    x = self.bn3(x_256 + x) #torch.Size([8, 256, 256, 280])                    
                    mask_prediction = self.mask_classifier(x) 

                    mask_preds = []
                    #gt_masks = []
                    for i,target in enumerate(targets):
                        batch_index = i
                        mask_pred = mask_prediction[i].repeat((len(target['masks']),1,1))
                        mask_preds.append(mask_pred)

                    mask_preds = torch.cat(mask_preds).sigmoid()

                    #attribute prediciton                    
                    pooled_feature = self.attribute_roi_align(input = encoder_output, rois = box_tensors.cuda()) 
                    resized_mask = F.interpolate(mask_preds.unsqueeze(1), size=(pooled_feature.shape[-2], pooled_feature.shape[-1]), mode='bilinear')
                    x = self.attribute_conv(pooled_feature*resized_mask) 
                    x = self.attribute_avgpool(x) 
                    x = torch.flatten(x, 1)
                    outputs_class = self.attribute_class_embed(x)
                    out = {'mask_pred_logits': mask_preds,'pred_logits':outputs_class,'type': dtype,'dataset':dataset}
                    return out

                elif args.input_masking:
                    binary_masks = []
                    for target in targets:
                        for mask in target['masks']:
                            binary_masks.append(mask)

                    assert len(binary_masks) == len(object_boxes)

                    pooled_features = []
                    feature_H, feature_W = encoder_output.shape[2], encoder_output.shape[3]
                    for mask, object_box in zip(binary_masks,object_boxes):
                        batch_index = int(object_box[0].item())
                        tmp_tensor = [torch.tensor(int(0)).unsqueeze(0)]
                        tmp_tensor.extend([box_tensor[1:]])
                        tmp_tensor = torch.cat(tmp_tensor).unsqueeze(0).cuda()
                        if mask is not None and mask.sum() != 0: 
                            size_H, size_W = masking_shape[batch_index][0], masking_shape[batch_index][1] 
                            resized_mask = F.interpolate(torch.tensor(mask).float().unsqueeze(0).unsqueeze(0), size=(size_H, size_W), mode='bilinear')
                            resized_mask = resized_mask.squeeze(0).squeeze(0).cpu().numpy()
                            padded_mask = torch.from_numpy(np.pad(resized_mask, ((0,feature_H-resized_mask.shape[0]),(0,feature_W-resized_mask.shape[1])), 'constant', constant_values=False)).cuda()
                            masked_feature = (encoder_output[batch_index]*padded_mask).unsqueeze(0)
                            pooled_feature = self.attribute_roi_align(input = masked_feature, rois = tmp_tensor.cuda())
                            if pooled_feature.sum() == 0:
                                orig_pooled_feature = self.attribute_roi_align(input = encoder_output[batch_index].unsqueeze(0), rois = tmp_tensor.cuda())
                                pooled_features.append(orig_pooled_feature)
                            else:
                                pooled_features.append(pooled_feature)

                        #binary_mask is None
                        else: 
                            pooled_feature = self.attribute_roi_align(input = encoder_output[batch_index].unsqueeze(0), rois = tmp_tensor.cuda())
                            pooled_features.append(pooled_feature)

                    x = self.attribute_conv(torch.cat(pooled_features)) 
                    x = self.attribute_avgpool(x) 
                    x = torch.flatten(x, 1)
                    outputs_class = self.attribute_class_embed(x)
                        
                elif args.output_masking:
                    binary_masks = []
                    for target in targets:
                        for mask in target['masks']:
                            binary_masks.append(mask)
                    
                    assert len(binary_masks) == len(object_boxes)

                    output_masks = []
                    for mask, object_box in zip(binary_masks,object_boxes):
                        if mask is not None and mask.sum() != 0: 
                            orig_size = mask.shape
                            h,w = orig_size[0], orig_size[1]
                            unnorm_x1, unnorm_x2 = w*object_box[1], w*object_box[3]  #object_box (batch_idx,x1,y1,x2,y2)
                            unnorm_y1, unnorm_y2 = h*object_box[2], h*object_box[4]
                            unnorm_box = (unnorm_x1, unnorm_y1, unnorm_x2, unnorm_y2)
                            cropped_mask_tensor = self.crop2box(unnorm_box, mask)    
                            if cropped_mask_tensor.sum() == 0: 
                                output_mask = torch.ones_like(torch.empty(7,7)).float().unsqueeze(0).unsqueeze(0).cuda()
                                output_masks.append(output_mask)
                            else:                        
                                output_mask = F.interpolate(torch.tensor(cropped_mask_tensor).float().unsqueeze(0).unsqueeze(0), size=(7, 7), mode='bilinear',align_corners=False)
                                output_masks.append(output_mask)
                        
                        else: #binary_mask is None
                            output_mask = torch.ones_like(torch.empty(7,7)).float().unsqueeze(0).unsqueeze(0).cuda()
                            output_masks.append(output_mask) 
                    pooled_feature = self.attribute_roi_align(input = encoder_output, rois = box_tensors.cuda())*torch.cat(output_masks).cuda()
                    if args.object_embedding:
                        object_embedding = [target['clip_em'] for target in targets]
                        gated_feature = self.gating_function(pooled_feature, object_embedding)
                        x = self.attribute_conv(gated_feature) 
                        x = self.attribute_avgpool(x) 
                        x = torch.flatten(x, 1)
                        outputs_class = self.attribute_class_embed(x)
                    else:
                        x = self.attribute_conv(pooled_feature) 
                        x = self.attribute_avgpool(x) 
                        x = torch.flatten(x, 1)
                        outputs_class = self.attribute_class_embed(x)

                #no mask training
                else:
                    pooled_feature = self.attribute_roi_align(input = encoder_output, rois = box_tensors.cuda()) 
                    if args.object_embedding:
                        object_embedding = [target['clip_em'] for target in targets]
                        gated_feature = self.gating_function(pooled_feature, object_embedding)
                        x = self.attribute_conv(gated_feature) 
                        x = self.attribute_avgpool(x) 
                        x = torch.flatten(x, 1)
                        outputs_class = self.attribute_class_embed(x)
                    else:
                        x = self.attribute_conv(pooled_feature) 
                        x = self.attribute_avgpool(x) 
                        x = torch.flatten(x, 1)
                        outputs_class = self.attribute_class_embed(x)

            out = {'pred_logits': outputs_class,'type': dtype,'dataset':dataset}

        elif dtype=='hoi':

            encoder_src = self.input_proj(src)
            hs = self.transformer(encoder_src, mask, self.query_embed.weight, pos[-1])[0]

            if dataset == 'hico':
                outputs_class = self.hico_verb_class_embed(hs)
            elif dataset =='vcoco':
                #import pdb; pdb.set_trace()
                outputs_class = self.vcoco_verb_class_embed(hs)            
            # outputs_class = self.verb_class_embed(hs)            
            # else:
                
            #     outputs_class = self.verb_class_embed(hs)            
            outputs_obj_class = self.obj_class_embed(hs)
            
            outputs_obj_coord = self.obj_bbox_embed(hs).sigmoid()
            out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_logits': outputs_class[-1],
                    'pred_obj_boxes': outputs_obj_coord[-1],'type': dtype,'dataset':dataset}

        if dtype=='hoi':
            outputs_sub_coord = self.sub_bbox_embed(hs).sigmoid()
            out.update({'pred_sub_boxes': outputs_sub_coord[-1]})

        if self.aux_loss:
            if dtype=='hoi':
                out['aux_outputs'] = self._set_aux_loss_hoi(outputs_obj_class, outputs_class,
                                                    outputs_sub_coord, outputs_obj_coord)
            elif dtype=='att':
                out['aux_outputs'] = self._set_aux_loss_att(outputs_class)

        return out

    def gating_function(self, pooled_feature, object_embeddings):
        object_embeds = []
        for object_embedding in object_embeddings:
            for embedding in object_embedding:
                object_embeds.append(torch.tensor(embedding).unsqueeze(0))
        object_embeds = torch.cat(object_embeds)
        
        if len(object_embeds) != len(pooled_feature):
            import pdb; pdb.set_trace()

        assert len(object_embeds) == len(pooled_feature)
        gated_embedding = self.obj_em_relu(self.obj_em_w1(object_embeds.cuda())).sigmoid()
        gated_feature = (gated_embedding.unsqueeze(-1).unsqueeze(-1))*pooled_feature

        return gated_feature 

    def masking_pts(self,masking_shape,feature_size,pts):
        masking_weight =[]
        for data in pts:
            if data is None: #no polygon annotation
                mask = np.ones([feature_size[0],feature_size[1]])
                masking_weight.append(mask)
            else:                
                mask_shape = (masking_shape[data[0]][0],masking_shape[data[0]][1])
                mask = polygon2mask(mask_shape, data[1])
                mask = np.pad(mask, ((0,feature_size[0]-mask.shape[0]),(0,feature_size[1]-mask.shape[1])), 'constant', constant_values=False)
                mask = mask.astype(int)
                #mask = np.where(mask==0,1,mask)
                masking_weight.append(mask)

        assert mask.shape == feature_size

        return np.float32(np.array(masking_weight))

    def crop2box(self, unnorm_box, binary_mask): 
        cropped_mask = binary_mask[int(unnorm_box[1].item()):int(unnorm_box[3].item()),int(unnorm_box[0].item()):int(unnorm_box[2].item())]
        return cropped_mask

    def poly2mask(self, orig_size, polygons):
        binary_masks = []
        for polygon in polygons:
            if polygon is not None:
                binary_mask = polygon2mask(orig_size, polygon[0]).astype(int) #orig_size : (H,W)
            else:
                binary_mask = None
            binary_masks.append(binary_mask)
        return binary_masks


    def mask_normalizer(self,orig_size,pts,feature_size): #orig_size : (H,W)
        if pts is None:        
            tmp = None
            # final_pts = [int(feature_size[2])]
            # final_pts.append(tmp)
            return tmp
        else:
            pts = np.array(pts[0])
            tmp = np.empty(pts.shape)                
            tmp[...,0], tmp[...,1] = feature_size[1]*(pts[...,0] / orig_size[1]), feature_size[0]*(pts[...,1] / orig_size[0])
            final_pts = [int(feature_size[2])]
            final_pts.append(tmp)
        return final_pts #[img_index, pts np array]
 
    def convert_bbox(self,bbox:List): #annotation bbox (c_x,c_y,w,h)-> (x1,y1,x2,y2) for roi align
        c_x, c_y, w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        x1,y1 = c_x-(w/2), c_y-(h/2)
        x2,y2 = c_x+(w/2), c_y+(h/2)  
        return [x1,y1,x2,y2]

    # def convert_bbox(self,bbox:List): #annotation bbox (c_x,c_y,w,h)-> (x1,y1,x2,y2) for roi align
    #     import pdb; pdb.set_trace()
    #     x1,y1,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
    #     x2,y2 = x1+w, y1+h  
    #     return [x1,y1,x2,y2]

    @torch.jit.unused
    def _set_aux_loss_hoi(self, outputs_obj_class, outputs_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_obj_logits': a, 'pred_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[:-1], outputs_class[:-1],
                                      outputs_sub_coord[:-1], outputs_obj_coord[:-1])]
    @torch.jit.unused
    def _set_aux_loss_att(self, outputs_class):
        # return [{'pred_logits': a}
        #         for a in outputs_class[:-1]]
        return [{'pred_obj_logits': a, 'pred_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_class[:-1],
                                      outputs_class[:-1], outputs_class[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, loss_type,args=None):
        super().__init__()

        assert loss_type == 'bce' or loss_type == 'focal'

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.loss_type = loss_type
        if args.att_det or 'vaw' in args.mtl_data:
            self.register_buffer('fed_loss_weight', load_class_freq(freq_weight=0.5))

    def loss_obj_labels(self, outputs, targets, indices, num_att_or_inter, dtype, log=True):
        
        if dtype=='att':
            #losses={'loss_obj_ce': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0]}
            losses={'loss_obj_ce': torch.zeros_like(torch.tensor(1)).cuda()}
            if log:
                losses.update({'obj_class_error':outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0]})
            return losses
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_att_or_inter, dtype):
        if dtype=='att':
            return {'obj_cardinality_error': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0]}
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_att_or_inter ,dtype):
        
        if dtype=='att':
            return {'loss_verb_ce': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0]}
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        if self.loss_type == 'bce':
            loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    def loss_sub_obj_boxes(self, outputs, targets, indices, num_att_or_inter,dtype):
        
        if dtype=='att':
            return {'loss_sub_bbox': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0],
                    'loss_sub_giou': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0],
                    'loss_obj_bbox': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0],
                    'loss_obj_giou': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0]}
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_att_or_inter
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_att_or_inter
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def loss_mask_labels(self, outputs, targets, indices, num_att_or_inter,dtype):
        mask_predictions = outputs['mask_pred_logits']
        gts = []
        for target in targets:
            binary_masks = target['masks']
            for binary_mask in binary_masks:
                if binary_mask is None:
                    gts.append(None)
                else:
                    gts.append(torch.tensor(binary_mask))
        # for i,target in enumerate(targets):
        #     batch_index = i
        #     binary_mask = target['masks']
        #     #mask_prediction = outputs['mask_pred_logits'][i].repeat((len(binary_mask),1,1))
        #     #mask_predictions.append(mask_prediction)
        #     gt_masks.append(binary_mask)

        # mask_predictions = torch.cat(mask_predictions)
        # import pdb; pdb.set_trace()
        # gts = []        
        # for masks in gt_masks:
        #     for mask in masks:
        #         if mask is None:
        #             gts.append(None)
        #         else:
        #             gts.append(torch.tensor(mask))
        assert len(mask_predictions) == len(gts)

        loss_mask_ce = 0
        for pred, gt in zip(mask_predictions,gts):
            if gt is not None:
                size_H,size_W = pred.shape[0], pred.shape[1]
                resized_gt = F.interpolate(gt.float().unsqueeze(0).unsqueeze(0), size=(size_H,size_W), mode='bilinear')
                loss = F.binary_cross_entropy_with_logits(pred.sigmoid(), resized_gt[0][0].cuda())
                loss_mask_ce += loss
        losses = {'loss_mask_ce': loss_mask_ce}        
        return losses

    def loss_att_labels(self, outputs, targets, indices, num_att_or_inter,dtype):
        
        if dtype=='hoi':
            return {'loss_att_ce': outputs['pred_logits'].new_zeros([1],dtype=torch.float32)[0]}


        src_logits = outputs['pred_logits'] 
        target_classes=torch.cat([t['pos_att_classes'] for t in targets])

        pos_gt_classes = torch.nonzero(target_classes==1)[...,-1]
        neg_classes_o = torch.cat([t['neg_att_classes'] for t in targets])
        neg_gt_classes = torch.nonzero(neg_classes_o==1)[...,-1]

        inds = get_fed_loss_inds(
            gt_classes=torch.cat([pos_gt_classes,neg_gt_classes]),
            num_sample_cats=50,
            weight=self.fed_loss_weight,
            C=src_logits.shape[1])
        if self.loss_type == 'bce':
            loss_att_ce = F.binary_cross_entropy_with_logits(src_logits[...,inds], target_classes[...,inds])
        elif self.loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            loss_att_ce = self._neg_loss(src_logits[...,inds], target_classes[...,inds])
        losses = {'loss_att_ce': loss_att_ce}        
        return losses

    # def loss_att_obj_labels(self, outputs, targets, indices, num_att_or_inter,dtype, log=True):
        
    #     if dtype=='hoi':
    #         losses = {'loss_att_obj_ce': outputs['pred_obj_logits'].new_zeros([1],dtype=torch.float32)[0]}
    #         if log:
    #             losses.update({'obj_att_class_error':outputs['pred_obj_logits'].new_zeros([1],dtype=torch.float32)[0]})
    #         return losses
    #     assert 'pred_obj_logits' in outputs
    #     src_logits = outputs['pred_obj_logits']

    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
    #     target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
    #                                 dtype=torch.int64, device=src_logits.device)
    #     target_classes[idx] = target_classes_o

    #     loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
    #     losses = {'loss_att_obj_ce': loss_obj_ce}

    #     if log:
    #         losses['obj_att_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    #     return losses

    # @torch.no_grad()
    # def loss_att_obj_cardinality(self, outputs, targets, indices, num_att_or_inter,dtype):
    #     if dtype=='hoi':
    #         return {'obj_cardinality_error': torch.tensor(0,dtype=torch.float32,device='cuda')}
    #     pred_logits = outputs['pred_obj_logits']
    #     device = pred_logits.device
    #     tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
    #     card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    #     card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    #     losses = {'obj_cardinality_error': card_err}
    #     return losses

    # def loss_att_obj_boxes(self, outputs, targets, indices, num_att_or_inter,dtype):
        
    #     if dtype=='hoi':
    #         return {'loss_att_obj_bbox': outputs['pred_obj_boxes'].new_zeros([1],dtype=torch.float32)[0],
    #                   'loss_att_obj_giou': outputs['pred_obj_boxes'].new_zeros([1],dtype=torch.float32)[0]  }
    #     assert 'pred_obj_boxes' in outputs
    #     idx = self._get_src_permutation_idx(indices)
    #     # src_sub_boxes = outputs['pred_sub_boxes'][idx]
    #     src_obj_boxes = outputs['pred_obj_boxes'][idx]
    #     target_obj_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    #     # exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

    #     losses = {}
    
    #     # loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
    #     loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
    #     # losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_att_or_inter
    #     losses['loss_att_obj_bbox'] = (loss_obj_bbox).sum() / num_att_or_inter
        
    #     loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
    #                                                         box_cxcywh_to_xyxy(target_obj_boxes)))
        
    #     losses['loss_att_obj_giou'] = loss_obj_giou.sum() / num_att_or_inter


        return losses
    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, dtype,**kwargs):
        # loss_map = {
        #     'obj_labels': self.loss_obj_labels,
        #     'att_obj_labels':self.loss_att_obj_labels,
        #     'obj_cardinality': self.loss_obj_cardinality,
        #     'att_obj_cardinality':self.loss_att_obj_cardinality,
        #     'verb_labels': self.loss_verb_labels,
        #     'att_labels': self.loss_att_labels,
        #     'sub_obj_boxes': self.loss_sub_obj_boxes,
        #     'obj_att_boxes':self.loss_att_obj_boxes,
        # }
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'att_labels': self.loss_att_labels,
            'mask_labels' : self.loss_mask_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        
        return loss_map[loss](outputs, targets, indices, num, dtype, **kwargs)
    
    def poly2mask(self, orig_size, polygons):
        binary_masks = []
        for polygon in polygons:
            if polygon is not None:
                binary_mask = polygon2mask(orig_size, polygon[0]).astype(int) #orig_size : (H,W)
            else:
                binary_mask = None
            binary_masks.append(binary_mask)
        return binary_masks

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        dtype=outputs['type']
        
        # Retrieve the matching between the outputs of the last layer and the targets
        
        if dtype == 'att':
            indices = None
        else:
            indices = self.matcher(outputs_without_aux, targets, dtype)

        num_att_or_inter = sum(len(t['obj_labels']) for t in targets) if outputs['type'] =='hoi' else sum(len(t['labels']) for t in targets)
        num_att_or_inter = torch.as_tensor([num_att_or_inter], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_att_or_inter)
        num_att_or_inter = torch.clamp(num_att_or_inter / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            #import pdb; pdb.set_trace()
            losses.update(self.get_loss(loss, outputs, targets, indices, num_att_or_inter,dtype))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if dtype == 'hoi':    
                    indices = self.matcher(aux_outputs, targets, dtype)
                    for loss in ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']:
                        kwargs = {}
                        if loss == 'obj_labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_att_or_inter,dtype, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

                #dummy value for att
                elif dtype == 'att':
                    if i>=5:
                        break
                    indices = None                
                    for loss in ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']:
                        kwargs = {}
                        if loss == 'obj_labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_att_or_inter,dtype, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                                                
        return losses


class PostProcessHOI_ATT(nn.Module):

    def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        if outputs['type'] == 'hoi':
            out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                            outputs['pred_logits'], \
                                                                            outputs['pred_sub_boxes'], \
                                                                            outputs['pred_obj_boxes']
            assert len(out_obj_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2
            if outputs['dataset']=='vcoco':
                obj_prob = F.softmax(out_obj_logits, -1)
                obj_scores, obj_labels = obj_prob[..., :-1].max(-1)
            elif outputs['dataset']=='hico':
                obj_prob = F.softmax(torch.cat([out_obj_logits[...,:-2],out_obj_logits[...,-1:]],-1), -1)
                obj_scores, obj_labels = obj_prob[..., :-1].max(-1)
            verb_scores = out_verb_logits.sigmoid()

            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
            sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
            sub_boxes = sub_boxes * scale_fct[:, None, :]
            obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
            obj_boxes = obj_boxes * scale_fct[:, None, :]

            results = []
            for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
                sl = torch.full_like(ol, self.subject_category_id)
                l = torch.cat((sl, ol))
                b = torch.cat((sb, ob))
                results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})
                vs = vs * os.unsqueeze(1)

                ids = torch.arange(b.shape[0])

                results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                    'obj_ids': ids[ids.shape[0] // 2:]})
        elif outputs['type'] == 'att':
            out_obj_logits, out_att_logits,out_obj_boxes = outputs['pred_obj_logits'], \
                                                                            outputs['pred_logits'], \
                                                                            outputs['pred_obj_boxes']

            assert len(out_obj_logits) == len(target_sizes)
            assert target_sizes.shape[1] == 2

            # obj_prob = F.softmax(out_obj_logits, -1)
            obj_prob = F.softmax(torch.cat([out_obj_logits[...,:-2],out_obj_logits[...,-1:]],-1), -1)
            obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

            attr_scores = out_att_logits.sigmoid()

            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(attr_scores.device)
            obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
            obj_boxes = obj_boxes * scale_fct[:, None, :]
            
            results = []
            for ol, ats, ob in zip(obj_labels, attr_scores, obj_boxes):
                results.append({'labels': ol.to('cpu'), 'boxes': ob.to('cpu')})                
                ids = torch.arange(ob.shape[0])
                res_dict = {
                    'attr_scores': ats.to('cpu'),
                    'obj_ids': ids                    
                }
                results[-1].update(res_dict)

        return results
