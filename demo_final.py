import argparse
import cv2
import os
from models import build_model
from main import get_args_parser
import torch
import numpy as np
from PIL import Image
import itertools
import util.misc as utils
from datasets.vcoco import make_vcoco_transforms
import pandas as pd
from index2cat import vcoco_index_2_cat, hico_index_2_cat, vaw_index_2_cat, color_index
import json
from util import builtin_meta,box_ops


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Demo():
    def __init__(self, args):
        self.video_path = args.video_file
        self.fps = args.fps
        self.output_dir = 'output_video/'+args.video_file.split('/')[-1][:-4]+'_demo.mp4'
        self.cap = cv2.VideoCapture(args.video_file) if not args.webcam else cv2.VideoCapture(1)
        # self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.checkpoint = args.checkpoint
        self.frame_num = 0
        self.inf_type = args.inf_type
        self.num_obj_classes = args.num_obj_classes
        self.color_index = color_index()
        if 'vcoco' in self.inf_type:
            CORRECT_MAT_PATH_VCOCO = 'data/v-coco/annotations/corre_vcoco.npy'
            correct_mat_vcoco = np.load(CORRECT_MAT_PATH_VCOCO)
            self.correct_mat_vcoco = np.concatenate((correct_mat_vcoco, np.ones((correct_mat_vcoco.shape[0], 1))), axis=1)
            self.correct_mat_vcoco[[1,4,5,18,23,26]]=0
        if 'hico' in self.inf_type:
            CORRECT_MAT_PATH_HICO = 'data/hico_20160224_det/annotations/corre_hico.npy'
            self.correct_mat_hico = np.load(CORRECT_MAT_PATH_HICO)
            self.correct_mat_hico[[57,36,86,87,76,35,23,24,43,101,8,104,9,15,16,20,44,73,102,58]]=0
        # if 'vaw' in  self.inf_type:
        #     attribute_freq = 'data/vaw/annotations/vaw_coco_train_cat_info.json'
        #     self.valid_masks = self.valid_att_idxs(attribute_freq)
        #     self.valid_masks[[330]]=0
        #     if args.color:
        #         non_color_index = [i for i in range(self.valid_masks.shape[0])if i not in color_index()]
        #         self.valid_masks[non_color_index]=0
        
        self.topk = args.top_k
        self.count_dict = {}
        self.output_i={}
        self.color_ = builtin_meta._get_coco_instances_meta()['thing_colors']
        if 'vcoco' in args.inf_type:
            self.color_.append([0,0,0])
            
        self.iou_threshold = args.iou_threshold
        self.attr_threshold = args.attr_threshold
        self.attr_topk = args.attr_topk

    def hoi_att_transforms(self, image_set):
        transforms = make_vcoco_transforms(image_set)
        return transforms

    # def inference_for_vid(self, model, frame, args=None):
    #     img = Image.fromarray(frame)
    #     transform = self.hoi_att_transforms('val')
    #     sample = img.copy()
    #     sample, _ = transform(sample, None)
    #     dataset = args.inf_type #hico or vcoco or hoi or vaw
    #     if dataset == 'vaw':
    #         dtype = 'att'
    #     else: 
    #         dtype = 'hoi'

        
    #     output = model(sample.unsqueeze(0).to(device),dtype,dataset)
    #     return output
                
    def inference_for_vid(self, model, frame, bbox, inf_type, args=None, eval_mode=False):
        img = Image.fromarray(frame)
        transform = self.hoi_att_transforms('val')
        sample = img.copy()
        sample, _ = transform(sample, None)
        if inf_type == 'vaw':
            outputs = model([sample.cuda()],bbox,'att',inf_type,args) #torch.Size([1, 620])
        elif inf_type == 'vcoco' or inf_type == 'hico': 
            outputs = model([sample.cuda()],None,'hoi',inf_type,args) 
        return outputs

    def valid_att_idxs(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)
        num_attributes = 620
        valid_masks = np.zeros(num_attributes)
        for i in annotations:
            if i['instance_count'] > 0:
                valid_masks[i['id']]=1

        return valid_masks
    
   
    def change_format(self,results,dataset, args,frame):

        
        if dataset in ['vcoco','hico']:
            if 'obj_box_iou' not in self.output_i:
                obj_box_iou = box_ops.box_iou(results['boxes'][100:],results['boxes'][100:])
                self.output_i['obj_box_iou'] = obj_box_iou
            if 'sub_box_iou' not in self.output_i:
                sub_box_iou = box_ops.box_iou(results['boxes'][:100],results['boxes'][:100])
                self.output_i['sub_box_iou'] = sub_box_iou
            img_preds = {k: v.to('cpu').numpy() for k, v in results.items() if 'attr' not in k}
            # self.output_i['box_predictions']=[]
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            # for h_box, o_box in zip(img_preds['boxes'][:100], img_preds['boxes'][100:]):
            #     self.output_i['box_predictions'].append({'h_bbox':h_box.tolist(), 'o_bbox': o_box.tolist()})
            
            self.output_i['hoi_box_predictions'] = bboxes
            
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()
            
            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                if img_preds['verb_scores'].shape[1]==29:
                    correct_mat = self.correct_mat_vcoco
                elif img_preds['verb_scores'].shape[1]==117:
                    correct_mat = self.correct_mat_hico
                # import pdb;pdb.set_trace()
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.topk]
            else:
                hois = []
            
            self.output_i[dataset]=hois
            num_hoi = len(hois)
            sub_boxes = np.stack([self.output_i['hoi_box_predictions'][hoi['subject_id']]['bbox'] for hoi in hois])
            obj_boxes = np.stack([self.output_i['hoi_box_predictions'][hoi['object_id']]['bbox'] for hoi in hois])
            all_boxes = torch.cat([torch.tensor(sub_boxes),torch.tensor(obj_boxes)])
            all_att_outputs = self.inference_for_vid(model, frame, all_boxes, 'vaw', args) 
            all_att_outputs = all_att_outputs.split(num_hoi,0)
            sub_outputs = all_att_outputs[0]
            obj_outputs = all_att_outputs[1]
            for nh in range(num_hoi):
                sub_att_score,sub_att_idx = torch.topk(sub_outputs[nh],k=self.attr_topk,dim=-1)
                obj_att_score,obj_att_idx = torch.topk(obj_outputs[nh],k=self.attr_topk,dim=-1)
                sub_att_idx = sub_att_idx[sub_att_score>=self.attr_threshold].cpu().tolist()
                obj_att_idx = obj_att_idx[obj_att_score>=self.attr_threshold].cpu().tolist()
                self.output_i[dataset][nh].update({'sub_att_idx':sub_att_idx, 'obj_att_idx':obj_att_idx})
            
                # import pdb;pdb.set_trace()
        else:
            if 'obj_box_iou' not in self.output_i:
                obj_box_iou = box_ops.box_iou(results['boxes'],results['boxes'])
                self.output_i['obj_box_iou'] = obj_box_iou
            img_preds = {k: v.to('cpu').numpy() for k, v in results.items() if 'verb' not in k}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            attr_scores = img_preds['attr_scores']
            masks = self.valid_masks[np.newaxis,:]
            attr_scores *= masks
            # import pdb;pdb.set_trace()
            attr_labels = np.tile(np.arange(attr_scores.shape[1]), (attr_scores.shape[0], 1))
            # subject_ids = np.tile(img_preds['sub_ids'], (attr_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (attr_scores.shape[1], 1)).T

            attr_scores = attr_scores.ravel()
            attr_labels = attr_labels.ravel()
            # subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()
            
            # if len(subject_ids) > 0:
            
            object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
            

            attrs = [{'object_id': object_id, 'category_id': category_id, 'score': score} for
                    object_id, category_id, score in zip(object_ids, attr_labels, attr_scores)]
            attrs.sort(key=lambda k: (k.get('score', 0)), reverse=True)
            attrs = attrs[:self.topk]
            
            self.output_i[dataset] = attrs
            self.output_i['attr_box_predictions'] = bboxes
            # for o_box in zip(img_preds['boxes']):
            #     self.output_i['attr_box_predictions'].append({'o_bbox': o_box.tolist()})
            
            
        # return self.output_i

    def index_2_cat(self, index, inf_type):
        if inf_type =='vcoco':
            return vcoco_index_2_cat(index)
        elif inf_type =='hico':
            return hico_index_2_cat(index)
        elif inf_type =='vaw':
            return vaw_index_2_cat(index)

    def make_color_dict(self, class_num):
        color_dict = {i: list(np.random.random(size=3) * 256) for i in range(class_num)}
        return color_dict 
    
    def draw_img_all(self, img, output_i, threshold, color_dict, inf_type,color=False):
        
        vis_img = img.copy()
        if 'vaw' in output_i:
            # import pdb;pdb.set_trace()
            # print('asdsds')
            
            object_list_vis={}
            for predict in output_i['vaw']:

                #prediction threshold
                if predict['score'] < self.attr_threshold:
                    continue
                object_id = predict['object_id']
                attr_class = predict['category_id']
                o_class = output_i['attr_box_predictions'][object_id]['category_id']
                o_bbox = output_i['attr_box_predictions'][object_id]['bbox']
                attr_score = predict['score']
                # max_attr_score = predict['max_score']
                single_out = {'object_box':np.array(o_bbox),'object_id':object_id, 'object_label':np.array(o_class), 'attr_score':np.array(attr_score)}
                # list_predictions.append(single_out)
                if object_id not in object_list_vis:
                    if len(object_list_vis)>0:
                        iou_ = torch.max(torch.tensor([output_i['obj_box_iou'][0][k,object_id].item() for k in object_list_vis.keys()]),dim=0)
                        if iou_[0]>self.iou_threshold:
                            k=[key for key in object_list_vis.keys()]
                            obj_id = iou_[1].item()
                            object_id = k[obj_id]
                            object_list_vis[object_id]+=1
                            o_bbox = output_i['attr_box_predictions'][object_id]['bbox']
                        else:
                            vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                            object_list_vis[object_id]=1
                    else:
                        vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                        object_list_vis[object_id]=1
                else:
                    object_list_vis[object_id]+=1
                print(f'drawing attr object box')
                text_size, BaseLine=cv2.getTextSize(self.index_2_cat(attr_class,'vaw'),cv2.FONT_HERSHEY_SIMPLEX,1,2)

                #text height for multiple attributes
                text_size_y = text_size[1] 
                cnt = object_list_vis[object_id]
                # for attr in attributes[0]:
                
                    
                text = self.index_2_cat(attr_class,'vaw')
                    
                text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                # if o_bbox[1]-cnt*text_size_y < 0 or o_bbox[0] < 0:
                #     break
                
                text_box = [o_bbox[0], o_bbox[1]-cnt*text_size_y, o_bbox[0]+text_size[0],o_bbox[1]-(cnt-1)*text_size_y]

                #draw text
                vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                print(f'drawing attribute box : {text}, {cnt}')
                # cnt += 1 
        if 'vcoco' in output_i or 'hico' in output_i:
            hoi_subject_list_vis={}
        if 'vcoco' in output_i:
            # import pdb;pdb.set_trace()
            # print('asdsds')
            
            # import pdb;pdb.set_trace()
            for predict in output_i['vcoco']:

                #prediction threshold
                if predict['score'] < threshold:
                    continue
                subject_id = predict['subject_id']
                s_bbox = output_i['hoi_box_predictions'][subject_id]['bbox']
                object_id = predict['object_id']
                o_class = output_i['hoi_box_predictions'][object_id]['category_id']
                o_bbox = output_i['hoi_box_predictions'][object_id]['bbox']
                hoi_class = predict['category_id']
                hoi_score = predict['score']
                sub_attr = predict['sub_att_idx']
                obj_attr = predict['obj_att_idx']

                # max_attr_score = predict['max_score']
                single_out = {'subject_box':np.array(s_bbox),
                              'subject_id':subject_id,
                              'object_box':np.array(o_bbox),
                              'object_id':object_id,
                              'object_label':np.array(o_class), 
                              'hoi_score':np.array(hoi_score),
                              'sub_att_idx':np.array(sub_attr),
                              'obj_att_idx':np.array(obj_attr)}
                # list_predictions.append(single_out)
                if subject_id not in hoi_subject_list_vis:
                    
                    if len(hoi_subject_list_vis)>0:
                        # import pdb;pdb.set_trace()
                        iou_ = torch.max(torch.tensor([output_i['sub_box_iou'][0][k,subject_id].item() for k in hoi_subject_list_vis.keys()]),dim=0)
                        if iou_[0]>self.iou_threshold:
                            k=[key for key in hoi_subject_list_vis.keys()]
                            subj_id = iou_[1].item()
                            subject_id = k[subj_id]
                            hoi_subject_list_vis[subject_id]+=1
                            s_bbox = output_i['hoi_box_predictions'][subject_id]['bbox']
                        else:
                            vis_img = cv2.rectangle(vis_img, (int(s_bbox[0]),int(s_bbox[1])), (int(s_bbox[2]),int(s_bbox[3])), color_dict[0], 3)
                            hoi_subject_list_vis[subject_id]=1
                        if o_class!=80:
                            vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                            
                    else:
                        vis_img = cv2.rectangle(vis_img, (int(s_bbox[0]),int(s_bbox[1])), (int(s_bbox[2]),int(s_bbox[3])), color_dict[0], 3)
                        if o_class!=80:
                            vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                        hoi_subject_list_vis[subject_id]=1
                else:
                    hoi_subject_list_vis[subject_id]+=1
                print(f'drawing hoi boxes')
                text = self.index_2_cat(hoi_class,'vcoco')
                text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)

                #text height for multiple attributes
                text_size_y = text_size[1] +5
                cnt = hoi_subject_list_vis[subject_id]
                # for attr in attributes[0]:
                
                    
                    
                # text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                # if o_bbox[1]-cnt*text_size_y < 0 or o_bbox[0] < 0:
                #     break
                
                text_box = [s_bbox[0], s_bbox[1]-cnt*text_size_y, s_bbox[0]+text_size[0],s_bbox[1]-(cnt-1)*text_size_y]

                #draw text
                vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                # topk = 0
                for ind,(sa,oa) in enumerate(zip(sub_attr,obj_attr)):
                    # sub_text_size, BaseLine=cv2.getTextSize(self.index_2_cat(sa,'vaw'),cv2.FONT_HERSHEY_SIMPLEX,1,2)
                    obj_text_size, BaseLine=cv2.getTextSize(self.index_2_cat(oa,'vaw'),cv2.FONT_HERSHEY_SIMPLEX,1,2)

                    #text height for multiple attributes
                    text_size_y = obj_text_size[1] 
                    # cnt = object_list_vis[object_id]
                    # for attr in attributes[0]:
                    
                        
                    text = self.index_2_cat(oa,'vaw')
                        
                    text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                    # if o_bbox[1]-cnt*text_size_y < 0 or o_bbox[0] < 0:
                    #     break
                    
                    text_obj_box = [o_bbox[2]-text_size[0], o_bbox[1]-(ind+1)*text_size_y, o_bbox[2],o_bbox[1]-(ind)*text_size_y]

                    #draw text
                    vis_img = cv2.rectangle(vis_img, (int(text_obj_box[0]),int(text_obj_box[1])),(int(text_obj_box[2]),int(text_obj_box[3])), color_dict[int(o_class)], -1)
                    vis_img = cv2.putText(vis_img, text, (int(text_obj_box[0]),int(text_obj_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                    print('drawing object attribute') 
                    print(f'obj attribute : {text}')    

                    sub_text_size, BaseLine=cv2.getTextSize(self.index_2_cat(sa,'vaw'),cv2.FONT_HERSHEY_SIMPLEX,1,2)

                    #text height for multiple attributes
                    text_size_y = sub_text_size[1] 
                    # cnt = object_list_vis[object_id]
                    # for attr in attributes[0]:
                    
                        
                    text = self.index_2_cat(sa,'vaw')
                        
                    text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                    # if o_bbox[1]-cnt*text_size_y < 0 or o_bbox[0] < 0:
                    #     break
                    
                    text_obj_box = [s_bbox[2]-text_size[0], s_bbox[1]-cnt*text_size_y, s_bbox[2],s_bbox[1]-(cnt-1)*text_size_y]

                    #draw text
                    vis_img = cv2.rectangle(vis_img, (int(text_obj_box[0]),int(text_obj_box[1])),(int(text_obj_box[2]),int(text_obj_box[3])), color_dict[0], -1)
                    vis_img = cv2.putText(vis_img, text, (int(text_obj_box[0]),int(text_obj_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                    
                    print('drawing subject attribute') 
                    print(f'sub attribute : {text}')   
                    # if ind == self.attr_topk-1:
                    #     break
                print('drawing vcoco action box') 
                print(f'action : {text}')
        
        if 'hico' in output_i:
            # import pdb;pdb.set_trace()
            # print('asdsds')
            
            # hoi_object_list_vis={}
            for predict in output_i['hico']:

                #prediction threshold
                if predict['score'] < threshold:
                    continue
                subject_id = predict['subject_id']
                s_bbox = output_i['hoi_box_predictions'][subject_id]['bbox']
                object_id = predict['object_id']
                o_class = output_i['hoi_box_predictions'][object_id]['category_id']
                o_bbox = output_i['hoi_box_predictions'][object_id]['bbox']
                hoi_class = predict['category_id']
                hoi_score = predict['score']
                # max_attr_score = predict['max_score']
                single_out = {'subject_box':np.array(s_bbox),
                              'subject_id':subject_id,
                              'object_box':np.array(o_bbox),
                              'object_id':object_id,
                              'object_label':np.array(o_class), 
                              'hoi_score':np.array(hoi_score)}
                # list_predictions.append(single_out)
                if subject_id not in hoi_subject_list_vis:
                    
                    if len(hoi_subject_list_vis)>0:
                        # import pdb;pdb.set_trace()
                        iou_ = torch.max(torch.tensor([output_i['sub_box_iou'][0][k,subject_id].item() for k in hoi_subject_list_vis.keys()]),dim=0)
                        if iou_[0]>self.iou_threshold:
                            k=[key for key in hoi_subject_list_vis.keys()]
                            subj_id = iou_[1].item()
                            subject_id = k[subj_id]
                            hoi_subject_list_vis[subject_id]+=1
                            s_bbox = output_i['hoi_box_predictions'][subject_id]['bbox']
                        else:
                            vis_img = cv2.rectangle(vis_img, (int(s_bbox[0]),int(s_bbox[1])), (int(s_bbox[2]),int(s_bbox[3])), color_dict[0], 3)
                            hoi_subject_list_vis[subject_id]=1
                        if o_class!=80:
                            vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                            
                    else:
                        vis_img = cv2.rectangle(vis_img, (int(s_bbox[0]),int(s_bbox[1])), (int(s_bbox[2]),int(s_bbox[3])), color_dict[0], 3)
                        if o_class!=80:
                            vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                        hoi_subject_list_vis[subject_id]=1
                else:
                    hoi_subject_list_vis[subject_id]+=1
                print(f'drawing hoi boxes')
                text = self.index_2_cat(hoi_class,'hico')
                text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                # text_size, BaseLine=cv2.getTextSize(self.index_2_cat(hoi_class,'hico'),cv2.FONT_HERSHEY_SIMPLEX,1,2)

                #text height for multiple attributes
                text_size_y = text_size[1]+5 
                cnt = hoi_subject_list_vis[subject_id]
                # for attr in attributes[0]:
                
                    
                # text = self.index_2_cat(attr,args.inf_type)
                    
                # text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                # if o_bbox[1]-cnt*text_size_y < 0 or o_bbox[0] < 0:
                #     break
                
                text_box = [s_bbox[0], s_bbox[1]-cnt*text_size_y, s_bbox[0]+text_size[0],s_bbox[1]-(cnt-1)*text_size_y]

                #draw text
                vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                print('drawing hico action box') 
                print(f'action : {text}')
        
        
        return vis_img
        
    def save_video(self, args):
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))	
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        orig_size = torch.as_tensor([frame_height,frame_width]).unsqueeze(0).to('cuda')
        output_file = cv2.VideoWriter(self.output_dir, self.fourcc, self.fps, frame_size)
        checkpoint = torch.load(self.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'],strict=False)
        # model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        # import pdb;pdb.set_trace()
        color_dict = self.make_color_dict(self.num_obj_classes)
        # import pdb;pdb.set_trace()
        while(True):
            retval, frame = self.cap.read() 
            print(f'frame_num : {self.frame_num}')
            frame = cv2.flip(frame,1)
            if not retval:
                break
            hoi_outputs = self.inference_for_vid(model, frame, None, 'vcoco', args)
            # outputs = self.inference_for_vid(model, frame, args)
            preds = []
            # if 'hico' in hoi_outputs:
            #     results_hico = postprocessors(hoi_outputs['hico'], orig_size)
            #     preds=list(itertools.chain.from_iterable(utils.all_gather(results_hico)))
            #     self.change_format(preds[0],'hico', args)
            # if 'vcoco' in hoi_outputs:
            results_vcoco = postprocessors(hoi_outputs, orig_size)
            preds=list(itertools.chain.from_iterable(utils.all_gather(results_vcoco)))
            self.change_format(preds[0],'vcoco', args, frame)
            # if 'vaw' in self.inf_type:
            #     # import pdb;pdb.set_trace()

            #     sub_attr = self.inference_for_vid(model, frame, sub_boxes, 'vaw', args)
            #     o_attr = self.inference_for_vid(model, frame, obj_boxes, 'vaw', args)

            #     results_vaw = postprocessors(hoi_outputs['vaw'], orig_size)
            #     preds=list(itertools.chain.from_iterable(utils.all_gather(results_vaw)))
            #     self.change_format(preds[0],'vaw', args)
            
           
            vis_img = self.draw_img_all(frame,self.output_i,threshold=args.threshold,color_dict=self.color_,inf_type=args.inf_type)
            self.output_i = {}
            if args.webcam:
                cv2.namedWindow('Online Demo', cv2.WINDOW_NORMAL)
                cv2.imshow('Online Demo',vis_img)
                if cv2.waitKey(1)==27:
                    break
            else:
                output_file.write(vis_img)
            self.frame_num += 1
            
        self.cap.release()
        if args.webcam:
            output_file.release()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    parser = argparse.ArgumentParser('video inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    model, _, postprocessors = build_model(args)
    demo = Demo(args)
    demo.save_video(args)