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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Demo():
    def __init__(self, args):
        self.video_path = args.video_file
        self.fps = args.fps
        self.output_dir = 'output_video/'+args.video_file.split('/')[-1]
        self.cap = cv2.VideoCapture(args.video_file)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.checkpoint = args.checkpoint
        self.frame_num = 0
        self.inf_type = args.inf_type
        self.num_obj_classes = args.num_obj_classes
        if 'vcoco' in self.inf_type:
            CORRECT_MAT_PATH_VCOCO = 'data/v-coco/annotations/corre_vcoco.npy'
            self.correct_mat_vcoco = np.load(CORRECT_MAT_PATH_VCOCO)
            self.correct_mat_vcoco[[1,4,18,23,26]]=0
        elif 'hico' in self.inf_type:
            CORRECT_MAT_PATH_HICO = 'data/hico_20160224_det/annotations/corre_hico.npy'
            self.correct_mat_hico = np.load(CORRECT_MAT_PATH_HICO)
        
        self.color_index = color_index()
        self.topk = args.top_k
        self.count_dict = {}

    def hoi_att_transforms(self, image_set):
        transforms = make_vcoco_transforms(image_set)
        return transforms

    def inference_for_vid(self, model, frame, args=None):
        img = Image.fromarray(frame)
        transform = self.hoi_att_transforms('val')
        sample = img.copy()
        sample, _ = transform(sample, None)
        dataset = args.inf_type #hico or vcoco or hoi or vaw
        if dataset == 'vaw':
            dtype = 'att'
        else: 
            dtype = 'hoi'
        
        # if 'hico' in dataset and 'vcoco' in dataset:
        #     output_hico = model(sample.unsqueeze(0).to(device), dtype, 'hico')
        #     output_vcoco = model(sample.unsqueeze(0).to(device), dtype, 'vcoco')
        #     return output_hico, output_vcoco
        # else:
        #     output = model(sample.unsqueeze(0).to(device), dtype, dataset)
        #     return output

        if 'vcoco' in dataset and 'vaw' in dataset and 'hico' in dataset:
            #import pdb; pdb.set_trace()
            output_vcoco = model(sample.unsqueeze(0).to(device), 'hoi', 'vcoco')
            output_hico = model(sample.unsqueeze(0).to(device), 'hoi', 'hico')
            output_vaw = model(sample.unsqueeze(0).to(device), 'att', 'vaw')
            return output_hico, output_vcoco, output_vaw
        else:
            if 'hico' in dataset and 'vcoco' in dataset:
                output_hico = model(sample.unsqueeze(0).to(device), 'hoi', 'hico')
                output_vcoco = model(sample.unsqueeze(0).to(device), 'hoi', 'vcoco')
                return output_hico, output_vcoco
            
            elif 'hico' in dataset and 'vaw' in dataset:
                output_hico = model(sample.unsqueeze(0).to(device), 'hoi', 'hico')
                output_vaw = model(sample.unsqueeze(0).to(device), 'att', 'vaw')
                return output_hico, output_vaw
            
            elif 'vcoco' in dataset and 'vaw' in dataset:
                #import pdb; pdb.set_trace()
                output_vcoco = model(sample.unsqueeze(0).to(device), 'hoi', 'vcoco')
                output_vaw = model(sample.unsqueeze(0).to(device), 'att', 'vaw')
                return output_vcoco, output_vaw
        
            else:
                output = model(sample.unsqueeze(0).to(device), dtype, dataset)
                return output

    def valid_att_idxs(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)
        num_attributes = 620
        valid_masks = np.zeros(num_attributes)
        for i in annotations:
            if i['instance_count'] > 0:
                valid_masks[i['id']]=1

        return valid_masks
    
    def change_format_all(self,results, args):
    
        
        if 'verb_scores' in results.keys():
            boxes,labels,pair_scores = list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['verb_scores']]))
            output_i={}
            output_i['box_predictions']=[]
            output_i['hoi_predictions']=[]
            
            h_boxes = boxes[:100]
            h_labels = labels[:100]
            o_boxes = boxes[100:]
            o_labels = labels[100:]

            for h_box, o_box in zip(h_boxes, o_boxes):
                output_i['box_predictions'].append({'h_bbox':h_box.tolist(), 'o_bbox': o_box.tolist()})

            for h_label, o_label, pair_score in zip(h_labels, o_labels, pair_scores):
                output_i['hoi_predictions'].append({'subject_id':h_label,'object_id':o_label,'max_score':max(pair_score),'pair_score':pair_score})
        else:
            boxes,labels,attr_scores = list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['attr_scores']]))
            output_i={}
            output_i['predictions']=[]
            attribute_freq = 'data/vaw/annotations/vaw_coco_train_cat_info.json'
            valid_masks = self.valid_att_idxs(attribute_freq)
            for box, label,attr_score in zip(boxes,labels,attr_scores):
                attr_score *= valid_masks
                output_i['predictions'].append({'bbox':box.tolist(), 'object_id': label, 'max_score':max(attr_score), 'pair_score':attr_score})

       
        return output_i
    
    def change_format(self,results, args):

        if args.inf_type == 'vcoco' or args.inf_type == 'hico' or ('hico' in args.inf_type and 'vcoco' in args.inf_type):
            # boxes,labels,pair_scores = list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['verb_scores']]))
            # output_i={}
            # output_i['box_predictions']=[]
            # output_i['hoi_predictions']=[]
            
            # h_boxes = boxes[:100]
            # h_labels = labels[:100]
            # o_boxes = boxes[100:]
            # o_labels = labels[100:]

            # for h_box, o_box in zip(h_boxes, o_boxes):
            #     output_i['box_predictions'].append({'h_bbox':h_box.tolist(), 'o_bbox': o_box.tolist()})

            # for h_label, o_label, pair_score in zip(h_labels, o_labels, pair_scores):
            #     output_i['hoi_predictions'].append({'subject_id':h_label,'object_id':o_label,'max_score':max(pair_score),'pair_score':pair_score})
            img_preds = {k: v.to('cpu').numpy() for k, v in results.items() if 'attr' not in k}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
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
                import pdb;pdb.set_trace()
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.topk]
            else:
                hois = []
            output_i['hoi_predictions']=hois

        elif ('vcoco' in args.inf_type and 'vaw' in args.inf_type) or ('hico' in args.inf_type and 'vaw' in args.inf_type):
            if 'verb_scores' in results.keys():
                # boxes,labels,pair_scores = list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['verb_scores']]))
                # output_i={}
                # output_i['box_predictions']=[]
                # output_i['hoi_predictions']=[]
                
                # h_boxes = boxes[:100]
                # h_labels = labels[:100]
                # o_boxes = boxes[100:]
                # o_labels = labels[100:]
                # # import pdb;pdb.set_trace()
                # for h_box, o_box in zip(h_boxes, o_boxes):
                #     output_i['box_predictions'].append({'h_bbox':h_box.tolist(), 'o_bbox': o_box.tolist()})

                # for h_label, o_label, pair_score in zip(h_labels, o_labels, pair_scores):
                #     output_i['hoi_predictions'].append({'subject_id':h_label,'object_id':o_label,'max_score':max(pair_score),'pair_score':pair_score})
                img_preds = {k: v.to('cpu').numpy() for k, v in results.items() if 'attr' not in k}
                bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
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
                    import pdb;pdb.set_trace()
                    masks = correct_mat[verb_labels, object_labels]
                    hoi_scores *= masks

                    hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                            subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                    hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                    hois = hois[:self.topk]
                else:
                    hois = []
                output_i['hoi_predictions']=hois
            else:
                boxes,labels,attr_scores = list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['attr_scores']]))
                output_i={}
                output_i['predictions']=[]
                attribute_freq = 'data/vaw/annotations/vaw_coco_train_cat_info.json'
                valid_masks = self.valid_att_idxs(attribute_freq)
                for box, label,attr_score in zip(boxes,labels,attr_scores):
                    attr_score *= valid_masks
                    output_i['predictions'].append({'bbox':box.tolist(), 'object_id': label, 'max_score':max(attr_score), 'pair_score':attr_score})

        else:
            boxes,labels,attr_scores = list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['attr_scores']]))
            output_i={}
            output_i['predictions']=[]
            attribute_freq = 'data/vaw/annotations/vaw_coco_train_cat_info.json'
            valid_masks = self.valid_att_idxs(attribute_freq)
            for box, label,attr_score in zip(boxes,labels,attr_scores):
                attr_score *= valid_masks
                output_i['predictions'].append({'bbox':box.tolist(), 'object_id': label, 'max_score':max(attr_score), 'pair_score':attr_score})
        return output_i

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

    def draw_img(self, img, output_i, top_k, threshold, color_dict, inf_type, color=False):
        vis_img = img.copy()
        if inf_type == 'vaw':
            list_predictions = []
            for predict in output_i['predictions']:

                #prediction threshold
                if predict['max_score'] < threshold:
                    continue
                object_id = predict['object_id']
                object_box = predict['bbox']
                attr_score = predict['pair_score']
                max_attr_score = predict['max_score']
                single_out = {'object_box':np.array(object_box), 'object_id':np.array(object_id), 'attr_score':np.array(attr_score), 'max_attr_score':np.array(max_attr_score)}
                list_predictions.append(single_out)
            
            if list_predictions:
                df = pd.DataFrame(list_predictions)
                df = df.sort_values(by=['max_attr_score'],ascending=False).iloc[:top_k]
                list_predictions = df.to_dict('records')
                for i, prediction in enumerate(list_predictions):

                    attributes = np.where(prediction['attr_score'] > threshold)
                    o_bbox = prediction['object_box']
                    o_class = prediction['object_id']                
                    
                    if color: #only color inference
                        for attr in attributes[0]:
                            if attr in self.color_index:
                                #if color index draw object box 
                                vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                                print(f'drawing object box')
                            else:
                                continue
 
                    else:
                        vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                        print(f'drawing object box')
                    
                    text_size, BaseLine=cv2.getTextSize(self.index_2_cat(attributes[0][0],args.inf_type),cv2.FONT_HERSHEY_SIMPLEX,1,2)

                    #text height for multiple attributes
                    text_size_y = text_size[1] 
                    cnt = 1
                    for attr in attributes[0]:
                        if attr not in self.color_index:
                            continue
                        text = self.index_2_cat(attr,args.inf_type)
                            
                        text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                        if o_bbox[1]-cnt*text_size_y < 0 or o_bbox[0] < 0:
                            break
                        text_box = [o_bbox[0], o_bbox[1]-cnt*text_size_y, o_bbox[0]+text_size[0],o_bbox[1]-(cnt-1)*text_size_y]

                        #draw text
                        vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                        vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                        print(f'drawing attribute box : {text}, {cnt}')
                        cnt += 1 
        
        #for hoi multi head inference
        elif ('vcoco' in inf_type) and ('hico' in inf_type):
            actions = {'hico':[],'vcoco':[]}
            for output in output_i:
                for box, hoi in zip(output['box_predictions'], output['hoi_predictions']):
                    
                    #prediction threshold
                    if hoi['max_score'] < threshold:
                        continue
                    
                    subject_id = hoi['subject_id']
                    subject_box = box['h_bbox']
                    object_id = hoi['object_id']
                    object_box = box['o_bbox']
                    verb_score = hoi['pair_score']
                    max_verb_score = hoi['max_score']
                    single_out = {'subject_box':np.array(subject_box), 'object_box':np.array(object_box), 'subject_id':np.array(subject_id), 'object_id':np.array(object_id), 'verb_score':np.array(verb_score), 'max_verb_score':np.array(max_verb_score)}
                    if single_out['verb_score'].shape == (117,): #hico
                        actions['hico'].append(single_out)
                    elif single_out['verb_score'].shape == (29,): #vcoco
                        actions['vcoco'].append(single_out)
                    
                if len(actions['hico']) > 0 and len(actions['vcoco']) > 0:
                    hico,vcoco = pd.DataFrame(actions['hico']), pd.DataFrame(actions['vcoco'])
                    hico,vcoco = hico.sort_values(by=['max_verb_score'],ascending=False).iloc[:top_k], vcoco.sort_values(by=['max_verb_score'],ascending=False).iloc[:top_k] 
                    hico_actions = hico.to_dict('records')
                    vcoco_actions = vcoco.to_dict('records')
                    
                    s_bbox = {}
                    o_bbox = {}
                    for i, action in enumerate(hico_actions):
                        self.count_dict[i] = 1
                        verbs = np.where(action['verb_score'] > threshold)
                        s_bbox[i] = action['subject_box']
                        o_bbox[i] = action['object_box']
                        o_class = action['object_id']
                        vis_img = cv2.rectangle(vis_img, (int(s_bbox[i][0]),int(s_bbox[i][1])), (int(s_bbox[i][2]),int(s_bbox[i][3])), color_dict[0], 3)
                        print('drawing subject box')
                        
                        vis_img = cv2.rectangle(vis_img, (int(o_bbox[i][0]),int(o_bbox[i][1])), (int(o_bbox[i][2]),int(o_bbox[i][3])), color_dict[int(o_class)], 3)
                        print('drawing object box')
                        text_size, BaseLine=cv2.getTextSize(self.index_2_cat(verbs[0][0],'hico'),cv2.FONT_HERSHEY_SIMPLEX,1,2)
                        
                        #text height for multiple verbs (interactions)
                        text_size_y = text_size[1] 
                        
                        for verb in verbs[0]:    
                            text = self.index_2_cat(verb,'hico')
                            text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            if s_bbox[i][1]-self.count_dict[i]*text_size_y < 0 or s_bbox[i][0] < 0:
                                break
                            text_box = [s_bbox[i][0], s_bbox[i][1]-self.count_dict[i]*text_size_y, s_bbox[i][0]+text_size[0],s_bbox[i][1]-(self.count_dict[i]-1)*text_size_y]

                            #draw text
                            vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                            vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                            print('drawing hico action box')
                            print(f'pair number : {i}, action : {text}, action number : {self.count_dict[i]}')
                            self.count_dict[i] += 1 

                    for i, action in enumerate(vcoco_actions):
                        verbs = np.where(action['verb_score'] > threshold)
                        for verb in verbs[0]:
                            text = self.index_2_cat(verb,'vcoco')
                            text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                            
                            #no subject box case
                            if i not in s_bbox:
                                break

                            if (s_bbox[i][1]-self.count_dict[i]*text_size_y < 0 or s_bbox[i][0] < 0):
                                break
  
                            text_box = [s_bbox[i][0], s_bbox[i][1]-self.count_dict[i]*text_size_y, s_bbox[i][0]+text_size[0],s_bbox[i][1]-(self.count_dict[i]-1)*text_size_y]

                            #draw vcoco verb text
                            vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                            vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                            print('drawing vcoco action box') 
                            print(f'pair number : {i}, action : {text}, action number : {self.count_dict[i]}')
                            self.count_dict[i] += 1 

        
        else: #for hoi single head inference 
            list_actions = []
            for box, hoi in zip(output_i['box_predictions'], output_i['hoi_predictions']):

                #prediction threshold
                if hoi['max_score'] < threshold:
                    continue
                
                subject_id = hoi['subject_id']
                subject_box = box['h_bbox']
                object_id = hoi['object_id']
                object_box = box['o_bbox']
                verb_score = hoi['pair_score']
                max_verb_score = hoi['max_score']
                single_out = {'subject_box':np.array(subject_box), 'object_box':np.array(object_box), 'subject_id':np.array(subject_id), 'object_id':np.array(object_id), 'verb_score':np.array(verb_score), 'max_verb_score':np.array(max_verb_score)}
                list_actions.append(single_out)
            
            if list_actions:
                df = pd.DataFrame(list_actions)
                df = df.sort_values(by=['max_verb_score'],ascending=False).iloc[:top_k]
                list_actions = df.to_dict('records')
                for i, action in enumerate(list_actions):
                    verbs = np.where(action['verb_score'] > threshold)
                    s_bbox = action['subject_box']
                    o_bbox = action['object_box']
                    o_class = action['object_id']
                    vis_img = cv2.rectangle(vis_img, (int(s_bbox[0]),int(s_bbox[1])), (int(s_bbox[2]),int(s_bbox[3])), color_dict[0], 3)
                    print('drawing subject box')
                    vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                    print('drawing object box')
                    text_size, BaseLine=cv2.getTextSize(self.index_2_cat(verbs[0][0],args.inf_type),cv2.FONT_HERSHEY_SIMPLEX,1,2)
            
                    #text height for multiple verbs (interactions)
                    text_size_y = text_size[1] 
                    cnt = 1
                    for verb in verbs[0]:
                        text = self.index_2_cat(verb,args.inf_type)
                        text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                        if s_bbox[1]-cnt*text_size_y < 0 or s_bbox[0] < 0:
                            break

                        #draw text
                        text_box = [s_bbox[0], s_bbox[1]-cnt*text_size_y, s_bbox[0]+text_size[0],s_bbox[1]-(cnt-1)*text_size_y]
                        vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                        vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                        print(f'drawing action box : {text}, {cnt}')
                        cnt += 1 
        return vis_img
    
    def draw_img_all(self, img, output_i, top_k, threshold, color_dict, inf_type,color=False):
        
        vis_img = img.copy()
        if 'att' in output_i:
            # import pdb;pdb.set_trace()
            # print('asdsds')
            list_predictions = []
            for predict in output_i['att']['predictions']:

                #prediction threshold
                if predict['max_score'] < threshold:
                    continue
                object_id = predict['object_id']
                object_box = predict['bbox']
                attr_score = predict['pair_score']
                max_attr_score = predict['max_score']
                single_out = {'object_box':np.array(object_box), 'object_id':np.array(object_id), 'attr_score':np.array(attr_score), 'max_attr_score':np.array(max_attr_score)}
                list_predictions.append(single_out)
            
            if list_predictions:
                df = pd.DataFrame(list_predictions)
                df = df.sort_values(by=['max_attr_score'],ascending=False).iloc[:top_k]
                list_predictions = df.to_dict('records')
                for i, prediction in enumerate(list_predictions):

                    attributes = np.where(prediction['attr_score'] > threshold)
                    o_bbox = prediction['object_box']
                    o_class = prediction['object_id']                
                    
                    if color: #only color inference
                        for attr in attributes[0]:
                            if attr in self.color_index:
                                #if color index draw object box 
                                vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                                print(f'drawing object box')
                            else:
                                continue

                    else:
                        vis_img = cv2.rectangle(vis_img, (int(o_bbox[0]),int(o_bbox[1])), (int(o_bbox[2]),int(o_bbox[3])), color_dict[int(o_class)], 3)
                        print(f'drawing object box')
                    
                    text_size, BaseLine=cv2.getTextSize(self.index_2_cat(attributes[0][0],args.inf_type),cv2.FONT_HERSHEY_SIMPLEX,1,2)

                    #text height for multiple attributes
                    text_size_y = text_size[1] 
                    cnt = 1
                    for attr in attributes[0]:
                        if attr not in self.color_index:
                            continue
                        text = self.index_2_cat(attr,args.inf_type)
                            
                        text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                        if o_bbox[1]-cnt*text_size_y < 0 or o_bbox[0] < 0:
                            break
                        text_box = [o_bbox[0], o_bbox[1]-cnt*text_size_y, o_bbox[0]+text_size[0],o_bbox[1]-(cnt-1)*text_size_y]

                        #draw text
                        vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                        vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                        print(f'drawing attribute box : {text}, {cnt}')
                        cnt += 1 
        
        #for hoi multi head inference
        # elif ('vcoco' in inf_type) and ('hico' in inf_type):
        actions = {'hico':[],'vcoco':[]}
        output_hois=[]
        output_hois.append(output_i['hico'])
        output_hois.append(output_i['vcoco'])
        for output in output_hois:
            for box, hoi in zip(output['box_predictions'], output['hoi_predictions']):
                
                #prediction threshold
                if hoi['max_score'] < threshold:
                    continue
                
                subject_id = hoi['subject_id']
                subject_box = box['h_bbox']
                object_id = hoi['object_id']
                object_box = box['o_bbox']
                verb_score = hoi['pair_score']
                max_verb_score = hoi['max_score']
                single_out = {'subject_box':np.array(subject_box), 'object_box':np.array(object_box), 'subject_id':np.array(subject_id), 'object_id':np.array(object_id), 'verb_score':np.array(verb_score), 'max_verb_score':np.array(max_verb_score)}
                if single_out['verb_score'].shape == (117,): #hico
                    actions['hico'].append(single_out)
                elif single_out['verb_score'].shape == (29,): #vcoco
                    actions['vcoco'].append(single_out)
                
            if len(actions['hico']) > 0 and len(actions['vcoco']) > 0:
                hico,vcoco = pd.DataFrame(actions['hico']), pd.DataFrame(actions['vcoco'])
                hico,vcoco = hico.sort_values(by=['max_verb_score'],ascending=False).iloc[:top_k], vcoco.sort_values(by=['max_verb_score'],ascending=False).iloc[:top_k] 
                hico_actions = hico.to_dict('records')
                vcoco_actions = vcoco.to_dict('records')
                
                s_bbox = {}
                o_bbox = {}
                for i, action in enumerate(hico_actions):
                    self.count_dict[i] = 1
                    verbs = np.where(action['verb_score'] > threshold)
                    s_bbox[i] = action['subject_box']
                    o_bbox[i] = action['object_box']
                    o_class = action['object_id']
                    vis_img = cv2.rectangle(vis_img, (int(s_bbox[i][0]),int(s_bbox[i][1])), (int(s_bbox[i][2]),int(s_bbox[i][3])), color_dict[0], 3)
                    print('drawing subject box')
                    
                    vis_img = cv2.rectangle(vis_img, (int(o_bbox[i][0]),int(o_bbox[i][1])), (int(o_bbox[i][2]),int(o_bbox[i][3])), color_dict[int(o_class)], 3)
                    print('drawing object box')
                    text_size, BaseLine=cv2.getTextSize(self.index_2_cat(verbs[0][0],'hico'),cv2.FONT_HERSHEY_SIMPLEX,1,2)
                    
                    #text height for multiple verbs (interactions)
                    text_size_y = text_size[1] 
                    
                    for verb in verbs[0]:    
                        text = self.index_2_cat(verb,'hico')
                        text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                        if s_bbox[i][1]-self.count_dict[i]*text_size_y < 0 or s_bbox[i][0] < 0:
                            break
                        text_box = [s_bbox[i][0], s_bbox[i][1]-self.count_dict[i]*text_size_y, s_bbox[i][0]+text_size[0],s_bbox[i][1]-(self.count_dict[i]-1)*text_size_y]

                        #draw text
                        vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                        vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                        print('drawing hico action box')
                        print(f'pair number : {i}, action : {text}, action number : {self.count_dict[i]}')
                        self.count_dict[i] += 1 

                for i, action in enumerate(vcoco_actions):
                    verbs = np.where(action['verb_score'] > threshold)
                    for verb in verbs[0]:
                        text = self.index_2_cat(verb,'vcoco')
                        text_size, BaseLine=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,2)
                        
                        #no subject box case
                        if i not in s_bbox:
                            break

                        if (s_bbox[i][1]-self.count_dict[i]*text_size_y < 0 or s_bbox[i][0] < 0):
                            break

                        text_box = [s_bbox[i][0], s_bbox[i][1]-self.count_dict[i]*text_size_y, s_bbox[i][0]+text_size[0],s_bbox[i][1]-(self.count_dict[i]-1)*text_size_y]

                        #draw vcoco verb text
                        vis_img = cv2.rectangle(vis_img, (int(text_box[0]),int(text_box[1])),(int(text_box[2]),int(text_box[3])), color_dict[int(o_class)], -1)
                        vis_img = cv2.putText(vis_img, text, (int(text_box[0]),int(text_box[3])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
                        print('drawing vcoco action box') 
                        print(f'pair number : {i}, action : {text}, action number : {self.count_dict[i]}')
                        self.count_dict[i] += 1 

        
        
        return vis_img
    
    def save_video_orig(self, args):
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))	
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        orig_size = torch.as_tensor([frame_height,frame_width]).unsqueeze(0).to('cuda')
        output_file = cv2.VideoWriter(self.output_dir, self.fourcc, self.fps, frame_size)
        checkpoint = torch.load(self.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'],strict=False)
        model.to(device)
        color_dict = self.make_color_dict(self.num_obj_classes)
        while(True):

            retval, frame = self.cap.read() 
            print(f'frame_num : {self.frame_num}')

            if not retval:
                break

            if 'hico' in self.inf_type and 'vcoco' in self.inf_type:
                outputs_hico, outputs_vcoco = self.inference_for_vid(model, frame, args)
                preds = []
                results_hico = postprocessors(outputs_hico, orig_size)
                results_vcoco = postprocessors(outputs_vcoco, orig_size)
                preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_hico))))
                preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_vcoco))))
                output_hico = self.change_format(preds[0], args)
                output_vcoco = self.change_format(preds[1], args)
                output_i = []
                output_i.append(output_hico)
                output_i.append(output_vcoco)
                vis_img = self.draw_img(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type)
                output_file.write(vis_img)
                self.frame_num += 1

            else:
                outputs = self.inference_for_vid(model, frame, args)
                preds = []
                results = postprocessors(outputs, orig_size)
                preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
                output_i = self.change_format(preds[0], args)

                if args.color and (args.inf_type == 'vaw'):
                    vis_img = self.draw_img(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type, color=args.color)
                else:
                    vis_img = self.draw_img(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type)
                output_file.write(vis_img)
                self.frame_num += 1

        self.cap.release()
        output_file.release()
        cv2.destroyAllWindows()

    def save_video(self, args):
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))	
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        orig_size = torch.as_tensor([frame_height,frame_width]).unsqueeze(0).to('cuda')
        output_file = cv2.VideoWriter(self.output_dir, self.fourcc, self.fps, frame_size)
        checkpoint = torch.load(self.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'],strict=False)
        model.to(device)
        color_dict = self.make_color_dict(self.num_obj_classes)
        # import pdb;pdb.set_trace()
        while(True):
            retval, frame = self.cap.read() 
            print(f'frame_num : {self.frame_num}')
            
            if not retval:
                break
            if 'vcoco' in self.inf_type and 'vaw' in self.inf_type and 'hico' in self.inf_type:
                outputs_hico, outputs_vcoco, outputs_vaw = self.inference_for_vid(model, frame, args)
                preds = []
                results_hico = postprocessors(outputs_hico, orig_size)
                results_vcoco = postprocessors(outputs_vcoco, orig_size)
                results_vaw = postprocessors(outputs_vaw, orig_size)
                preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_vcoco))))
                preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_hico))))
                preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_vaw))))
                output_vcoco = self.change_format_all(preds[0], args)
                output_hico = self.change_format_all(preds[1], args)
                output_vaw = self.change_format_all(preds[2], args)
                output_i = {}
                output_i['vcoco'] = output_vcoco
                output_i['hico'] = output_hico
                output_i['att'] = output_vaw
                # import pdb;pdb.set_trace()
                vis_img = self.draw_img_all(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type, color=args.color)
                if args.webcam:
                    cv2.namedWindow('Online Demo', cv2.WINDOW_NORMAL)
                    cv2.imshow('Online Demo',vis_img)
                    if cv2.waitKey(1)==27:
                        break
                else:
                    output_file.write(vis_img)
                self.frame_num += 1
            else:
                if 'hico' in self.inf_type and 'vcoco' in self.inf_type:
                    outputs_hico, outputs_vcoco = self.inference_for_vid(model, frame, args)
                    preds = []
                    results_hico = postprocessors(outputs_hico, orig_size)
                    results_vcoco = postprocessors(outputs_vcoco, orig_size)
                    preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_hico))))
                    preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_vcoco))))
                    output_hico = self.change_format(preds[0], args)
                    output_vcoco = self.change_format(preds[1], args)
                    output_i = []
                    output_i.append(output_hico)
                    output_i.append(output_vcoco)
                    vis_img = self.draw_img(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type)
                    if args.webcam:
                        cv2.namedWindow('Online Demo', cv2.WINDOW_NORMAL)
                        cv2.imshow('Online Demo',vis_img)
                        if cv2.waitKey(1)==27:
                            break
                    else:
                        output_file.write(vis_img)
                    self.frame_num += 1
                elif 'hico' in self.inf_type and 'vaw' in self.inf_type:
                    outputs_hico, outputs_vaw = self.inference_for_vid(model, frame, args)
                    preds = []
                    results_hico = postprocessors(outputs_hico, orig_size)
                    results_vcoco = postprocessors(outputs_vcoco, orig_size)
                    preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_hico))))
                    preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_vcoco))))
                    output_hico = self.change_format(preds[0], args)
                    output_vcoco = self.change_format(preds[1], args)
                    output_i = []
                    output_i.append(output_hico)
                    output_i.append(output_vcoco)
                    vis_img = self.draw_img_all(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type)
                    if args.webcam:
                        cv2.namedWindow('Online Demo', cv2.WINDOW_NORMAL)
                        cv2.imshow('Online Demo',vis_img)
                        if cv2.waitKey(1)==27:
                            break
                    else:
                        output_file.write(vis_img)
                    self.frame_num += 1
                elif 'vcoco' in self.inf_type and 'vaw' in self.inf_type:
                    outputs_vcoco, outputs_vaw = self.inference_for_vid(model, frame, args)
                    preds = []
                    results_vcoco = postprocessors(outputs_vcoco, orig_size)
                    results_vaw = postprocessors(outputs_vaw, orig_size)
                    preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_vcoco))))
                    preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results_vaw))))
                    output_vcoco = self.change_format(preds[0], args)
                    output_vaw = self.change_format(preds[1], args)
                    output_i = {}
                    output_i['hoi'] = output_vcoco
                    output_i['att'] = output_vaw
                    vis_img = self.draw_img_all(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type)
                    if args.webcam:
                        cv2.namedWindow('Online Demo', cv2.WINDOW_NORMAL)
                        cv2.imshow('Online Demo',vis_img)
                        if cv2.waitKey(1)==27:
                            break
                    else:
                        output_file.write(vis_img)
                    self.frame_num += 1


                else:
                    outputs = self.inference_for_vid(model, frame, args)
                    preds = []
                    results = postprocessors(outputs, orig_size)
                    preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
                    output_i = self.change_format(preds[0], args)
                    vis_img = self.draw_img(frame,output_i,top_k=args.top_k,threshold=args.threshold,color_dict=color_dict,inf_type=args.inf_type)
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
