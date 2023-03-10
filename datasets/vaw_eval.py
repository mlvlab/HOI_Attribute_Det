import numpy as np
from collections import defaultdict
import copy
import json

class VAWEvaluator():
    def __init__(self, preds, gts, subject_category_id, rare_triplets,non_rare_triplets, valid_masks, max_pred):
        self.overlap_iou = 0.5
        self.max_attrs = max_pred

        # self.head = head
        # self.medium = medium
        # self.tail = tail
        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_double = []
        self.valid_masks = valid_masks
        self.preds = []

        self.ggg = gts
        self.gts = []
        
        for img_gts in gts:
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k not in ['id','type','dataset'] }
            self.gts.append({'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])]
                            ,'attr_annotation':[]})
            #import pdb; pdb.set_trace()
            for i,attr in enumerate(img_gts['pos_att_classes']): 
                #attr.shape : (620,0)

                #attr_idxs : array([412])
                attr_idxs = np.nonzero(attr==1)[0]
                for j in attr_idxs: #j:412
                    if self.valid_masks[j]==1: #valid_masks에 attr_idx가 있으면
                        # self.gts.append({
                        #     'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_gts['boxes'], img_gts['labels'])],
                        #     'attr_annotation': [{ 'object_id': i, 'category_id': j}]
                        # })
                        self.gts[-1]['attr_annotation'].append({'object_id': i, 'category_id': j}) #i : image 내 object index
                    
            for attr in self.gts[-1]['attr_annotation']: #[{'object_id': 0, 'category_id': 412}]
                double = (attr['category_id']) #(412)

                if double not in self.gt_double:
                    self.gt_double.append(double) #[412]

                self.sum_gts[double] += 1 # #defaultdict(<function VAWEvaluator.__init__.<locals>.<lambda> at 0x7efc5650a9d0>,{412: 1, 232: 1, 259: 1})

        for i, img_preds in enumerate(preds):
            import pdb; pdb.set_trace()
            #len(preds) : 3452
            #img_preds.keys() : 'labels', 'boxes', 'attr_scores', 'obj_ids'
            #len(img_preds['labels']),len(img_preds['boxes']),len(img_preds['attr_scores']),len(img_preds['obj_ids']) : 100
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items() if k != 'att_recognition_time'}

            #len(bboxes) : 100
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]

            #attr_scores.shape : (100,620)
            '''
            array([[0.02596947, 0.0583656 , 0.03330475, ..., 0.03529264, 0.48919117,
                0.6836143 ],
            [0.04019799, 0.043267  , 0.03159507, ..., 0.04269576, 0.44529814,
                0.6828197 ],
            [0.03269851, 0.05208412, 0.03599585, ..., 0.04921223, 0.49292317,
                0.7622846 ],
            ...,
            [0.04096896, 0.05102941, 0.03148905, ..., 0.04965477, 0.53773934,
                0.6137796 ],
            [0.03571606, 0.05448545, 0.03412009, ..., 0.04537631, 0.5010768 ,
                0.6937183 ],
            [0.03066489, 0.04988421, 0.03673794, ..., 0.03909696, 0.5097801 ,
                0.62213   ]], dtype=float32)
            '''
            attr_scores = img_preds['attr_scores']

            #attr_label.shape : (100,620)
            '''
            array([[  0,   1,   2, ..., 617, 618, 619],
            [  0,   1,   2, ..., 617, 618, 619],
            [  0,   1,   2, ..., 617, 618, 619],
            ...,
            [  0,   1,   2, ..., 617, 618, 619],
            [  0,   1,   2, ..., 617, 618, 619],
            [  0,   1,   2, ..., 617, 618, 619]])
            '''
            attr_labels = np.tile(np.arange(attr_scores.shape[1]), (attr_scores.shape[0], 1))
            
            #object_ids.shape : (100, 620)
            '''
            array([[ 0,  0,  0, ...,  0,  0,  0],
            [ 1,  1,  1, ...,  1,  1,  1],
            [ 2,  2,  2, ...,  2,  2,  2],
            ...,
            [97, 97, 97, ..., 97, 97, 97],
            [98, 98, 98, ..., 98, 98, 98],
            [99, 99, 99, ..., 99, 99, 99]])
            '''
            object_ids = np.tile(img_preds['obj_ids'], (attr_scores.shape[1], 1)).T

            #(62000,)
            attr_scores = attr_scores.ravel()
            
            #(62000,)
            attr_labels = attr_labels.ravel()
            
            #(62000,)
            object_ids = object_ids.ravel()

            #valid_masks.shape : (620,)
            #valid_masks[attr_labels].shape : (62000,)
            #array([1., 1., 1., ..., 1., 0., 0.])
            #masks.shape : (62000,)
            masks = valid_masks[attr_labels]

            #attr_scores.shape : (62000,)
            '''
            array([0.02596947, 0.0583656 , 0.03330475, ..., 0.03909696, 0.        ,
              0.        ], dtype=float32)
            '''
            attr_scores *= masks

            #len(attrs) : 62000
            attrs = [{'object_id': object_id, 'category_id': category_id, 'score': score} for
                    object_id, category_id, score in zip(object_ids, attr_labels, attr_scores)]
            
            
            attrs.sort(key=lambda k: (k.get('score', 0)), reverse=True)

            #len(attrs) : 100
            '''
            attrs[0]
            {'object_id': 76, 'category_id': 49, 'score': 0.29883718}
            '''
            attrs = attrs[:self.max_attrs]
            
            self.preds.append({
                'predictions': bboxes, #{'bbox': array([2.3223832e-01, 2.0291336e+02, 8.1720352e+01, 3.4907608e+02],dtype=float32), 'category_id'(object_id): 56}
                'attr_prediction': attrs #{'object_id'(query_id): 76, 'category_id'(attribute_id): 49, 'score': 0.29883718}
            })
    def evaluate(self):
        count_dict = dict()
        #import pdb;pdb.set_trace()
        for img_id, (img_preds, img_gts) in enumerate(zip(self.preds, self.gts)):#len(self.preds):3452, len(self.gts):3452
            
            if img_gts['attr_annotation']:
                for annotation in img_gts['attr_annotation']:
                    attribute_idx = str(annotation['category_id'])
                    if attribute_idx in count_dict.keys():
                        count_dict[attribute_idx] += 1
                    else:
                        count_dict[attribute_idx] = 1
                #import pdb;pdb.set_trace()
            print(f"Evaluating Score Matrix... : [{(img_id+1):>4}/{len(self.gts):<4}]" ,flush=True, end="\r")
            
            '''
            img_preds.keys()
            dict_keys(['predictions', 'attr_prediction'])
            img_gts.keys()
            dict_keys(['annotations', 'attr_annotation'])
            '''
            pred_bboxes = img_preds['predictions'] #(박스와 object category에 대한 예측), 길이 100
            gt_bboxes = img_gts['annotations'] #(박스와 object category에 대한 gt), 길이 1
            pred_attrs = img_preds['attr_prediction'] #(쿼리id, attribute category, attribute score), 길이 100
            gt_attrs = img_gts['attr_annotation'] #(image 내 object index, attribute category에 대한 gt), 길이 1 : [{'object_id': 0, 'category_id': 412}]
            #prediction box to gt for sanity check
            #pred_bboxes = img_preds['predictions']

            #pred_bboxes = np.tile(img_gts['annotations'],reps=(len(img_preds['predictions'])))
            # pred_attrs_score = [{'score':item['score']} for item in img_preds['attr_prediction']]
            # pred_attrs_tiled = np.tile(img_gts['attr_annotation'],reps=(len(img_preds['attr_prediction'])))
            # pred_attrs = []
            # for score, attr in zip(pred_attrs_score, pred_attrs_tiled):
            #     attr['score'] = score['score']
            #     pred_attrs.append(copy.deepcopy(attr))
                
            #import pdb;pdb.set_trace()
            if len(gt_bboxes) != 0:

                #iou 기반 bbox_pair를 추려낸다. 
                #bbox_pairs : {32: [0], 52: [0], 54: [0], 65: [0]}
                #bbox_overlaps : {32: [0.6335495859634046], 52: [0.5193357826044028], 54: [0.5932722466722482], 65: [0.78254467186614]}
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_attrs, gt_attrs, bbox_pairs, pred_bboxes, bbox_overlaps)
            else:
                for pred_attr in pred_attrs:
                    double = [pred_attr['category_id']]
                    if double not in self.gt_double:
                        continue
                    self.tp[double].append(0)
                    self.fp[double].append(1)
                    self.score[double].append(pred_attr['score'])

        with open('count_dict.json', 'w') as f:
            json.dump(count_dict, f)
        #import pdb;pdb.set_trace()
        print(f"[stats] Score Matrix Generation completed!!")
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        #import pdb; pdb.set_trace()
        for double in self.gt_double:
            sum_gts = self.sum_gts[double]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[double]))
            fp = np.array((self.fp[double]))
            if len(tp) == 0:
                ap[double] = 0
                max_recall[double] = 0
                if double in self.rare_triplets:
                    rare_ap[double] = 0
                elif double in self.non_rare_triplets:
                    non_rare_ap[double] = 0
                else:
                    print('Warning: triplet {} is neither in rare double nor in non-rare double'.format(double))
                continue

            score = np.array(self.score[double])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            ap[double] = self.voc_ap(rec, prec)
            max_recall[double] = np.amax(rec)
            if double in self.rare_triplets:
                rare_ap[double] = ap[double]
            elif double in self.non_rare_triplets:
                non_rare_ap[double] = ap[double]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(double))
        
        # ap_dict = {str(k):v for k,v in dict(ap).items()}
        # rare_ap_dict = {str(k):v for k,v in dict(rare_ap).items()}
        # non_rare_ap_dict = {str(k):v for k,v in dict(non_rare_ap).items()}
        # max_recall_dict = {str(k):v for k,v in dict(max_recall).items()}
        
        #import pdb; pdb.set_trace()
        
        # with open('ap.json', 'w') as f:
        #     json.dump(dict(ap_dict), f)
        
        # with open('rare_ap.json', 'w') as f:
        #     json.dump(dict(rare_ap_dict), f)
        
        # with open('non_rare_ap.json', 'w') as f:
        #     json.dump(dict(non_rare_ap_dict), f)

        # with open('max_recall.json', 'w') as f:
        #     json.dump(dict(max_recall_dict), f)
        
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))

        print('--------------------')
        print('mAP: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare, m_max_recall))
        print('--------------------')

        return {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_attrs, gt_attrs, match_pairs, pred_bboxes, bbox_overlaps):
        #match_pairs(bbox_pairs) : {32: [0], 52: [0], 54: [0], 65: [0]}
        #bbox_overlaps : {32: [0.6335495859634046], 52: [0.5193357826044028], 54: [0.5932722466722482], 65: [0.78254467186614]}
        #match_pairs에는 gt_boxes와 IoU가 일정수준 이상인 prediction index가 key 값으로 들어있다. 
        
        pos_pred_ids = match_pairs.keys() #{32: [0], 52: [0], 54: [0], 65: [0]}
        vis_tag = np.zeros(len(gt_attrs)) #len(gt_attrs):1 

        #score 순으로 sorting
        pred_attrs.sort(key=lambda k: (k.get('score', 0)), reverse=True) #len(pred_attrs):100
        #import pdb;pdb.set_trace()

        if len(pred_attrs) != 0:
            for pred_attr in pred_attrs:#len(pred_attrs) : 100           
                is_match = 0
                
                #pos_pred_ids에는 GT box와의 IOU가 일정수준 이상인 query index(object id)이 들어있음 
                if len(match_pairs) != 0 and pred_attr['object_id'] in pos_pred_ids:
                    
                    #object prediction id가 positive(IOU 일정 수준 이상)인 pred_attr 애들만 선별
                    #{'object_id': 65, 'category_id': 179, 'score': 0.24628997}
                    pred_obj_ids = match_pairs[pred_attr['object_id']] #[0]                    
                    pred_obj_overlaps = bbox_overlaps[pred_attr['object_id']] #해당 query 예측 값의 overlap 값 
                    pred_category_id = pred_attr['category_id'] #attribute id
                    max_overlap = 0
                    max_gt_attr = 0

                    #import pdb; pdb.set_trace()
                    for gt_attr in gt_attrs: #gt_attrs 여러개 있을때 1개 씩

                        #gt object_id가 pred_obj_ids에 있고, prediction category가 gt_attr category와 일치 할 때 
                        if gt_attr['object_id'] in pred_obj_ids and pred_category_id == gt_attr['category_id']:
                            is_match = 1

                            #gt_attr['object_id'] 값을 갖고 있는 index에 있는 overlap(IoU) 값
                            min_overlap_gt = pred_obj_overlaps[pred_obj_ids.index(gt_attr['object_id'])]

                            #overlap 되는게 가장 큰 gt 찾기
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt

                                #overlap 크기가 가장 큰 gt_attr
                                max_gt_attr = gt_attr
                
                #예측 attribute idx
                double = (pred_attr['category_id'])
                
                #import pdb; pdb.set_trace()
                
                if double not in self.gt_double:
                    continue
                
                if is_match == 1 and vis_tag[gt_attrs.index(max_gt_attr)] == 0:
                    self.fp[double].append(0) #fp double index에 0 추가
                    self.tp[double].append(1) #tp double index에 1 추가 
                    vis_tag[gt_attrs.index(max_gt_attr)] = 1
                
                else:
                    self.fp[double].append(1) #fp double index에 1 추가
                    self.tp[double].append(0) #tp double index에 0 추가 
                self.score[double].append(pred_attr['score'])
        #import pdb; pdb.set_trace()


    def compute_iou_mat(self, bbox_list1, bbox_list2): #(gt_boxes, pred_boxes) -> (1,4), (100,4)
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2))) #(1, 100)
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1): #i : 0
            for j, bbox2 in enumerate(bbox_list2): #j : 0~99
                iou_i = self.compute_IOU(bbox1, bbox2) #예측값과 gt_box의 IOU 계산
                iou_mat[i, j] = iou_i

        #iou_mat : gt box와 pred box 간 iou matrix 
        iou_mat_ov=iou_mat.copy()

        #IoU 값이 특정 overlap_iou 값 보다 크면 1 할당 아니면 0 할당 
        iou_mat[iou_mat>=self.overlap_iou] = 1
        iou_mat[iou_mat<self.overlap_iou] = 0

        #np.nonzero -> 요소들 중 0이 아닌 값을의 인덱스 반환 
        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]): #gt와의 iou가 특정 값 이상인 prediction index 
                if pred_id not in match_pairs_dict.keys(): #해당 prediction index를 match_pairs_dict, match_pair_overlaps의 key값에 추가 
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i]) #match_pairs의 value를 해당 prediciton index key에 있는 list에 추가 
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i],pred_id]) #겹치는 pair value append
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        # if bbox1['category_id'] == bbox2['category_id']:
        rec1 = bbox1['bbox']
        rec2 = bbox2['bbox']
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
        S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
            return intersect / (sum_area - intersect)
        # else:
        #     return 0
