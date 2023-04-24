import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch
import wandb

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
#from datasets.vaw_eval import VAWEvaluator
from datasets.vaw_evaluator import Evaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, log: bool = False, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # else:
    #     metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    # print_freq = int(len(data_loader)/10)
    print_freq = 10
    for samples,targets in metric_logger.log_every(data_loader, print_freq, header):
        dataset_name = targets[0]['dataset']
        assert len(set([t['dataset'] for t in targets]))==1
        samples = samples.to(device)
        targets = [{k: v.to(device)  if type(v)!=str and k != 'masks' and k != 'polygons' and k != 'clip_em' and k != 'object_name' else v for k, v in t.items()} for t in targets]
        dtype=targets[0]['type']
        dataset=targets[0]['dataset']
        outputs = model(samples,targets,dtype,dataset,args)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # if dataset == 'vaw':
        #     import pdb; pdb.set_trace()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # print(len(loss_dict_reduced_scaled))
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # if dataset == 'vcoco':
        #     import pdb; pdb.set_trace()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # else:
        #     metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if utils.get_rank() == 0 and log: wandb.log(loss_dict_reduced_scaled)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             )
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_hoi_att(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args=None):
    model.eval()
    eval_mode = True
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    preds = []
    gts = []
    dataset_name = os.fspath(data_loader.dataset.img_folder)

    if 'vaw' in dataset_name:
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            dtype = targets[0]['type'] 
            dataset=targets[0]['dataset'] 
            samples = samples.to(device)
            outputs = model(samples,targets,dtype,dataset,args)['pred_logits'].sigmoid()    
            preds.extend(outputs.detach().cpu().numpy())
        
        preds = np.array(preds) #(31819, 620)
        annos = np.load(args.vaw_gts) #(31819, 620)

        evaluator = Evaluator(
            args.fpath_attribute_index, args.fpath_attribute_types,
            args.fpath_attribute_parent_types, args.fpath_head_tail)
        
        scores_overall, scores_per_class = evaluator.evaluate(preds, annos)
        scores_overall_topk, scores_per_class_topk = evaluator.evaluate(
            preds, annos, threshold_type='topk')

        CATEGORIES = ['all', 'head', 'medium', 'tail', 'color', 'material', 'shape', 'size', 'action', 'state', 'texture', 'other'] + list(evaluator.attribute_parent_type.keys())
        stats = {}
        for category in CATEGORIES:
            stats['mAP_'+category] = scores_per_class[category]['ap']

            #per class
            stats['recall_'+category] = scores_per_class[category]['recall']
            stats['precision_'+category] = scores_per_class[category]['precision']
            stats['f1_'+category] = scores_per_class[category]['f1']
            stats['bacc_'+category] = scores_per_class[category]['bacc']

            #top_k score
            stats['topk_recall_'+category] = scores_per_class_topk[category]['recall'] #['recall', 'precision', 'f1']
            stats['topk_precision_'+category] = scores_per_class_topk[category]['precision'] #['recall', 'precision', 'f1']
            stats['topk_f1_'+category] = scores_per_class_topk[category]['f1'] #['recall', 'precision', 'f1']



        with open(args.output_dir+'/class_AP.txt', 'w') as f:
            f.write('| {:<18}| AP\t\t| Recall@K\t| B.Accuracy\t| N_Pos\t| N_Neg\t|\n'.format('Name'))
            f.write('-----------------------------------------------------------------------------------------------------\n')
            for i_class in range(evaluator.n_class):
                att = evaluator.idx2attr[i_class]
                f.write('| {:<18}| {:.4f}\t| {:.4f}\t| {:.4f}\t\t| {:<6}| {:<6}|\n'.format(
                    att,
                    evaluator.get_score_class(i_class).ap,
                    evaluator.get_score_class(i_class, threshold_type='topk').get_recall(),
                    evaluator.get_score_class(i_class).get_bacc(),
                    evaluator.get_score_class(i_class).n_pos,
                    evaluator.get_score_class(i_class).n_neg))


        return stats, dataset_name

    
    elif 'hico' in dataset_name or 'v-coco' in dataset_name:
        #indices = []
        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            dtype = targets[0]['type'] 
            dataset=targets[0]['dataset'] 
            samples = samples.to(device)
            outputs = model(samples,None,dtype,dataset) #outputs['pred_logits'].shape : torch.Size([8, 620])
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors(outputs, orig_target_sizes)
            preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))            

            # For avoiding a runtime error, the copy is used
            gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        img_ids = [img_gts['id'] for img_gts in gts]
        _, indices = np.unique(img_ids, return_index=True)
        preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
        gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

        if 'hico' in dataset_name:
            evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                    data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat ,args.max_pred)    
            stats = evaluator.evaluate()
            return stats, dataset_name

        elif 'v-coco' in dataset_name:
            evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat ,args.max_pred)
            stats = evaluator.evaluate()
            return stats, dataset_name
