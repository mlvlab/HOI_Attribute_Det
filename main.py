import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, evaluate_hoi_att
from models import build_model
from torch.utils.data.dataset import ConcatDataset
from util.sampler import BatchSchedulerSampler, ComboBatchSampler
import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr_drop', default=20, type=int)
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

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=81,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')
                        
    #vaw evaluation
    parser.add_argument('--vaw_gts', default='data/vaw/annotations/test_orig_vaw.npy')
    parser.add_argument('--fpath_attribute_index', type=str,
                        default='data/vaw/annotations/attribute_index.json') #o
    parser.add_argument('--fpath_attribute_types', type=str,
                        default='data/vaw/annotations/attribute_types.json') #o
    parser.add_argument('--fpath_attribute_parent_types', type=str,
                        default='data/vaw/annotations/attribute_parent_types.json') #o
    parser.add_argument('--fpath_head_tail', type=str,
                        default='data/vaw/annotations/head_tail.json') #o


    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Attribute detection
    parser.add_argument('--att_det', action='store_true',
                        help="Train Attribute Detection head if the flag is provided")
    
    # Attribute detection coeff
    parser.add_argument('--att_idx_loss_coef', default=1, type=float)
    parser.add_argument('--att_loss_coef', default=1, type=float)
    parser.add_argument('--set_cost_att', default=1, type=float,
                    help="Action coefficient in the matching cost")

    # * ATT Detection                       
    parser.add_argument('--num_att_classes', default=620, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--att_enc_layers', default=1, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--att_dec_layers', default=1, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--att_nheads', default=8, type=int,
                        help="Number of decoding layers in HOI transformer")
    parser.add_argument('--att_dim_feedforward', default=2048, type=int,
                        help="Number of decoding layers in HOI transformer")
    # parser.add_argument('--hoi_mode', type=str, default=None, help='[inst | pair | all]')
    parser.add_argument('--num_att_queries', default=100, type=int,
                        help="Number of Queries for Interaction Decoder")
    parser.add_argument('--att_aux_loss', action='store_true')
    parser.add_argument('--update_obj_att', action='store_true')
    parser.add_argument('--num_obj_att_classes', type=int, default=1,
                        help="Number of object classes")

    #masked roi align
    parser.add_argument('--masking', action='store_true')
    parser.add_argument('--test_polygon', default='data/vaw/annotations/test_vaw_polygon.npy')

    # mtl
    parser.add_argument('--mtl', action='store_true')
    parser.add_argument('--mtl_data', type=utils.arg_as_list,default=[],
                            help='[hico,vcoco,vaw]')
    parser.add_argument('--num_hico_verb_classes', type=int, default=117,
                    help="Number of verb hico classes")
    parser.add_argument('--num_vcoco_verb_classes', type=int, default=29,
                    help="Number of verb coco classes")

    #eval
    parser.add_argument('--max_pred', default=100, type=int)
    

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--data_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')


    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    #backbone roi_align flag
    parser.add_argument('--br', action='store_true')

    #freeze transformer (pretrained hoi parmeters)
    parser.add_argument('--freeze_hoi', action='store_true')


    #feature map vis 
    parser.add_argument('--img_path', default='samples/2361755.jpg')
    parser.add_argument('--vis_img', action='store_true')

    #mask prediction mode 
    parser.add_argument('--predict_mask', action='store_true')

    #masking argument
    parser.add_argument('--output_masking', action='store_true')
    parser.add_argument('--input_masking', action='store_true')

    #object embedding argument
    parser.add_argument('--object_embedding', action='store_true')


    #fc version
    parser.add_argument('--fc_version', action='store_true')

    # logging
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project_name', default='qpic')
    parser.add_argument('--group_name', default='Neubla')
    parser.add_argument('--run_name', default='train_num_1')

    #for video vis
    parser.add_argument('--show_vid', action='store_true',help='check video inference')
    parser.add_argument('--video_file', default='video/example2.mp4',help='video source')
    parser.add_argument('--checkpoint', default='checkpoints/hoi/checkpoint.pth',help='model checkpoint path')
    parser.add_argument('--inf_type', default='vcoco',help='inference type')
    parser.add_argument('--top_k', default=1,type=int,help='top_k value')
    parser.add_argument('--threshold', default=0.3,type=float,help='threshold value')
    parser.add_argument('--fps', default=30,type=int,help='fps')
    parser.add_argument('--all', action='store_true',help='check hoi+attribute inference')
    parser.add_argument('--color', action='store_true',help='only color inference for vaw')
    parser.add_argument('--webcam', default='', type=str)
    parser.add_argument('--vis_demo',action='store_true')
    parser.add_argument('--iou_threshold', default=0.9,type=float,help='iou threshold value')
    parser.add_argument('--attr_threshold', default=0.4,type=float)
    parser.add_argument('--attr_topk', default=1,type=int)


    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.mtl:
        from torch.utils.data.dataset import ConcatDataset
        from util.mtl_loader import MultiTaskDataLoader,CombinationDataset
        from pytorch_lightning.trainer.supporters import CombinedLoader
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
      
        if 'vaw' in args.mtl_data:
            
            args.num_att_classes = dataset_train[-1].num_attributes() 

        if args.distributed:            
            sampler_train = [torch.utils.data.DistributedSampler(d) for d in dataset_train]
            sampler_val = [torch.utils.data.DistributedSampler(dv,shuffle=False) for dv in dataset_val]

        else:
            sampler_train = [torch.utils.data.RandomSampler(d)for d in dataset_train]
            sampler_val = [torch.utils.data.SequentialSampler(dv) for dv in dataset_val]

        batch_sampler_train = ComboBatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(CombinationDataset(dataset_train),
                                        batch_sampler = batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        data_loader_val = [DataLoader(dv, args.batch_size, sampler=sv,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) for dv,sv in zip(dataset_val,sampler_val)]
       

    else:
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)

        if args.att_det and args.dataset_file=='vaw':
            args.num_att_classes = dataset_train.num_attributes() 

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    
    # if args.freeze_hoi:
    #     import pdb; pdb.set_trace()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if not args.hoi and not args.att_det:
        if args.dataset_file == "coco_panoptic":
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = datasets.coco.build("val", args)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(dataset_val)

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

    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'],strict=False)

    if args.eval:
        if args.hoi or args.att_det or args.mtl:
            if args.mtl:
                for dlv in data_loader_val:
                    test_stats,dataset_name = evaluate_hoi_att(args.dataset_file, model, postprocessors, dlv, args.subject_category_id, device, args)
                    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
                    if 'v-coco' in dataset_name:
                        if args.output_dir and utils.is_main_process():
                            with (output_dir / "log.txt").open("a") as f:
                                f.write(json.dumps(log_stats) + "\n")

                        if utils.get_rank() == 0 and args.wandb:
                    
                            wandb.log({
                                'mAP_all': test_stats['mAP_all'],
                                'mAP_thesis':test_stats['mAP_thesis']
                            })
                        performance=test_stats['mAP_thesis']

                    elif 'hico' in dataset_name:
                        if args.output_dir and utils.is_main_process():
                            with (output_dir / "log.txt").open("a") as f:
                                f.write(json.dumps(log_stats) + "\n")

                        if utils.get_rank() == 0 and args.wandb:
                            wandb.log({
                                'mAP': test_stats['mAP'],
                                'mAP rare': test_stats['mAP rare'],
                                'mAP non-rare':test_stats['mAP non-rare'],
                                'mean max recall':test_stats['mean max recall']
                            })
                        performance=test_stats['mAP']

                    elif 'vaw' in dataset_name:
                        #CATEGORIES = ['all', 'head', 'medium', 'tail'] 
                        if args.output_dir and utils.is_main_process():
                            with (output_dir / "log.txt").open("a") as f:
                                f.write(json.dumps(log_stats) + "\n")

                        if utils.get_rank() == 0 and args.wandb:
                            wandb.log({
                                'vaw_mAP_all': test_stats['mAP_all'],
                                'vaw_mAP_head': test_stats['mAP_head'],
                                'vaw_mAP_medium':test_stats['mAP_medium'],
                                'vaw_mAP_tail':test_stats['mAP_tail']
                            })



                    # elif 'vaw' in dataset_name:
                    #     if utils.get_rank() == 0 and args.wandb:
                    #         wandb.log({
                    #             'mAP': test_stats['mAP'],
                    #             'mAP rare': test_stats['mAP rare'],
                    #             'mAP non-rare':test_stats['mAP non-rare'],
                    #             'mean max recall':test_stats['mean max recall']
                    #         })
                    #     performance=test_stats['mAP']
                    # coco_evaluator = None
            else:
                test_stats,dataset_name = evaluate_hoi_att(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args)
                if 'v-coco' in dataset_name:
                    if utils.get_rank() == 0 and args.wandb:
                
                        wandb.log({
                            'mAP_all': test_stats['mAP_all'],
                            'mAP_thesis':test_stats['mAP_thesis']
                        })
                    performance=test_stats['mAP_thesis']
                elif 'hico' in dataset_name:
                    if utils.get_rank() == 0 and args.wandb:
                        wandb.log({
                            'mAP': test_stats['mAP'],
                            'mAP rare': test_stats['mAP rare'],
                            'mAP non-rare':test_stats['mAP non-rare'],
                            'mean max recall':test_stats['mean max recall']
                        })
                    performance=test_stats['mAP']
                elif 'vaw' in dataset_name:
                    if utils.get_rank() == 0 and args.wandb:
                        wandb.log({
                            'mAP': test_stats['mAP'],
                            # 'mAP rare': test_stats['mAP rare'],
                            # 'mAP non-rare':test_stats['mAP non-rare'],
                            'mean max recall':test_stats['mean max recall']
                        })
                    performance=test_stats['mAP']
                coco_evaluator = None
            return 
        else:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return


    if args.freeze_hoi:
        print('freeze hoi parameters')
        for name, param in model.named_parameters():
            if 'attribute' not in name:
                param.requires_grad = False

        # trainable_parameters = []
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         trainable_parameters.append(name)
        
        # print(trainable_parameters)
        # exit()

    if args.wandb:
        os.environ['WANDB_API_KEY'] = '3f007fbea1ce3a73739cf5da719f6d4fe7c91960'

    # add argparse
    if args.wandb and utils.get_rank() == 0:
        wandb.init(
            project=args.project_name,
            group=args.group_name,
            name=args.run_name,
            config=args
        )
        wandb.watch(model)



    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
            # for st in sampler_train:
            #     st.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm,args.wandb, args)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)


        if (epoch+1)%100==0:
            if args.hoi or args.att_det or args.mtl:
                if args.mtl:
                    for dlv in data_loader_val:
                        test_stats,dataset_name = evaluate_hoi_att(args.dataset_file, model, postprocessors, dlv, args.subject_category_id, device, args)
                        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
                        if 'v-coco' in dataset_name:
                            if args.output_dir and utils.is_main_process():
                                with (output_dir / "log.txt").open("a") as f:
                                    f.write(json.dumps(log_stats) + "\n")

                            if utils.get_rank() == 0 and args.wandb:
                        
                                wandb.log({
                                    'mAP_all': test_stats['mAP_all'],
                                    'mAP_thesis':test_stats['mAP_thesis']
                                })
                            performance=test_stats['mAP_thesis']

                        elif 'hico' in dataset_name:
                            if args.output_dir and utils.is_main_process():
                                with (output_dir / "log.txt").open("a") as f:
                                    f.write(json.dumps(log_stats) + "\n")

                            if utils.get_rank() == 0 and args.wandb:
                                wandb.log({
                                    'mAP': test_stats['mAP'],
                                    'mAP rare': test_stats['mAP rare'],
                                    'mAP non-rare':test_stats['mAP non-rare'],
                                    'mean max recall':test_stats['mean max recall']
                                })
                            performance=test_stats['mAP']

                        elif 'vaw' in dataset_name:
                            #CATEGORIES = ['all', 'head', 'medium', 'tail'] 
                            if args.output_dir and utils.is_main_process():
                                with (output_dir / "log.txt").open("a") as f:
                                    f.write(json.dumps(log_stats) + "\n")

                            if utils.get_rank() == 0 and args.wandb:
                                wandb.log({
                                    'vaw_mAP_all': test_stats['mAP_all'],
                                    'vaw_mAP_head': test_stats['mAP_head'],
                                    'vaw_mAP_medium':test_stats['mAP_medium'],
                                    'vaw_mAP_tail':test_stats['mAP_tail']
                                })

                        # #if 'v-coco' in os.fspath(dlv.dataset.img_folder) or 'hico' in os.fspath(dlv.dataset.img_folder):
                        # test_stats,dataset_name = evaluate_hoi_att(args.dataset_file, model, postprocessors, dlv, args.subject_category_id, device, args)
                        # if 'v-coco' in dataset_name:
                        #     if utils.get_rank() == 0 and args.wandb:                        
                        #         wandb.log({
                        #             'mAP_all': test_stats['mAP_all'],
                        #             'mAP_thesis':test_stats['mAP_thesis']
                        #         })
                        #     performance=test_stats['mAP_thesis']
                        # elif 'hico' in dataset_name:
                        #     if utils.get_rank() == 0 and args.wandb:
                        #         wandb.log({
                        #             'mAP': test_stats['mAP'],
                        #             'mAP rare': test_stats['mAP rare'],
                        #             'mAP non-rare':test_stats['mAP non-rare'],
                        #             'mean max recall':test_stats['mean max recall']
                        #         })
                        #     performance=test_stats['mAP']
                        # elif 'vaw' in dataset_name:
                        #     if utils.get_rank() == 0 and args.wandb:
                        #         wandb.log({
                        #             'mAP': test_stats['mAP'],
                        #             'mAP rare': test_stats['mAP rare'],
                        #             'mAP non-rare':test_stats['mAP non-rare'],
                        #             'mean max recall':test_stats['mean max recall']
                        #         })
                        #     performance=test_stats['mAP']
                        # coco_evaluator = None
                else:
                    test_stats,dataset_name = evaluate_hoi_att(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args)
                    if 'v-coco' in dataset_name:
                        if utils.get_rank() == 0 and args.wandb:
                    
                            wandb.log({
                                'mAP_all': test_stats['mAP_all'],
                                'mAP_thesis':test_stats['mAP_thesis']
                            })
                        performance=test_stats['mAP_thesis']
                    elif 'hico' in dataset_name:
                        if utils.get_rank() == 0 and args.wandb:
                            wandb.log({
                                'mAP': test_stats['mAP'],
                                'mAP rare': test_stats['mAP rare'],
                                'mAP non-rare':test_stats['mAP non-rare'],
                                'mean max recall':test_stats['mean max recall']
                            })
                        performance=test_stats['mAP']
                    elif 'vaw' in dataset_name:
                        if utils.get_rank() == 0 and args.wandb:
                            wandb.log({
                                'vaw_mAP_all': test_stats['mAP_all'],
                                'vaw_mAP_head': test_stats['mAP_head'],
                                'vaw_mAP_medium':test_stats['mAP_medium'],
                                'vaw_mAP_tail':test_stats['mAP_tail']
                            })
                        performance=test_stats['mAP']
                    coco_evaluator = None
            else:
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
                )

            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #             **{f'test_{k}': v for k, v in test_stats.items()},
            #             'epoch': epoch,
            #             'n_parameters': n_parameters}

            # if args.output_dir and utils.is_main_process():
            #     with (output_dir / "log.txt").open("a") as f:
            #         f.write(json.dumps(log_stats) + "\n")

            #     # for evaluation logs
            #     if coco_evaluator is not None:
            #         (output_dir / 'eval').mkdir(exist_ok=True)
            #         if "bbox" in coco_evaluator.coco_eval:
            #             filenames = ['latest.pth']
            #             if epoch % 50 == 0:
            #                 filenames.append(f'{epoch:03}.pth')
            #             for name in filenames:
            #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                         output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)