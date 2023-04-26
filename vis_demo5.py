
 
import argparse
import datetime
import json
import random
import time
import multiprocessing
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import hotr.data.datasets as datasets
import hotr.util.misc as utils
from hotr.engine.arg_parser import get_args_parser
from hotr.data.datasets import build_dataset, get_coco_api_from_dataset
from hotr.data.datasets.vcoco import make_hoi_transforms
from PIL import Image
from hotr.util.logger import print_params, print_args
from hotr.util.tool import draw_img_vcoco
import copy
from hotr.data.datasets import builtin_meta
from PIL import Image
import requests
import mmcv
from matplotlib import pyplot as plt
import imageio


from threading import Thread
import time
from hotr.util.misc import NestedTensor
import math
from hotr.models import build_model
from hotr.models.backbone import *
from hotr.models.transformer import *
from hotr.models.hotr import *
from hotr.models.detr import *
from hotr.models.criterion import *
from hotr.models.detr_matcher import *
from hotr.models.feed_forward import *
from hotr.models.hotr_matcher import *
from hotr.models.position_encoding import *
from hotr.models.post_process import *



def change_format_ori(results,valid_ids):
   
    boxes,labels,pair_score =\
                    list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['pair_score']]))
    output_i={}
    output_i['predictions']=[]
    output_i['hoi_prediction']=[]

    h_idx=np.where(labels==1)[0]
    for box,label in zip(boxes,labels):
       
        output_i['predictions'].append({'bbox':box.tolist(),'category_id':label})
   
    for i,verb in enumerate(pair_score):
        if i in [1,4,10,23,26,5,18]:
            continue
        for j,hum in enumerate(h_idx):
            for k in range(len(boxes)):
                if verb[j][k]>0:
                    output_i['hoi_prediction'].append({'subject_id':hum,'object_id':k,'category_id':i+2,'score':verb[j][k]})
           
    return output_i


def change_format(results,valid_ids):
    st=time.time()

    boxes,labels,pair_score =\
                    list(map(lambda x: x.cpu().numpy(), [results['boxes'], results['labels'], results['pair_score']]))

    output_i={}
    output_i['predictions']=[]
    output_i['hoi_prediction']=[]
    hoi_predictions = []
    h_idx=np.where(labels==1)[0]

    for box,label in zip(boxes,labels):

        output_i['predictions'].append({'bbox':box.tolist(),'category_id':label})

    p=pair_score[...,:len(boxes)]
    p_idx = np.where(p>0)
    for i in range(len(p_idx[0])):
        hoi_predictions.append({'subject_id':h_idx[p_idx[1][i]],'object_id':p_idx[2][i],'category_id':p_idx[0][i]+2,'score':p[p_idx][i]})
    for dic in hoi_predictions:
        if dic['category_id']-2 not in [1,4,10,23,26,5,18]:

            output_i['hoi_prediction'].append(dic)


    return output_i

class VideoStream():
    def __init__(self, src='0'):
        self.stream = cv2.VideoCapture('rtmp://10.42.0.1/live/drone')
        #self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def get_video_dimensions(self):
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)

    def stop_process(self):
        self.stopped = True

class PositionEmbeddingSine(torch.nn.Module):
   
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
   
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(torch.nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos

def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = False # args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_model(args):
   
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality'] if args.frozen_weights is None else []
    if args.HOIDet:
        hoi_matcher = build_hoi_matcher(args)
        hoi_losses = []
        hoi_losses.append('pair_labels')
        hoi_losses.append('pair_actions')
        if args.dataset_file == 'hico-det': hoi_losses.append('pair_targets')
       
        hoi_weight_dict={}
        hoi_weight_dict['loss_hidx'] = args.hoi_idx_loss_coef
        hoi_weight_dict['loss_oidx'] = args.hoi_idx_loss_coef
        hoi_weight_dict['loss_act'] = args.hoi_act_loss_coef
        if args.dataset_file == 'hico-det': hoi_weight_dict['loss_tgt'] = args.hoi_tgt_loss_coef
        if args.hoi_aux_loss:
            hoi_aux_weight_dict = {}
            for i in range(args.hoi_dec_layers):
                hoi_aux_weight_dict.update({k + f'_{i}': v for k, v in hoi_weight_dict.items()})
            hoi_weight_dict.update(hoi_aux_weight_dict)

        criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=hoi_weight_dict,
                                 eos_coef=args.eos_coef, losses=losses, num_actions=args.num_actions,
                                 HOI_losses=hoi_losses, HOI_matcher=hoi_matcher, args=args)

        interaction_transformer = build_hoi_transformer(args) # if (args.share_enc and args.pretrained_dec) else None

        kwargs = {}
        if args.dataset_file == 'hico-det': kwargs['return_obj_class'] = args.valid_obj_ids
        model = HOTR(
            detr=model,
            num_hoi_queries=args.num_hoi_queries,
            num_actions=args.num_actions,
            interaction_transformer=interaction_transformer,
            freeze_detr=(args.frozen_weights is not None),
            share_enc=args.share_enc,
            pretrained_dec=args.pretrained_dec,
            temperature=args.temperature,
            hoi_aux_loss=args.hoi_aux_loss,
            **kwargs # only return verb class for HICO-DET dataset
        )
        postprocessors = {'hoi': PostProcess(args.HOIDet)}
    else:
        criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)
        postprocessors = {'bbox': PostProcess(args.HOIDet)}
    criterion.to(device)

    return model, criterion, postprocessors

def vis(args,id=294):

    if args.frozen_weights is not None:
        print("Freeze weights for detector")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data Setup
    dataset_train = build_dataset(image_set='train', args=args)
    args.num_classes = dataset_train.num_category()
    args.num_actions = dataset_train.num_action()
    args.action_names = dataset_train.get_actions()
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers
    if args.dataset_file == 'vcoco':
        # Save V-COCO dataset statistics
        args.valid_ids = np.array(dataset_train.get_object_label_idx()).nonzero()[0]
        args.invalid_ids = np.argwhere(np.array(dataset_train.get_object_label_idx()) == 0).squeeze(1)
        args.human_actions = dataset_train.get_human_action()
        args.object_actions = dataset_train.get_object_action()
        args.num_human_act = dataset_train.num_human_act()
    elif args.dataset_file == 'hico-det':
        args.valid_obj_ids = dataset_train.get_valid_obj_ids()
    print_args(args)

    args.HOIDet=True
    args.eval=True
    args.pretrained_dec=True
    args.share_enc=True
    if args.dataset_file=='hico-det':
        args.valid_ids=args.valid_obj_ids
 
    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    n_parameters = print_params(model)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    output_dir = Path(args.output_dir)
   
    checkpoint = torch.load(args.resume, map_location='cpu')
    #수정
    module_name=list(checkpoint['model'].keys())
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
   
    #video_stream = VideoStream(source).start()
    cap = cv2.VideoCapture("rtmp://192.168.0.98/live/drone")
    starting_time = time.time()
    frame_id = 0
    #img_w, img_h =  video_stream.get_video_dimensiont()

    while True:
        _, frame = cap.read()
        h, w, c = frame.shape
       
        orig_size = torch.as_tensor([int(h), int(w)]).unsqueeze(0).to(device)
        transform = make_hoi_transforms('val')
       
        frame=frame.copy()

        frame=Image.fromarray(frame,'RGB')

        sample,_=transform(frame,None)
        sample=sample.unsqueeze(0).to(device)

        with torch.no_grad():
            model.eval()
            out=model(sample)
            results = postprocessors['hoi'](out, orig_size,dataset='vcoco')
            output_i=change_format(results[0],args.valid_ids)

        vis_img=draw_img_vcoco(np.array(frame),output_i,top_k=args.topk,threshold=args.threshold,color=builtin_meta.COCO_CATEGORIES)
        #frames.append(vis_img)
        #video_writer.write(vis_img)

        cv2.imshow("Demo", vis_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
   
    cv2.destroyAllWindows()
 
       

       
    '''
    frames=[]
    video_file=id

    video_reader = mmcv.VideoReader('./vid/'+video_file+'.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
            './vid/'+video_file+'_vis.mp4', fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    orig_size = torch.as_tensor([int(video_reader.height), int(video_reader.width)]).unsqueeze(0).to(device)
    transform=make_hoi_transforms('val')

    for frame in mmcv.track_iter_progress(video_reader):

        frame=mmcv.imread(frame)
        frame=frame.copy()

        frame=Image.fromarray(frame,'RGB')

        sample,_=transform(frame,None)
        sample=sample.unsqueeze(0).to(device)

        with torch.no_grad():
            model.eval()
            out=model(sample)
            results = postprocessors['hoi'](out, orig_size,dataset='vcoco')
            output_i=change_format(results[0],args.valid_ids)

        vis_img=draw_img_vcoco(np.array(frame),output_i,top_k=args.topk,threshold=args.threshold,color=builtin_meta.COCO_CATEGORIES)
        frames.append(vis_img)
        video_writer.write(vis_img)

    with imageio.get_writer("smiling.gif", mode="I") as writer:
        for idx, frame in enumerate(frames):
            # print("Adding frame to GIF file: ", idx + 1)
            writer.append_data(frame)
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    '''

def visualization(id, video_vis=False, dataset_file='vcoco', data_path='v-coco', threshold=0.4, topk=10):

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    checkpoint_dir= './checkpoint/hotr_vcoco/vcoco_q16.pth' if dataset_file=='vcoco' else './checkpoints/hico-det/hico_ft_q16.pth'
    with open('./v-coco/data/vcoco_test.ids') as file:
      test_idxs = [line.rstrip('\n') for line in file]
    if not video_vis:
      id = test_idxs[id]
    args = parser.parse_args(args=['--dataset_file',dataset_file,'--data_path',data_path,'--resume',checkpoint_dir,'--num_hoi_queries' ,'16','--temperature' ,'0.05' ])
    args.video_vis=video_vis
    args.threshold=threshold
    args.topk=topk
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    vis(args)
if __name__ == '__main__':
    visualization(id='tri_cut', video_vis = True, dataset_file = 'vcoco', data_path = 'v-coco', threshold=0.4, topk=10) 