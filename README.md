## Preparation

### Dependencies
Our implementation uses external libraries such as NumPy and PyTorch. You can resolve the dependencies with the following command.
```
pip install numpy
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools
pip install git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi
pip install scipy cython submitit
```
Note that this command may dump errors during installing pycocotools, but the errors can be ignored.

### Dataset

#### HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.

#### V-COCO
First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle` from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make directories as follows.

```
neubla_hoi_att
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 |       |   |─ corre_vcoco.npy
 |       |   |─ trainval_vcoco.json
 |       |   |─ test_vcoco.json
 :       :   :
     └─ hico_20160224_det
 |       |─ images
 |       |   |─ train2015
 |       |   |   |─ HICO_train2015_00000001.jpg
 |       |   |   :
 |       |   └─ test2015
 |       |       |─ HICO_test2015_00000001.jpg
 |       |       :
 |       |─ annotations
 |       |   |─ corre_hico.npy
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 :       :   :
      └─ vaw
 |       |─ images
 |       |   |─ VG_100K
 |       |   |   |─ 2.jpg
 |       |   |   :
 |       |   └─ VG_100K_2
 |       |       |─ 1.jpg
 |       |       :
 |       |─ annotations
 |       |   |─ attribute_index.json
 |       |   |─ vaw_coco_train.json
 |       |   |─ vaw_coco_test.json
 |       |   |─ vaw_coco_train_cat_info.json
 :       :   :
```


For our implementation, the annotation file have to be converted to the HOIA format. The conversion can be conducted as follows.
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be generated to `annotations` directory.

### Pre-trained parameters
Our QPIC have to be pre-trained with the COCO object detection dataset. For the HICO-DET training, this pre-training can be omitted by using the parameters of DETR. The parameters can be downloaded from [here](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth) for the ResNet50 backbone, and [here](https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth) for the ResNet101 backbone. For the V-COCO training, this pre-training has to be carried out because some images of the V-COCO evaluation set are contained in the training set of DETR. You have to pre-train QPIC without those overlapping images by yourself for the V-COCO evaluation.

For HICO-DET, move the downloaded parameters to the `params` directory and convert the parameters with the following command.
```
python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-hico.pth
```

For V-COCO, convert the pre-trained parameters with the following command.
```
python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-vcoco.pth \
        --dataset vcoco
```

For VAW, convert the pre-trained parameters with the following command.
```
python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-vaw.pth \
        --use_vaw
```

For MTL(attribute + hoi detection), convert the pre-trained parameters with the following command.
```
python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-mtl.pth \
        --use_vaw \
        --dataset vcoco
```




# Training

## For single task training
```
CUDA_VISIBLE_DEVICES=1,2 GPUS_PER_NODE=2 ./tool/run_dist_launch.sh 2 configs/mtl_train.sh \
        --mtl_data [\'vaw\'] \
        --output_dir checkpoints/vaw \
        --pretrained params/detr-r50-pre-vaw.pth
```  

## For multi task training
```
CUDA_VISIBLE_DEVICES=1,2 GPUS_PER_NODE=2 ./tool/run_dist_launch.sh 2 configs/mtl_train.sh \
        --mtl_data [\'vcoco\',\'hico\',\'vaw\'] \
        --output_dir checkpoints/mtl_all \
        --pretrained params/detr-r50-pre-mtl.pth
``` 

# Evaluation

## Multi task learning evaluation
```
configs/mtl_eval.sh \
        --pretrained checkpoints/mtl_all/checkpoint.pth \
        --output_dir test_results/ \
        --mtl_data [\'vcoco\',\'hico\',\'vaw\']
```

## vcoco evaluation
```
"test_mAP_all": 0.5455718251429631, "test_mAP_thesis": 0.5663461447990525
```
## hico evaluation
```
"test_mAP": 0.27877264960450454, "test_mAP rare": 0.20416854381834068, "test_mAP non-rare": 0.30105699289128085, "test_mean max recall": 0.6536133057960736
```
## vaw evaluation
```
"test_mAP": 0.0524253535493328, "test_mAP rare": 0.029776368059209662, "test_mAP non-rare": 0.07093753803669373, "test_mean max recall": 0.37233911507467743
```

# Video demo version 1
![cycle2](https://user-images.githubusercontent.com/87055052/208564990-197d157c-c830-4cae-9557-9d7900f1b8c6.gif)
## For vcoco verb inference
```
python vis_demo.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type vcoco \
        --mtl_data [\'vcoco\'] \
        --mtl \
        --video_file video/cycle.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4 \
        --fps 30
```  

## For hico verb inference
```
python vis_demo.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type hico \
        --mtl_data [\'hico\'] \
        --mtl \
        --video_file video/cycle.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4 \
        --fps 30
``` 

## For hoi inference (hico verb + vcoco verb) 
```
python vis_demo.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type [\'hico\',\'vcoco\'] \
        --mtl_data [\'hico\',\'vcoco\'] \
        --mtl \
        --video_file video/cycle.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4 \
        --fps 30
``` 

## For vaw attribute inference
```
python vis_demo.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type vaw \
        --mtl_data [\'vaw\'] \
        --mtl \
        --video_file video/animal.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4 \
        --fps 30
```  

## For vaw color inference
```
python vis_demo.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type vaw \
        --mtl_data [\'vaw\'] \
        --mtl \
        --video_file video/animal.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4 \
        --fps 30 \
        --color
```  

# Video demo version 2 
![cycle](https://user-images.githubusercontent.com/87055052/208564037-b6054ccd-bc28-41ea-bc77-ce1195a19f33.gif)

## For hoi+attribute inference
```
python vis_demo2.py \
        --checkpoint checkpoints/mtl_all/checkpoint.pth \
        --inf_type [\'vcoco\',\'vaw\'] \
        --mtl_data [\'vcoco\',\'vaw\'] \
        --mtl \
        --video_file video/cycle.mp4 \
        --show_vid \
        --top_k 2 \
        --threshold 0.4 \
        --fps 30 \
        --all
```        
## Citation
Our implementation is based on the official code QPIC
```
@inproceedings{tamura_cvpr2021,
author = {Tamura, Masato and Ohashi, Hiroki and Yoshinaga, Tomoaki},
title = {{QPIC}: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information},
booktitle={CVPR},
year = {2021},
}
```
