#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --pretrained params/detr-r50-pre-vaw.pth \
    --run_name ${EXP_DIR} \
    --project_name QPIC_VAW \
    --mtl \
    --batch_size 4 \
    --update_obj_att \
    --epochs 90 \
    --lr_drop 30 \
    --dataset_file vaw \
    --data_path data/vaw \
    --num_obj_att_classes 80 \
    --num_obj_classes 81 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --output_dir checkpoints/vaw/ \
    ${PY_ARGS}