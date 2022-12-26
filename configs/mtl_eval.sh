#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --pretrained checkpoints/mtl_all/checkpoint.pth \
    --mtl \
    --update_obj_att \
    --dataset_file vaw \
    --mtl_data [\'vcoco\',\'hico\',\'vaw\'] \
    --data_path data/vaw \
    --num_obj_att_classes 80 \
    --num_obj_classes 81 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --eval \
    ${PY_ARGS}