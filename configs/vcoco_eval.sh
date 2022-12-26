#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --pretrained checkpoints/vaw/checkpoint.pth \
    --hoi \
    --dataset_file vcoco \
    --data_path data/v-coco \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --backbone resnet50 \
    --eval \
    ${PY_ARGS}