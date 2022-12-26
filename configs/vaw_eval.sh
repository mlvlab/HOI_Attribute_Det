#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --pretrained checkpoints/vaw/checkpoint.pth \
    --att_det \
    --dataset_file vaw \
    --data_path data/vaw \
    --num_obj_classes 81 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --eval \
    ${PY_ARGS}