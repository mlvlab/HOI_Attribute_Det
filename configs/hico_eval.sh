#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --pretrained checkpoints/hico/checkpoint.pth \
    --hoi \
    --dataset_file hico \
    --data_path data/hico_20160224_det \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --eval \
    ${PY_ARGS}