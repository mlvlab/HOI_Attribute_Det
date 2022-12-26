#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --pretrained params/detr-r50-pre-hico-cpc.pth \
    --run_name ${EXP_DIR} \
    --project_name CPC_QPIC_HICODET \
    --hoi \
    --epochs 80 \
    --lr_drop 50 \
    --dataset_file hico \
    --data_path data/hico_20160224_det \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    ${PY_ARGS}