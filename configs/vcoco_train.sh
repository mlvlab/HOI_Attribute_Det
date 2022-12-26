
#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --project_name QPIC_VCOCO \
    --run_name ${EXP_DIR} \
    --pretrained params/detr-r50-pre-vcoco.pth \
    --hoi \
    --epochs 80 \
    --lr_drop 50 \
    --lr 1e-4 \
    --batch_size 8 \
    --lr_backbone 1e-5 \
    --dataset_file vcoco \
    --data_path data/v-coco \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --backbone resnet50 \
    --output_dir checkpoints/vcoco/ \
    ${PY_ARGS}