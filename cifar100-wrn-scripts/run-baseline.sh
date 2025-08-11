#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=6
                        python -u main.py  \
                        --data_dir ~/data \
                        --exp_dir ~/exp \
                        --dataset cifar100 \
                        --model wresnet28_10 \
                        --batch_size 128 \
                        --epochs 200 \
                        --lr 0.1 \
                        --lr_scheduler cosine \
                        --momentum 0.9 \
                        --weight_decay 5e-4 \
                        --workers 2 \
                        --cutout 8 \
                        --aug_type basic \
                        --exp_type baseline \

