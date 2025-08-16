#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=7
                        python -u main.py  \
                        --print_freq 400 \
                        --data_dir ~/data \
                        --exp_dir ~/exp \
                        --dataset cifar100 \
                        --model reskagnet \
                        --batch_size 128 \
                        --epochs 300 \
                        --lr 0.1 \
                        --lr_scheduler cosine \
                        --momentum 0.9 \
                        --weight_decay 5e-4 \
                        --workers 2 \
                        --cutout 8 \
                        --aug_type autoaug_cifar10 \
                        --exp_type baseline \
                        --optimiser AdamW
