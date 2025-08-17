#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=5
                        python -u main.py  \
                        --print_freq 400 \
                        --data_dir ~/data \
                        --exp_dir ~/exp \
                        --dataset reduced_cifar10 \
                        --model reskagnet \
                        --batch_size 128 \
                        --epochs 300 \
                        --lr 0.1 \
                        --lr_scheduler cosine \
                        --momentum 0.9 \
                        --weight_decay 5e-4 \
                        --workers 2 \
                        --cutout 16 \
                        --perturb_vae vae_conv_cifar_v1 \
                        --z_dim 8 \
                        --fea_dim 512 \
                        --adv_weight_vae 10 \
                        --div_weight_vae 1e-3 \
                        --bn_num 2 \
                        --inner_num 4 \
                        --aug_type basic \
                        --exp_type perturb_vae \
                        --optimiser AdamW

