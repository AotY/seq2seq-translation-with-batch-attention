#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4

python train.py \
    --filename ./eng-fra.txt \
    --encoder_embedding_size 256 \
    --encoder_hidden_size 256 \
    --encoder_num_layers 2 \
    --encoder_bidirectional \
    --decoder_embedding_size 256 \
    --decoder_hidden_size 256 \
    --decoder_num_layers 2 \
    --tied \
    --dropout_ratio 0.5 \
    --max_norm 50.0 \
    --max_len 20 \
    --min_count 3 \
    --lr 0.005 \
    --epochs 5 \
    --start_epoch 1 \
    --batch_size 128 \
    --teacher_forcing_ratio 0.5 \
    --seed 7 \
    --device cuda \
    --log_interval 256 \
    --log_file ./logs/train.log \
    --model_save_path ./models \
    --train_or_eval train \
    # --checkpoint ./models/checkpoint.epoch-5.pth \

/


