python train.py \
    --filename ./eng-fra.txt \
    --encoder_embedding_size 100 \
    --encoder_hidden_size 100 \
    --encoder_num_layers 2 \
    --encoder_bidirectional \
    --decoder_embedding_size 100 \
    --decoder_hidden_size 100 \
    --decoder_num_layers 2 \
    --tied \
    --dropout_ratio 0.5 \
    --max_len 20 \
    --min_count 3 \
    --lr 0.001 \
    --epochs 5 \
    --batch_size 128 \
    --teacher_forcing_ratio 0.5 \
    --seed 7 \
    --device cpu \
    --log_interval 50 \
    --log_file ./logs/train.log \
    --model_save_path ./models \
    --train_from \
    --checkpoint ./models/checkpoint.epoch-3.pth \


/


