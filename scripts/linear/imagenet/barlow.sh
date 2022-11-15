python3 main_linear.py \
    --dataset imagenet \
    --backbone resnet50 \
    --train_data_path /data/datasets/imagenet/train \
    --val_data_path /data/dataset/simagenet/val \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lr 0.3 \
    --scheduler warmup_cosine \
    --warmup_epochs 0 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 10 \
    --data_format dali \
    --pretrained_feature_extractor $1 \
    --name barlow-resnet50-imagenet-linear-eval \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --auto_resume