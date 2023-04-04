CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset miniimagenet \
    --backbone resnet18 \
    --train_data_path /mnt/hdd1/chencheng/cl_dataset/mini-imagenet/ \
    --val_data_path /mnt/hdd1/chencheng/cl_dataset/mini-imagenet/ \
    --max_epochs 50 \
    --Task 10 \
    --devices 0 \
    --accelerator gpu \
    --optimizer sgd \
    --scheduler none \
    --lr 0.03 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0 0 \
    --num_crops_per_aug 1 1 \
    --crop_size 32 \
    --name der \
    --project CUCL_855_Mini \
    --entity zacks \
    --method der \
    --buffer_size 256 \
    --train_alpha 0.01 \
    --supervised \
    --wandb \