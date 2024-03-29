CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset tinyimagenet \
    --backbone resnet18 \
    --train_data_path /mnt/hdd1/chencheng/cl_dataset/Tinyimagenet/ \
    --val_data_path /mnt/hdd1/chencheng/cl_dataset/Tinyimagenet/ \
    --max_epochs 50 \
    --Task 10 \
    --devices 0 \
    --accelerator gpu \
    --optimizer sgd \
    --scheduler none \
    --lr 0.03 \
    --classifier_lr 0.03 \
    --weight_decay 1e-4 \
    --batch_size 32 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0 0 \
    --num_crops_per_aug 1 1 \
    --crop_size 64 \
    --name si \
    --project CUCL_855_Tiny \
    --entity zacks \
    --method si \
    --train_c 0.01 \
    --supervised \
    --wandb \