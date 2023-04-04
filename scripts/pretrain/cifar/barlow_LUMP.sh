CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset cifar100 \
    --backbone resnet18 \
    --train_data_path /mnt/hdd1/chencheng/cl_dataset/CIFAR100/ \
    --val_data_path /mnt/hdd1/chencheng/cl_dataset/CIFAR100/ \
    --max_epochs 300 \
    --Task 10 \
    --devices 0 \
    --accelerator gpu \
    --optimizer lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler none \
    --lr 0.03 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 0 0 \
    --num_crops_per_aug 1 1 \
    --crop_size 32 \
    --name barlow_LUMP  \
    --project CIFAR \
    --entity zacks \
    --method barlow_twins \
    --scale_loss 0.1 \
    --knn_k 100 \
    --knn_temperature 0.1 \
    --knn_distance_function cosine \
    --knn_feature_type backbone \
    --proj_hidden_dim 2048 \
    --proj_output_dim 512 \
    --wandb \
    --LUMP \
    --LUMP_lambda 0.1 \
    --LUMP_size 256 \
    # --N_books 8 \
    # --N_words 8 \
    # --L_word 16 \
    # --sample_type Old \
    # --CUCL_lr 0.03  \
    # --CUCL_lambda 1.0 \
    # --train_from_task 1 \
    # --CUCL_loadPath ./checkpoints/cifar100_results/barlow_CUCL_testp-oc6yls5q \
    # --CUCL_for_Loss \
    # --train_task 1 \
    # --CUCL_cosine \
    # --CUCL_epoch 10 \
    # --save_checkpoint \
    # --auto_resume \
