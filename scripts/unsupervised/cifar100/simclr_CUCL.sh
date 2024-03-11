CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset cifar100 \
    --backbone resnet18 \
    --train_data_path /mnt/hdd1/chencheng/cl_dataset/CIFAR100/ \
    --val_data_path /mnt/hdd1/chencheng/cl_dataset/CIFAR100/ \
    --max_epochs 200 \
    --Task 10 \
    --devices 0 \
    --accelerator gpu \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler none \
    --lr 0.03 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0 0 \
    --num_crops_per_aug 1 1 \
    --crop_size 32 \
    --name simclr_CUCL_Forloss \
    --project 855-CIFAR \
    --entity zacks \
    --wandb \
    --method simclr \
    --knn_k 100 \
    --knn_temperature 0.1 \
    --knn_distance_function cosine \
    --knn_feature_type backbone \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --proj_output_dim 512 \
    --CUCL \
    --buffer_size $2 \
    --N_books 8 \
    --N_words 8 \
    --L_word 16 \
    --sample_type Old \
    --CUCL_lr 0.03  \
    --CUCL_lambda 1.0 \
    --CUCL_for_Loss \
    --train_task 1 \
    # --train_from_task 2 \
    # --CUCL_loadPath ./checkpoints/cifar100_results/simclr_CUCL_testp-1gws8t0w \
    # --CUCL_for_Loss \
    # --train_task 1 \
    # --offline \
    # --save_checkpoint \
    # --auto_resume \
