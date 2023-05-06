CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset miniimagenet \
    --backbone resnet18 \
    --train_data_path /mnt/hdd1/|*|/cl_dataset/mini-imagenet/ \
    --val_data_path /mnt/hdd1/|*|/cl_dataset/mini-imagenet/ \
    --max_epochs 200 \
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
    --crop_size 84 \
    --name byol_CUCL_testp \
    --project CUCL_855_Mini \
    --entity zacks \
    --wandb \
    --method byol \
    --knn_k 100 \
    --knn_temperature 0.1 \
    --knn_distance_function cosine \
    --knn_feature_type backbone \
    --proj_output_dim 512 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --CUCL \
    --buffer_size $2 \
    --N_books 8 \
    --N_words 8 \
    --L_word 16 \
    --sample_type Old \
    --CUCL_lr 0.03  \
    --CUCL_lambda 1.0 \
    # --load_first \
    # --CUCL_loadPath ./checkpoints/cifar100_results/Final_byol_CUCL/0.pth \
    # --train_task 1 \
    # --offline \
    # --save_checkpoint \
    # --auto_resume \
