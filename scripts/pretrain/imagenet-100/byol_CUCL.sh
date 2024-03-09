CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset imagenet100 \
    --backbone resnet18 \
    --train_data_path /mnt/hdd1/chencheng/cl_dataset/Imagenet100-kg/train \
    --val_data_path /mnt/hdd1/chencheng/cl_dataset/Imagenet100-kg/val \
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
    --batch_size 64 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --data_format dali \
    --num_crops_per_aug 2 \
    --name byol_CUCL \
    --project CUCL_855_100 \
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
    # --train_from_task 2 \
    # --CUCL_loadPath ./checkpoints/tinyimagenet_results/byol_CUCL_testp-1zn6pbgy \
    # --train_task 1 \
    # --offline \
    # --save_checkpoint \
    # --auto_resume \
