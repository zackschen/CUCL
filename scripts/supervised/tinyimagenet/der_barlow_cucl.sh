CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset tinyimagenet \
    --backbone resnet18 \
    --train_data_path /mnt/hdd1/chencheng/cl_dataset/Tinyimagenet/ \
    --val_data_path /mnt/hdd1/chencheng/cl_dataset/Tinyimagenet/ \
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
    --crop_size 64 \
    --name der_barlow_CUCL \
    --unsuper_method Barlow \
    --project CUCL_855_Tiny \
    --entity zacks \
    --method der \
    --scale_loss 0.1 \
    --der_size 256 \
    --train_alpha 0.01 \
    --knn_k 100 \
    --knn_temperature 0.1 \
    --knn_distance_function cosine \
    --knn_feature_type backbone \
    --proj_hidden_dim 2048 \
    --proj_output_dim 512 \
    --CUCL \
    --buffer_size 20 \
    --N_books 8 \
    --N_words 8 \
    --L_word 16 \
    --sample_type Old \
    --CUCL_lr 0.03  \
    --CUCL_lambda 1.0 \
    --wandb \