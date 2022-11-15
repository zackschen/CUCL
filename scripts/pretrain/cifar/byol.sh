CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset cifar100 \
    --backbone vit_base \
    --train_data_path /home/chencheng/Code/CUCL/Data/CIFAR100/ \
    --val_data_path /home/chencheng/Code/CUCL/Data/CIFAR100/ \
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
    --crop_size 32 \
    --name byol_finetune_Vit \
    --project 855-CIFAR \
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
    # --train_task 1 \