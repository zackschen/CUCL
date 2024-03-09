CUDA_VISIBLE_DEVICES=$1 python3 main_pretrain_CL.py \
    --dataset miniimagenet \
    --backbone resnet18 \
    --train_data_path /mnt/hdd1/chencheng/cl_dataset/mini-imagenet/ \
    --val_data_path /mnt/hdd1/chencheng/cl_dataset/mini-imagenet/ \
    --max_epochs 200 \
    --Task 10 \
    --devices 0 \
    --accelerator gpu \
    --optimizer sgd \
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
    --name simsiam_LUMP_CUCL \
    --project CUCL_855_Mini \
    --entity zacks \
    --method simsiam \
    --knn_k 100 \
    --knn_temperature 0.1 \
    --knn_distance_function cosine \
    --knn_feature_type backbone \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --proj_output_dim 512 \
    --wandb \
    --LUMP \
    --LUMP_lambda 0.4 \
    --LUMP_size 256 \
    --CUCL \
    --buffer_size 20 \
    --N_books 8 \
    --N_words 8 \
    --L_word 16 \
    --sample_type Old \
    --CUCL_lr 0.03  \
    --CUCL_lambda 1.0 \
    # --N_books 8 \
    # --N_words 8 \
    # --L_word 32 \
    # --sample_type Old \
    # --CUCL_lr 0.03  \
    # --CUCL_lambda 1.0 \
    # --train_task 2 \
    # --load_first \
    # --CUCL_loadPath ./checkpoints/cifar100_results/Final_simsiam_CUCL/0.pth \
    # --CUCL_for_Loss \
    # --CUCL_type projector \
    # --CUCL_epoch 10 \
    # --CUCL_cosine \
    # --save_checkpoint \
    # --zero_init_residual \
    # --offline \
    # --auto_resume \
    
