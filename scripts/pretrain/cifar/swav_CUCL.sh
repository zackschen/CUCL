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
    --scheduler none \
    --lr 0.03 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --min_scale 0.14 \
    --gaussian_prob 0 0 \
    --num_crops_per_aug 1 1 \
    --crop_size 32 \
    --name swav_CUCL_Forloss  \
    --wandb \
    --project 855-CIFAR \
    --entity zacks \
    --method swav \
    --knn_k 100 \
    --knn_temperature 0.1 \
    --knn_distance_function cosine \
    --knn_feature_type backbone \
    --proj_hidden_dim 2048 \
    --queue_size 512 \
    --proj_output_dim 512 \
    --num_prototypes 200 \
    --epoch_queue_starts 15 \
    --freeze_prototypes_epochs 2 \
    --CUCL \
    --buffer_size $2 \
    --N_books 8 \
    --N_words 8 \
    --L_word 16 \
    --sample_type Old \
    --CUCL_lr 0.03  \
    --CUCL_lambda 1.0 \
    --CUCL_for_Loss \
    # --train_from_task 1 \
    # --CUCL_loadPath ./checkpoints/cifar100_results/swav_CUCL_test-27ws1s34 \
    # --train_task 1 \
    # --offline \
    # --save_checkpoint \
    # --auto_resume \
