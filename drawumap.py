import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from computeAA import getAccs
plt.switch_backend('agg')
import re
import torch
from pytorch_lightning import Trainer, seed_everything
from solo.args.setup import parse_args_pretrain
from solo.methods import METHODS
from solo.utils.misc import make_contiguous
import os
from solo.data.classification_dataloader import prepare_datasets as prepare_datasets_classification
from solo.data.classification_dataloader import prepare_transforms as prepare_transforms_classification
from torch.utils.data import DataLoader, Dataset
from main_knn import extract_features, run_knn
from solo.utils.plot_codebook import plot_codebook

def main():
    seed_everything(1)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

    args = parse_args_pretrain()

    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    if args.num_large_crops != 2:
        assert args.method in ["wmse", "mae"]

    model = METHODS[args.method](**args.__dict__)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not args.no_channel_last:
        model = model.to(memory_format=torch.channels_last)

    # path = './checkpoints/cifar100_results/simsiam_CUCL_Forloss-2rzy5rdq'
    # path = './checkpoints/cifar100_results/simsiam_CUCL_testp20-11v3k2vd'
    path = './checkpoints/cifar100_results/simsiam_finetune-2t5dh3ix'
    args.name = 'Umap-Simsiam-finetune-90'
    path = os.path.join(path, "9.pth")
    save_dict = torch.load(path, map_location='cpu')
    buffer = model.buffers
    msg = model.load_state_dict(save_dict['state_dict'], strict=True)
    model.buffers = buffer
    make_contiguous(model)
    if not args.no_channel_last:
        model = model.to(memory_format=torch.channels_last)

    task_id = 00
    order = 0
    N_CLASSES_PER_TASK = model.class_per_task
    
    model.eval()
    model = model.cuda()
    # prepare data
    _, T = prepare_transforms_classification(args.dataset)
    train_dataset, val_dataset = prepare_datasets_classification(
        args.dataset,
        T_train=T,
        T_val=T,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        data_format=args.data_format,
    )
    train_mask = np.logical_and(np.array(train_dataset.targets) >= order*N_CLASSES_PER_TASK,
    np.array(train_dataset.targets) < (order+1)*N_CLASSES_PER_TASK)
    train_dataset.data = np.array(train_dataset.data)[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order*N_CLASSES_PER_TASK

    test_mask = np.logical_and(np.array(val_dataset.targets) >= order*N_CLASSES_PER_TASK,
    np.array(val_dataset.targets) < (order+1)*N_CLASSES_PER_TASK)
    val_dataset.data = np.array(val_dataset.data)[test_mask]
    val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order*N_CLASSES_PER_TASK

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,drop_last=False,)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,drop_last=False,)

    # extract train features
    train_features_bb, train_features_proj, train_targets = extract_features(train_loader, model)
    train_features = {"backbone": train_features_bb, "projector": train_features_proj}

    # extract test features
    test_features_bb, test_features_proj, test_targets = extract_features(val_loader, model)
    test_features = {"backbone": test_features_bb, "projector": test_features_proj}

    plot_codebook(model,train_features['backbone'],train_targets.cpu().numpy(),args,task_id)

    # quanti_feature = model.quanti_Model(train_features['projector'],0)
    # plot_codebook(model,train_features['projector'],train_targets.cpu().numpy(),args,task_id)

    # args.name = 'Umap-Simsiam-For_loss-00-after'
    # plot_codebook(model,quanti_feature[1],train_targets.cpu().numpy(),args,task_id)

if __name__ == "__main__":
    main()
