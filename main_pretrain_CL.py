# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from ast import arg
from http.client import MOVED_PERMANENTLY
from lib2to3.pgen2.token import N_TOKENS
import os
import pandas as pd
from pprint import pprint
from random import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch,copy
import torchvision
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from main_knn import extract_features, run_knn
from solo.args.setup import parse_args_pretrain
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.classification_dataloader import prepare_datasets as prepare_datasets_classification
from solo.data.classification_dataloader import prepare_transforms as prepare_transforms_classification
from solo.data.classification_dataloader import prepare_dataloaders as prepare_dataloaders_classification
from solo.data.pretrain_dataloader import (
    prepare_dataloader,
    prepare_cl_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
    dataset_with_index,
)
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous
import torch.nn.functional as F 
import torchvision.transforms as T
from solo.utils.plot_codebook import plot_codebook
try:
    from solo.data.dali_dataloader import PretrainDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


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

    # validation dataloader for when it is available
    # if args.dataset == "custom" and (args.no_labels or args.val_data_path is None):
    #     val_loader = None
    # elif args.dataset in ["imagenet100", "imagenet"] and (args.val_data_path is None):
    #     val_loader = None
    # else:
    #     if args.data_format == "dali":
    #         val_data_format = "image_folder"
    #     else:
    #         val_data_format = args.data_format

    #     _, val_loader = prepare_data_classification(
    #         args.dataset,
    #         train_data_path=args.train_data_path,
    #         val_data_path=args.val_data_path,
    #         data_format=val_data_format,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #     )

    # pretrain dataloader
    if args.data_format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."

        dali_datamodule = PretrainDALIDataModule(
            dataset=args.dataset,
            train_data_path=args.train_data_path,
            unique_augs=args.unique_augs,
            transform_kwargs=args.transform_kwargs,
            num_crops_per_aug=args.num_crops_per_aug,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            no_labels=args.no_labels,
            data_fraction=args.data_fraction,
            dali_device=args.dali_device,
            encode_indexes_into_labels=args.encode_indexes_into_labels,
        )
        # dali_datamodule.val_dataloader = lambda: val_loader
    else:
        transform_kwargs = (
            args.transform_kwargs if args.unique_augs > 1 else [args.transform_kwargs]
        )
        transform = prepare_n_crop_transform(
            [prepare_transform(args.dataset, **kwargs) for kwargs in transform_kwargs],
            num_crops_per_aug=args.num_crops_per_aug,
        )
        transform_mixup = copy.deepcopy(transform.transforms[0].transform.transform.transforms)
        transform_mixup.insert(0,torchvision.transforms.ToPILImage())
        model.transform = torchvision.transforms.Compose(transform_mixup)
        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)
        
    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    if args.auto_resume and args.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(args.checkpoint_dir, args.method),
            max_hours=args.auto_resumer_max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(args)
        if resume_from_checkpoint is not None:
            print(
                "Resuming from previous checkpoint that matches specifications:",
                f"'{resume_from_checkpoint}'",
            )
            ckpt_path = resume_from_checkpoint
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    callbacks = []

    if args.save_checkpoint:
        # save checkpoint on last epoch only
        ckpt = Checkpointer(
            args,
            logdir=os.path.join(args.checkpoint_dir, args.method),
            frequency=args.checkpoint_frequency,
        )
        callbacks.append(ckpt)

    if args.auto_umap:
        assert (
            _umap_available
        ), "UMAP is not currently avaiable, please install it first with [umap]."
        auto_umap = AutoUMAP(
            args,
            logdir=os.path.join(args.auto_umap_dir, args.method),
            frequency=args.auto_umap_frequency,
        )
        callbacks.append(auto_umap)

    # wandb logging
    if args.wandb:
        wandb_logger = WandbLogger(
            name=args.name,
            project=args.project,
            entity=args.entity,
            offline=args.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)

        os.system(f'cp -r solo/* {wandb_logger.experiment.dir}')
        os.system(f'cp -r main_pretrain_CL.py {wandb_logger.experiment.dir}')

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    class_per_task = model.class_per_task
    acc_matrix=np.zeros((model.task_num,model.task_num))

    sample_orders = [*range(model.task_num)]
    for i in range(model.task_num):
        if i > model.train_task-1:
            break
        model.train()
        trainer = Trainer.from_argparse_args(
            args, 
            logger=wandb_logger if args.wandb else None,
            callbacks=callbacks,
            enable_checkpointing=False,
            detect_anomaly=True,
            strategy=DDPStrategy(find_unused_parameters=False)
            if args.strategy == "ddp"
            else args.strategy,
            check_val_every_n_epoch = model.max_epochs + 1,
        )
        train = True
        if i < args.train_from_task:
            # model_path
            path = os.path.join(model.CUCL_loadPath, f"{i}.pth")
            # if i == 0:
            #     path = os.path.join('./checkpoints/cifar100_results/byol_CUCL_testp-3p2edh4k', f"{i}.pth")
            save_dict = torch.load(path, map_location='cpu')
            buffer = model.buffers
            msg = model.load_state_dict(save_dict['state_dict'], strict=True)
            model.buffers = buffer
            make_contiguous(model)
            if not args.no_channel_last:
                model = model.to(memory_format=torch.channels_last)
            train = False
        model.curr_task = i
        if args.data_format == "dali":
            order = sample_orders[i]
            dali_datamodule.class_order = (order*class_per_task,(order+1)*class_per_task)
            trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
        else:
            train_dataset = prepare_datasets(args.dataset,transform,train_data_path=args.train_data_path,data_format=args.data_format,\
                no_labels=args.no_labels,data_fraction=args.data_fraction,)
            #which task samples to use
            order = sample_orders[i]
            train_mask = np.logical_and(np.array(train_dataset.targets) >= order*class_per_task,
                np.array(train_dataset.targets) < (order+1)*class_per_task)
            if args.dataset == 'imagenet100':
                train_dataset.samples = np.array(train_dataset.samples)[train_mask]
                train_dataset.imgs = np.array(train_dataset.imgs)[train_mask]
            else:
                train_dataset.data = np.array(train_dataset.data)[train_mask]
                train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order*class_per_task
            train_loader = prepare_dataloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            if train:
                print(f'Training on task {i} samples {order}')
                trainer.fit(model, train_loader, ckpt_path=ckpt_path)
            else:
                trainer.train_dataloader = train_loader
                model.trainer = trainer

        test_model(args, model, i, sample_orders[:i+1], class_per_task, acc_matrix, wandb_logger)
        if model.CUCL:
            if not args.data_format == "dali":
                calculate_Codeword(args, model, train_loader, i)
            if model.buffer_size > 0 and i < model.task_num - 1:
                model.save_samples()

        if train:
            model_path = os.path.join(args.ckpt_dir, args.name + '-' + wandb_logger.version if wandb_logger.version else '', f"{i}.pth")
            os.makedirs(os.path.join(args.ckpt_dir, args.name + '-' + wandb_logger.version if wandb_logger.version else ''),exist_ok=True)
            dict = {'state_dict':model.state_dict(),}
            torch.save(dict, model_path)
            print(f"Task Model saved to {model_path}")


@torch.no_grad()
def test_model(args,model,task_id, sample_orders,N_CLASSES_PER_TASK, acc_matrix, wandb_logger):
    model.eval()
    model = model.cuda()
    for i in range(task_id+1):
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
        order = sample_orders[i]
        train_mask = np.logical_and(np.array(train_dataset.targets) >= order*N_CLASSES_PER_TASK,
        np.array(train_dataset.targets) < (order+1)*N_CLASSES_PER_TASK)
        if args.dataset == 'imagenet100':
            train_dataset.samples = np.array(train_dataset.samples)[train_mask]
            train_dataset.imgs = np.array(train_dataset.imgs)[train_mask]
            train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order*N_CLASSES_PER_TASK
        else:
            train_dataset.data = np.array(train_dataset.data)[train_mask]
            train_dataset.targets = np.array(train_dataset.targets)[train_mask] - order*N_CLASSES_PER_TASK

        test_mask = np.logical_and(np.array(val_dataset.targets) >= order*N_CLASSES_PER_TASK,
        np.array(val_dataset.targets) < (order+1)*N_CLASSES_PER_TASK)
        if args.dataset == 'imagenet100':
            val_dataset.samples = np.array(val_dataset.samples)[test_mask]
            val_dataset.imgs = np.array(val_dataset.imgs)[test_mask]
            val_dataset.targets = np.array(val_dataset.targets)[test_mask] - order*N_CLASSES_PER_TASK
        else:
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

        if args.dataset == 'imagenet100':
            train_targets  = train_targets - order*N_CLASSES_PER_TASK
            test_targets  = test_targets - order*N_CLASSES_PER_TASK

        # plot_codebook(model,test_features['backbone'],test_targets.cpu().numpy(),args,task_id)
        # save_codeword_samples(args, model, val_loader, task_id)
        # count_class_Codeword(args, model, train_loader, task_id)

        # run k-nn for all possible combinations of parameters
        feat_type = args.knn_feature_type
        print(f"\n### {feat_type.upper()} ###")
        distance_fx = args.knn_distance_function
        print("---")
        print(f"Running k-NN with params: distance_fx={distance_fx}, k={args.knn_k}, T={args.knn_temperature}...")
        acc1, acc5 = run_knn(
            train_features=train_features[feat_type],
            train_targets=train_targets,
            test_features=test_features[feat_type],
            test_targets=test_targets,
            k=args.knn_k,
            T=args.knn_temperature,
            distance_fx=distance_fx,
        )
        print(f"Task: {i} Result: acc@1={acc1}, acc@5={acc5}")
        acc_matrix[task_id,i] = acc1
    
    print(f'Task:{task_id} Accuracies =')
    average = []
    for i_a in range(task_id+1):
        print('\t',end='')
        for j_a in range(acc_matrix.shape[1]):
            print('{:5.1f}% '.format(acc_matrix[i_a,j_a]),end='')
        print()
        average.append(acc_matrix[i_a][:i_a+1].mean())
    print ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[task_id].mean()))
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    print ('Backward transfer: {:5.2f}%'.format(bwt))
    print ('Mean Avg Accuracy: {:5.2f}%'.format(np.mean(average)))
    wandb_logger.log_table("AA"+str(task_id), columns=[("Task" + str(i)) for i in range(model.task_num)], data=acc_matrix)

@torch.no_grad()
def save_codeword_samples(args, model, test_loader, task_id):
    quant_idx_bank = []
    data_bank = []
    for im, lab in tqdm(test_loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        quant_idx,Xa,Za = model.foward_CUCL(im,task_id)
        data_bank.append(im)
        quant_idx_bank.append(quant_idx.cpu())
    data_bank = torch.cat(data_bank,dim=0).contiguous()
    quant_idx_bank = torch.cat(quant_idx_bank,dim=0).contiguous()

    for i in range(model.N_books):
        # Za_sn = F.normalize(Za_s[i],p=2,dim=1)
        # dots = torch.mm(Za_sn.data, Za_sn.data.t())
        # n = Za_sn.data.shape[0]
        # dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
        # _, I = torch.max(dots, 1)  # max inner prod -> min distance
        # distances = F.pairwise_distance(Za_sn, Za_sn[I])
        # dis = - torch.log(n * distances).mean()
        path = './code_images/{}/{}'.format(args.name,i)
        id_list = torch.sort(torch.bincount(quant_idx_bank[:,i]),descending=True)[1].tolist()
        for j in id_list:
            path_j = os.path.join(path,f'{j}')
            os.makedirs(path_j,exist_ok=True)
            index = torch.where(quant_idx_bank[:,i] == j)[0]
            images = data_bank[index]
            for k,image in enumerate(images):
                pil_image = transform_convert(image.cpu(),test_loader.dataset.transform)
                pil_image.save(os.path.join(path_j,f'{index[k]}.png'))
            


from PIL import Image
def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, T.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
 
    # C x H x W  ---> H x W x C
    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)
 
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255
 
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()
 
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
    return img

@torch.no_grad()
def calculate_Codeword(args, net, train_loader, task_id):
    net.eval()
    Quant_idx_bank = []
    with torch.no_grad():
        for index, (images1, images2, notaug_images), target in tqdm(train_loader, desc='Feature extracting', leave=False, disable=True):
            quant_idx,Feature,_ = net.foward_CUCL(notaug_images.cuda(non_blocking=True),task_id)
            Quant_idx_bank.append(quant_idx)
        Quant_idx_bank = torch.cat(Quant_idx_bank, dim=0).contiguous()
        codeword_dict = {}
        for i in range(net.N_books):
            total_length = len(Quant_idx_bank[:,i])
            valid_length = total_length*args.CUCL_lambda
            id_list = torch.sort(torch.bincount(Quant_idx_bank[:,i]),descending=True)
            r = np.where((np.cumsum(np.array(id_list[0].cpu()))>=valid_length) == True)[0][0]
            codeword_dict[i] = id_list[1][:r+1].tolist()
            print('Task: {}, Codeword: {} for CodeBook: {}'.format(task_id,id_list[1][:r+1].tolist(),i))
            print()
        net.codeword_dict[task_id] = codeword_dict
        for task, dict in net.codeword_dict.items():
            print('Task: {}, Codeword: {}'.format(task,dict))

@torch.no_grad()
def count_class_Codeword(args, net, train_loader, task_id):
    net.eval()
    Quant_idx_bank = []
    targets = []
    with torch.no_grad():
        for im, lab in tqdm(train_loader):
            quant_idx,Feature,_ = net.foward_CUCL(im.cuda(non_blocking=True),task_id)
            Quant_idx_bank.append(quant_idx)
            targets.append(lab)
        Quant_idx_bank = torch.cat(Quant_idx_bank, dim=0).contiguous()
        targets = torch.cat(targets, dim=0).contiguous()
        print('='*20)
        dict = {}
        for i in targets.unique():
            print('Class: {} '.format(i))
            subdict = {}
            index = torch.where(targets == i)
            quant_idxs = Quant_idx_bank[index]
            for j in range(net.N_books):
                print('CodeBook: {} '.format(j))
                id_list = torch.sort(torch.bincount(quant_idxs[:,j]),descending=True)
                list1 = id_list[1].tolist()
                list0 = id_list[0].tolist()
                print(list1)
                print(list0)
                for k in range(len(list1)):
                    subdict[list1[k]] = list0[k]
                print()
            dict[i.item()] = subdict
        print('='*20)
        print(dict)
        df = pd.DataFrame(dict)
        print(df)
        print('='*20)
        dict = {}
        for i in range(net.N_books):
            print('Codebook: {} '.format(i))
            quantis = Quant_idx_bank[:,i]
            for j in range(net.N_words):
                print('Codeword: {} '.format(j))
                index = torch.where(quantis == j)
                target = targets[index]
                id_list = torch.sort(torch.bincount(target),descending=True)
                print(id_list[1].tolist())
                print(id_list[0].tolist())
                print()

if __name__ == "__main__":
    main()
