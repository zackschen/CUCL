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

import argparse
from posixpath import join
from re import S
from typing import Any, Dict, List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseMethod
from solo.utils.buffer import Buffer
import numpy as np
from solo.utils.knn import WeightedKNNClassifier

class SimSiam(BaseMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        **kwargs,
    ):
        """Implements SimSiam (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """
        
        kwargs["proj_output_dim"] = proj_output_dim
        super().__init__(**kwargs)
        self.as_backbone = False
        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN
        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        if self.LUMP:
            self.lump_buffer = Buffer(self.LUMP_size, self.device)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiam, SimSiam).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiam")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        return parent_parser


    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        extra_learnable_params: List[dict] = [
            {"name": "projector","params": self.projector.parameters()},
            {"name": "predictor","params": self.predictor.parameters(), "static_lr": True},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out
    
    def foward_CUCL(self,data,task_id):
        out = super().forward(data)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        Xa,Za = self.quanti_Model(p,task_id)
        quant_idx = self.indexing(self.quanti_Model.C[task_id],Xa)
        return quant_idx, Xa,Za

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimSiam reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimSiam loss and classification loss.
        """
        # opt = self.optimizers()
        # opt.zero_grad()
        if self.LUMP:
            if not self.lump_buffer.is_empty():
                buf_inputs, buf_inputs1 = self.lump_buffer.get_data(
                    self.batch_size, transform=self.transform)
                lam = np.random.beta(self.LUMP_lambda, self.LUMP_lambda)
                mixed_x = lam * batch[1][0].to(self.device) + (1 - lam) * buf_inputs[:batch[1][0].shape[0]].to(self.device)
                mixed_x_aug = lam * batch[1][1].to(self.device) + (1 - lam) * buf_inputs1[:batch[1][1].shape[0]].to(self.device)
                batch[1][0] = mixed_x
                batch[1][1] = mixed_x_aug

        out = super().training_step(batch, batch_idx)
        loss = out["loss"]
        z1, z2 = out["z"]
        p1, p2 = out["p"]

        metrics = {}
        if self.CUCL:
            self.quantization(p1, p2, metrics, out)
            if self.curr_task > 0 and self.buffer_size > 0:
                memory_z1, memory_z2 = out["memory_out1"]["z"], out["memory_out2"]["z"]
                memory_p1, memory_p2 = out["memory_out1"]["p"], out["memory_out2"]["p"]
                p1 = torch.cat((p1,memory_p1),dim=0).contiguous()
                p2 = torch.cat((p2,memory_p2),dim=0).contiguous()
                z1 = torch.cat((z1,memory_z1),dim=0).contiguous()
                z2 = torch.cat((z2,memory_z2),dim=0).contiguous()

            if not self.CUCL_for_Loss:
                loss += self.CUCL_lambda*out["quanti_loss"] + out["sample_loss"]

            if self.CUCL_for_Loss:
                p1 = out["Z_Feat"][0]
                p2 = out["Z_Feat"][1]
        
        
        if self.LUMP:
            self.lump_buffer.add_data(examples=batch[1][-1], logits=batch[1][1])

        # ------- negative cosine similarity loss -------
        neg_cos_sim = simsiam_loss_func(p1, z2) / 2 + simsiam_loss_func(p2, z1) / 2
        loss += neg_cos_sim

        metrics.update({"train_neg_cos_sim": neg_cos_sim,})
        if not self.as_backbone:
            self.log_dict(metrics, on_epoch=True, sync_dist=True)


        return loss

    def base_validation_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Allows user to re-write how the forward step behaves for the validation_step.
        Should always return a dict containing, at least, "loss", "acc1" and "acc5".
        Defaults to _base_shared_step
        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.
        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        return self._base_shared_step(X, targets)

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.
        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.
        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self.base_validation_step(X, targets)

        if not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
        }
        return metrics

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        log = {}
        if not self.trainer.sanity_checking:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)

