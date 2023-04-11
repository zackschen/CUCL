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
from typing import Any, List, Sequence

import torch
import torch.nn as nn
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.buffer import Buffer
import numpy as np

class BarlowTwins(BaseMethod):
    def __init__(
        self, proj_hidden_dim: int, proj_output_dim: int, lamb: float, scale_loss: float, **kwargs
    ):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Args:
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            proj_output_dim (int): number of dimensions of projected features.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
        """
        kwargs["proj_output_dim"] = proj_output_dim
        super().__init__(**kwargs)
        self.as_backbone = False
        self.lamb = lamb
        self.scale_loss = scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        if self.LUMP:
            self.lump_buffer = Buffer(self.LUMP_size, self.device)

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BarlowTwins, BarlowTwins).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("barlow_twins")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--lamb", type=float, default=0.0051)
        parser.add_argument("--scale_loss", type=float, default=0.024)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def foward_CUCL(self,data,task_id):
        out = super().forward(data)
        z = self.projector(out["feats"])
        Xa,Za = self.quanti_Model(z,task_id)
        quant_idx = self.indexing(self.quanti_Model.C[task_id],Xa)
        return quant_idx, Xa,Za

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """
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

        # ------- barlow twins loss -------
        metrics = {}

        if self.CUCL:
            self.quantization(z1, z2, metrics, out)
            if self.curr_task > 0 and self.buffer_size > 0:
                memory_z1 = out["memory_out1"]["z"]
                memory_z2 = out["memory_out2"]["z"]
                z1 = torch.cat((z1,memory_z1),dim=0).contiguous()
                z2 = torch.cat((z2,memory_z2),dim=0).contiguous()

            if not self.CUCL_for_Loss:
                loss += self.CUCL_lambda*out["quanti_loss"] + out["sample_loss"] + out["cosine_loss"]

            if self.CUCL_for_Loss:
                # z1 = out["Z_Feat"][0]
                z2 = out["Z_Feat"][1]

        if self.LUMP:
            self.lump_buffer.add_data(examples=batch[1][-1], logits=batch[1][1])

        barlow_loss = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)
        loss += barlow_loss
        metrics.update({"train_barlow_loss": barlow_loss,})
        if not self.as_backbone:
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return loss
