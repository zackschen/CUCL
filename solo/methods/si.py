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
from solo.methods.gem import overwrite_grad
from solo.methods.gem import store_grad
from solo.utils.buffer import Buffer
from solo.methods.simsiam import SimSiam
from solo.methods.barlow_twins import BarlowTwins
import numpy as np

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger

class SI(BaseMethod):
    def __init__(
        self,
        **kwargs,
    ):
        super(SI,self).__init__(**kwargs)
        self.unsuper_method = kwargs['unsuper_method']
        if self.unsuper_method  == "SimSiam":
            self.net = SimSiam(**kwargs)
        else:
            self.net = BarlowTwins(**kwargs)
        self.net.as_backbone = True
        self.supervised = kwargs['supervised']
        if not self.supervised:
            self.net.classifier = nn.Identity()
        self.loss = torch.nn.CrossEntropyLoss()
        self.checkpoint = self.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0
        self.c = kwargs['train_c']
        self.xi = 1.0

        self.automatic_optimization = False

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        if self.supervised:
            return [
                {"name": "backbone", "params": self.net.backbone.parameters()},
                {
                    "name": "classifier",
                    "params": self.classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                },
            ]
        else:
            return [{"name": "backbone", "params": self.net.parameters()}]
        
    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        if self.supervised:
            for pp in list(self.net.backbone.parameters()):
                params.append(pp.contiguous().view(-1))
        else:
            for name, pp in list(self.net.named_parameters()):
                if not 'bias' in name:
                    params.append(pp.contiguous().view(-1))
        return torch.cat(params)

    def get_backbone_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        return torch.cat(self.get_backbone_grads_list())
    
    def get_backbone_grads_list(self):
        """
        Returns a list containing the gradients (a tensor for each layer).
        :return: gradients list
        """
        grads = []
        if self.supervised:
            for pp in list(self.net.backbone.parameters()):
                grads.append(pp.grad.contiguous().view(-1))
        else:
            for name, pp in list(self.net.named_parameters()):
                if not 'bias' in name:
                    grads.append(pp.grad.contiguous().view(-1))
        return grads
    
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiam, SimSiam).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiam")

        # projector
        parser.add_argument("--supervised", action="store_true")
        parser.add_argument("--train_c", type=float, default=0.1)
        parser.add_argument("--unsuper_method", default="SimSiam", type=str)
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        parser.add_argument("--lamb", type=float, default=0.0051)
        parser.add_argument("--scale_loss", type=float, default=0.024)
        return parent_parser
    
        
    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty
    
    
    def on_train_end(self) -> None:
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.get_params()).to(self.device)

        self.big_omega = self.small_omega / ((self.get_params().data - self.checkpoint.to(self.device)) ** 2 + self.xi)

        self.checkpoint = self.get_params().data.clone().to(self.device)
        self.small_omega = 0
        return super().on_train_end()

    def foward_CUCL(self,data,task_id):
        return self.net.foward_CUCL(data,task_id)

    def forward(self, X) -> Dict:
        if self.supervised:
            return super().forward(X)
        else:
            return self.net(X)
        
    def eval_forward(self, X) -> Dict:
        if self.supervised:
            return self.classifier(self.net.backbone(X.to(self.device)))
        else:
            return self.net(X)
        
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad()

        _, X, targets = batch

        if self.supervised:
            labels = targets.to(self.device) - self.curr_task * self.class_per_task
            outputs = self.classifier(self.net.backbone(X[0].to(self.device)))[self.curr_task]
            penalty = self.c * self.penalty()
            loss = self.loss(outputs, labels).mean() + penalty
            self.manual_backward(loss)
            nn.utils.clip_grad.clip_grad_value_(self.net.parameters(), 1)
        else:
            loss = self.net.training_step(batch,batch_idx)
            penalty = self.c * self.penalty() 
            loss = loss + penalty
            self.manual_backward(loss)
            self.log_dict({"loss": loss}, on_epoch=True, sync_dist=True)
        opt.step()

        self.small_omega += self.lr * self.get_backbone_grads().data ** 2