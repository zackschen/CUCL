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

class DER(BaseMethod):
    def __init__(
        self,
        **kwargs,
    ):
        super(DER,self).__init__(**kwargs)
        self.unsuper_method = kwargs['unsuper_method']
        if self.unsuper_method  == "SimSiam":
            self.net = SimSiam(**kwargs)
        else:
            self.net = BarlowTwins(**kwargs)
        self.net.as_backbone = True
        self.supervised = kwargs['supervised']
        self.train_alpha = kwargs['train_alpha']
        if not self.supervised:
            self.net.classifier = nn.Identity()
        self.loss = torch.nn.CrossEntropyLoss()
        self.buffer = Buffer(self.extra_args["der_size"], self.device)
        self.automatic_optimization = False
    
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiam, SimSiam).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiam")

        # projector
        parser.add_argument("--supervised", action="store_true")
        parser.add_argument("--train_alpha", type=float, default=0.1)
        parser.add_argument("--der_size", type=int, default=256)
        parser.add_argument("--unsuper_method", default="SimSiam", type=str)
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)
        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        parser.add_argument("--lamb", type=float, default=0.0051)
        parser.add_argument("--scale_loss", type=float, default=0.024)
        return parent_parser
    
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
            loss = self.loss(outputs, labels)
        else:
            loss = self.net.training_step(batch,batch_idx)
            outputs = self.net.backbone(X[0].to(self.device))
             
        if not self.buffer.is_empty():
            buf_inputs, buf_logits, task_labels = self.buffer.get_data(
                self.batch_size, transform=None)
            if self.supervised:
                buf_outputs = self.classifier(self.net.backbone(buf_inputs.to(self.device)))
                new_outputs = []
                for i in range(targets.shape[0]):
                    new_outputs.append(buf_outputs[task_labels[i]][i])
                buf_outputs = torch.stack(new_outputs)
            else:
                buf_outputs = self.net.backbone(buf_inputs.to(self.device))
            penalty = self.train_alpha * F.mse_loss(buf_outputs, buf_logits.to(self.device))
            loss += penalty 
        self.manual_backward(loss)
        opt.step()
        self.buffer.add_data(examples=X[0], logits=outputs.data, task_labels=torch.ones(X[-1].shape[0])*self.curr_task)