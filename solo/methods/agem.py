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
from torch.utils.data import DataLoader
import numpy as np

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger

class AGem(BaseMethod):
    def __init__(
        self,
        **kwargs,
    ):
        super(AGem,self).__init__(**kwargs)
        self.loss = torch.nn.CrossEntropyLoss()
        self.buffer = Buffer(self.extra_args['buffer_size'], self.device)
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

        self.automatic_optimization = False

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]
    
    
    def on_train_end(self) -> None:
        # add data to the buffer
        samples_per_task = self.extra_args['buffer_size'] // self.task_num

        train_loader = DataLoader(self.train_dataset,batch_size=samples_per_task,shuffle=True,num_workers=4,pin_memory=True,drop_last=True,)
        batch_id, (aug_1, aug_2, no_aug), y = next(iter(train_loader))
        self.buffer.add_data(
            examples=aug_1.to(self.device),
            labels=y.to(self.device),
            task_labels=torch.ones(aug_1.shape[0])*self.curr_task
        )
        return super().on_train_end()
    
    def eval_forward(self, X) -> Dict:
        return self.classifier(self.backbone(X.to(self.device)))

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad()

        _, X, targets = batch

        opt.zero_grad()
        labels = targets.to(self.device) - self.curr_task * self.class_per_task
        p = self.classifier(self.backbone(X[0].to(self.device)))[self.curr_task]
        loss = self.loss(p, labels)
        self.manual_backward(loss)
        data_dict = {'loss': loss, 'penalty': 0}

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels, task_labels = self.buffer.get_data(self.batch_size, transform=None)
            opt.zero_grad()
            buf_labels = buf_labels.to(self.device)
            buf_outputs = self.classifier(self.backbone(buf_inputs.to(self.device)))
            new_outputs = []
            new_lables = []
            for i in range(buf_labels.shape[0]):
                    new_outputs.append(buf_outputs[task_labels[i]][i])
                    new_lables.append(buf_labels[i]-task_labels[i]*self.class_per_task)
            buf_outputs = torch.stack(new_outputs)
            buf_labels = torch.stack(new_lables)
            penalty = self.loss(buf_outputs, buf_labels)
            self.manual_backward(penalty)
            data_dict['penalty'] = penalty
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        opt.step()