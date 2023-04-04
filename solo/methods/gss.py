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
from solo.utils.gss_buffer import Buffer
import numpy as np

class GSS(BaseMethod):
    def __init__(
        self,
        **kwargs,
    ):
        super(GSS,self).__init__(**kwargs)
        self.buffer = Buffer(self.extra_args["buffer_size"], self.device,
                            self.batch_size, self)
        self.alj_nepochs = 1 
        self.loss = torch.nn.CrossEntropyLoss()
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
    
    def get_grads(self, inputs, labels):
        opt = self.optimizers()

        self.backbone.eval()
        opt.zero_grad()
        labels = labels.to(self.device)
        outputs = self.classifier(self.backbone(inputs.to(self.device)))[self.curr_task]
        loss = self.loss(outputs, labels)
        self.manual_backward(loss)
        grads = self.get_backbone_grads().clone().detach()
        opt.zero_grad()
        self.backbone.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads
    
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
        for pp in list(self.backbone.parameters()):
            grads.append(pp.grad.contiguous().view(-1))
        return grads
    
    def eval_forward(self, X) -> Dict:
        return self.classifier(self.backbone(X.to(self.device)))
        
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad()

        _, X, targets = batch

        real_batch_size = X[0].shape[0]
        self.buffer.drop_cache()
        self.buffer.reset_fathom()
        labels = targets.to(self.device)

        for _ in range(self.alj_nepochs):
            opt.zero_grad()
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.batch_size, transform=self.transform)
                buf_inputs = buf_inputs.to(self.device)
                buf_labels = buf_labels.to(self.device)
                tinputs = torch.cat((X[0].to(self.device), buf_inputs))
                tlabels = torch.cat((labels, buf_labels))
            else:
                tinputs = X[0].to(self.device)
                tlabels = labels

            outputs = self.classifier(self.backbone(tinputs))[self.curr_task]
            loss = self.loss(outputs, tlabels)
            self.manual_backward(loss)
            opt.step()

        self.buffer.add_data(examples=X[-1],
                             labels=labels[:real_batch_size])