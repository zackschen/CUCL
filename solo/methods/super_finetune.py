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

class su_Finetune(BaseMethod):
    def __init__(
        self,
        **kwargs,
    ):
        super(su_Finetune,self).__init__(**kwargs)
        self.loss = torch.nn.CrossEntropyLoss()
        if kwargs["multitask"]:
            self.multitask = True
            self.classifier = nn.Linear(self.features_dim, self.num_classes)

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
    
    def eval_forward(self, X) -> Dict:
        return self.classifier(self.backbone(X.to(self.device)))
        
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        _, X, targets = batch

        labels = targets.to(self.device)
        inputs = X[0].to(self.device)
        metrics = {}
        if self.multitask:
            outputs = self.classifier(self.backbone(inputs))
        else:
            outputs = self.classifier(self.backbone(inputs))[self.curr_task]
        loss = self.loss(outputs, labels)
        metrics.update({"train_loss": loss,})
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        return loss
