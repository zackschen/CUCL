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
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.mocov2plus import mocov2plus_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.misc import gather


class MoCoV2Plus(BaseMomentumMethod):
    queue: torch.Tensor

    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        queue_size: int,
        **kwargs
    ):
        """Implements MoCo V2+ (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """
        kwargs["proj_output_dim"] = proj_output_dim
        super().__init__(**kwargs)

        self.temperature = temperature
        self.queue_size = queue_size

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # create the queue
        self.register_buffer("queue", torch.randn(2, proj_output_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(MoCoV2Plus, MoCoV2Plus).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("mocov2plus")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)

        # queue settings
        parser.add_argument("--queue_size", default=65536, type=int)

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters together with parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner.

        Args:
            keys (torch.Tensor): output features of the momentum backbone.
        """

        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the online backbone and projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and query.
        """

        out = super().forward(X)
        z = F.normalize(self.projector(out["feats"]), dim=-1)
        out.update({"z": z})
        return out

    def foward_CUCL(self,data,task_id):
        out = super().forward(data)
        z = F.normalize(self.projector(out["feats"]), dim=-1)
        Xa,Za = self.quanti_Model(z,task_id)
        quant_idx = self.indexing(self.quanti_Model.C[task_id],Xa)
        return quant_idx, Xa,Za

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the key.
        """

        out = super().momentum_forward(X)
        z = F.normalize(self.momentum_projector(out["feats"]), dim=-1)
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo V2+ reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_large_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MoCo loss and classification loss.

        """
        out = super().training_step(batch, batch_idx)
        loss = out["loss"]
        q1, q2 = out["z"]
        k1, k2 = out["momentum_z"]

        final_q1 = q1
        final_q2 = q2
        final_k1 = k1
        final_k2 = k2
        metrics = {}
        if self.CUCL:
            self.quantization(q1, q2,  metrics, out)
            if self.curr_task > 0 and self.buffer_size > 0:
                memory_q1 = out["memory_out1"]["z"]
                memory_q2 = out["memory_out2"]["z"]
                memory_k1 = out["memory_out1"]["momentum_z"]
                memory_k2 = out["memory_out2"]["momentum_z"]
                final_q1 = torch.cat((q1,memory_q1),dim=0).contiguous()
                final_q2 = torch.cat((q2,memory_q2),dim=0).contiguous()
                final_k1 = torch.cat((k1,memory_k1),dim=0).contiguous()
                final_k2 = torch.cat((k2,memory_k2),dim=0).contiguous()

                # final_q1 = torch.cat((q1,memory_q1),dim=0).contiguous()
                # final_q2 = torch.cat((q2,memory_q2),dim=0).contiguous()
                # final_k1 = torch.cat((k1,memory_k1),dim=0).contiguous()
                # final_k2 = torch.cat((k2,memory_k2),dim=0).contiguous()
            loss += self.CUCL_lambda*out["quanti_loss"] + out["sample_loss"]
            
        queue = self.queue.clone().detach()
        nce_loss = (
            mocov2plus_loss_func(final_q1, final_k2, queue[1], self.temperature)
            + mocov2plus_loss_func(final_q2, final_k1, queue[0], self.temperature)
        ) / 2
        loss += nce_loss

        # ------- update queue -------
        keys = torch.stack((gather(final_k1), gather(final_k2)))
        self._dequeue_and_enqueue(keys)
        metrics.update({"train_nce_loss": nce_loss})
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return loss