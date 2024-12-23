# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import paddle
import paddle.nn as nn

from .ring import cal_inf_loss

__all__ = ["SimpleInfclLoss", "MatryoshkaInfclLoss"]


class SimpleInfclLoss(nn.Layer):
    def __init__(self, inf_cl_head_dim=64):
        """
        Initializes the Simple Inf_cl Loss class.

        Args:
            inf_cl_head_dim (int, optional): Dimension of the projection head. Default is 64.
        """
        super().__init__()
        self.head_dim = inf_cl_head_dim

    def forward(self, q_reps, p_reps):
        """
        Computes the instance discrimination loss.

        Args:
            q_reps (Tensor): Query representations.
            p_reps (Tensor): key representations.

        Returns:
            Tensor: The computed loss.
        """
        group_size = p_reps.shape[0] // q_reps.shape[0]  # Number of keys per query
        labels = paddle.arange(q_reps.shape[0], dtype="int64")  # Generate labels for queries
        labels = labels * group_size  # Adjust labels based on group size
        loss = cal_inf_loss(q_reps, p_reps, labels=labels, scale=None, head_dim=self.head_dim)
        return loss


class MatryoshkaInfclLoss(nn.Layer):
    def __init__(self, embedding_matryoshka_dims: Optional[List[int]] = None, inf_cl_head_dim=64):
        """
        Initializes the Matryoshka Inf_cl Loss class.

        Args:
            embedding_matryoshka_dims (List[int], optional): List of dimensions for Matryoshka embeddings.
                If None, no Matryoshka embedding is used. Default is None.
            inf_cl_head_dim (int, optional): Dimension of the projection head. Default is 64.
        """
        super().__init__()
        if embedding_matryoshka_dims is None:
            self.embedding_matryoshka_dims = []
        else:
            self.embedding_matryoshka_dims = embedding_matryoshka_dims
        self.loss_fn = SimpleInfclLoss(inf_cl_head_dim)

    def forward(self, q_reps, p_reps):
        """
        Computes the Matryoshka instance discrimination loss.

        Args:
            q_reps (Tensor): Query representations.
            p_reps (Tensor): key representations.

        Returns:
            Tensor: The computed loss.
        """
        if len(self.embedding_matryoshka_dims) > 0:
            loss = 0.0
            for dim in self.embedding_matryoshka_dims:
                reduced_q_reps = q_reps[:, :dim]  # Reduce query representations to the current Matryoshka dimension
                reduced_q_reps = nn.functional.normalize(
                    reduced_q_reps, axis=-1
                )  # Normalize the reduced query representations along the last axis

                reduced_p_reps = p_reps[:, :dim]  # Reduce key representations to the current Matryoshka dimension
                reduced_p_reps = nn.functional.normalize(
                    reduced_p_reps, axis=-1
                )  # Normalize the reduced key representations along the last axis

                dim_loss = self.loss_fn(
                    reduced_q_reps, reduced_p_reps
                )  # Compute the loss for the current Matryoshka dimension using the internal loss function
                loss += dim_loss
        else:
            loss = self.loss_fn(
                q_reps, p_reps
            )  # If no Matryoshka dimensions are specified, compute the loss using the full representations
        return loss
