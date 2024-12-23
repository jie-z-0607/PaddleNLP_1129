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

__all__ = ["Simple_Inf_cl_loss", "Matryoshka_Inf_cl_loss"]


class Simple_Inf_cl_loss(nn.Layer):
    def __init__(self, inf_cl_head_dim=64):
        super().__init__()
        self.head_dim = inf_cl_head_dim

    def forward(self, q_reps, p_reps):
        group_size = p_reps.shape[0] // q_reps.shape[0]
        labels = paddle.arange(q_reps.shape[0], dtype="int64")
        labels = labels * group_size
        loss = cal_inf_loss(q_reps, p_reps, labels=labels, scale=None, head_dim=self.head_dim)
        return loss


class Matryoshka_Inf_cl_loss(nn.Layer):
    def __init__(self, embedding_matryoshka_dims: Optional[List[int]] = None, inf_cl_head_dim=64):
        super().__init__()
        if embedding_matryoshka_dims is None:
            self.embedding_matryoshka_dims = []
        else:
            self.embedding_matryoshka_dims = embedding_matryoshka_dims
        self.loss_fn = Simple_Inf_cl_loss(inf_cl_head_dim)

    def forward(self, q_reps, p_reps):
        if len(self.embedding_matryoshka_dims) > 0:
            loss = 0.0
            for dim in self.embedding_matryoshka_dims:
                reduced_q_reps = q_reps[:, :dim]
                reduced_q_reps = nn.functional.normalize(reduced_q_reps, axis=-1)

                reduced_p_reps = p_reps[:, :dim]
                reduced_p_reps = nn.functional.normalize(reduced_p_reps, axis=-1)

                dim_loss = self.loss_fn(reduced_q_reps, reduced_p_reps)
                loss += dim_loss
        else:
            loss = self.loss_fn(q_reps, p_reps)
        return loss