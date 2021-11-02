#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.tensor_utils import unfold2d, unfold3d

from .utils import register_grad_sampler
import torch.nn.functional as F


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for Conv2d layers.

    args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    n = activations.shape[0]
    # get activations and backprops in shape depending on the Conv layer
    if type(layer) == nn.Conv2d:
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    elif type(layer) == nn.Conv1d:
        activations = activations.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        activations = torch.nn.functional.unfold(
            activations,
            kernel_size=(1, layer.kernel_size[0]),
            padding=(0, layer.padding[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
    elif type(layer) == nn.Conv3d:
        activations = unfold3d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    backprops = backprops.reshape(n, -1, activations.shape[-1])
    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    grad_sample = torch.einsum("noq,npq->nop", backprops, activations)
    # rearrange the above tensor and extract diagonals.
    grad_sample = grad_sample.view(
        n,
        layer.groups,
        -1,
        layer.groups,
        int(layer.in_channels / layer.groups),
        np.prod(layer.kernel_size),
    )
    grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
    shape = [n] + list(layer.weight.shape)

    ret = {layer.weight: grad_sample.view(shape)}
    if layer.bias is not None:
        ret[layer.bias] = torch.sum(backprops, dim=[-1, -2])

    return ret


# @register_grad_sampler([nn.Conv2d])
def convolution2d_backward_as_a_convolution(
    layer: nn.Conv2d,
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for Conv2d layers.

    args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    batch_size = activations.shape[0]
    I = activations.shape[1]
    O = backprops.shape[1]

    # activations has shape (B, I, H, W)
    # backprops has shape (B, O, H, W)
    activations_ = activations.view(batch_size, layer.groups, I // layer.groups, activations.shape[2], activations.shape[3]) # (B, G, I/G, H, W)

    activations_ = activations_.view(activations_.shape[0] * activations_.shape[1], activations_.shape[2], activations_.shape[3], activations_.shape[4]) # (B*G, I / G, H, W)
    activations_ = activations_.transpose(0, 1) # (I / G, B * G, H, W)
    backprops_ = backprops.view(backprops.shape[0] * backprops.shape[1], 1, backprops.shape[2], backprops.shape[3]) # (B*O, 1, H, W)

    # Without groups (I, B, H, W) X (B*O, 1, H, W) -> (I, B*O, H, W)
    # With groups (I / G, B*G, H, W) X (B*O, 1, H, W) -> (I / G, B * O, H, W)
    weight_grad_sample = F.conv2d(activations_, backprops_, bias=None, dilation=layer.stride, padding=layer.padding, stride=layer.dilation, groups=batch_size * layer.groups)
    weight_grad_sample = weight_grad_sample.view(I // layer.groups, batch_size, O, *weight_grad_sample.shape[-2:]) # (I / G, B, O, H, W)
    weight_grad_sample = weight_grad_sample.movedim(0, 2) # (B, O, I/G, H, W)
    weight_grad_sample = weight_grad_sample[:, :, :, :layer.weight.shape[2], :layer.weight.shape[3]]

    ret = {
        layer.weight: weight_grad_sample
    }
    if layer.bias is not None:
        ret[layer.bias] = torch.sum(backprops, dim=[-1, -2])

    return ret