# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

gumbel_map: Dict[torch.device, Callable] = {}


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


def top2gating(logits: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    gates = F.softmax(logits, dim=2)
    min_logit = torch.finfo(logits.dtype).min  # type: ignore

    # gates has shape of GSE
    num_tokens = gates.shape[1]
    num_experts = gates.shape[2]
    # capacity = 2S/E
    capacity = 2 * num_tokens // num_experts
    assert num_tokens % num_experts == 0

    # Create a mask for 1st's expert per token
    indices1_gs = torch.argmax(gates, dim=2)
    mask1 = F.one_hot(indices1_gs, num_classes=num_experts)

    # Create a mask for 2nd's expert per token using Gumbel-max trick
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # Replace top-expert with min value
    mins = torch.full_like(logits, min_logit)
    logits_except1 = torch.where(mask1.bool(), mins, logits_w_noise)
    indices2_gs = torch.argmax(logits_except1, dim=2)
    mask2 = F.one_hot(indices2_gs, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=1) - 1
    locations2 = torch.cumsum(mask2, dim=1) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=1, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=1)
    ce = torch.mean(mask1.float(), dim=1)
    l_aux = torch.mean(me * ce)

    # Remove locations outside capacity from mask
    mask1 *= torch.lt(locations1, capacity)
    mask2 *= torch.lt(locations2, capacity)

    # Store the capacity location for each token
    locations1_gs = torch.sum(locations1 * mask1, dim=2)
    locations2_gs = torch.sum(locations2 * mask2, dim=2)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_gs = torch.einsum("gse,gse->gs", gates, mask1_float)
    gates2_gs = torch.einsum("gse,gse->gs", gates, mask2_float)
    denom_gs = gates1_gs + gates2_gs
    # Avoid divide-by-zero
    denom_gs = torch.where(denom_gs > 0, denom_gs, torch.ones_like(denom_gs))
    gates1_gs /= denom_gs
    gates2_gs /= denom_gs

    # Calculate combine_weights and dispatch_mask
    gates1 = torch.einsum("gs,gse->gse", gates1_gs, mask1_float)
    gates2 = torch.einsum("gs,gse->gse", gates2_gs, mask2_float)
    locations1_gsc = F.one_hot(locations1_gs, num_classes=capacity)
    locations2_gsc = F.one_hot(locations2_gs, num_classes=capacity)
    combine1_gsec = torch.einsum("gse,gsc->gsec", gates1, locations1_gsc)
    combine2_gsec = torch.einsum("gse,gsc->gsec", gates2, locations2_gsc)
    combine_weights = combine1_gsec + combine2_gsec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask


class Top2Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(self, model_dim: int, num_experts: int,) -> None:
        super().__init__()
        self.wg = torch.nn.Linear(num_experts, model_dim, bias=False)

    def forward(self, input: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        logits = torch.einsum("gsm,me -> gse", input, self.wg.weight)
        return top2gating(logits)
