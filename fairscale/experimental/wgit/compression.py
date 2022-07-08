# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


def sparsify(in_tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Prune a tensor with a certain sparsity level using a threshold corresponding to that sparsity.

    Args:
        in_tensor (torch.Tensor)
            input dense tensor
        sparsity (float)
            target sparsity for tensor sparsification
    """
    abs_tensor = torch.abs(in_tensor)
    threshold = torch.quantile(abs_tensor, sparsity)  # type: ignore
    sps_tensor = torch.where(abs_tensor < threshold, torch.tensor(0.0, device=in_tensor.device), in_tensor)
    return sps_tensor


def get_mask(in_tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Get a mask for a tensor corresponding to a certain sparsity level.

    Args:
        in_tensor (torch.Tensor)
            input torch tensor for which sparse mask is generated.
        sparsity (float)
            target sparsity of the tensor for mask generation.

    """
    abs_tensor = torch.abs(in_tensor)
    if sparsity == 0.0:  # if sparsity is zero, we want a mask with all 1's
        threshold = torch.tensor(float("-Inf"), device=in_tensor.device)
    else:
        threshold = torch.quantile(abs_tensor, sparsity)  # type: ignore
    return abs_tensor > threshold


def layerwise_threshold(state_dict: OrderedDict, comp_layers: List[str], sparsity: float) -> OrderedDict:
    """Threshold a state_dict's weights layerwise with a target sparsity level in interval [0, 1]

    Args:
        state_dict (OrderedDict)
            state_dict of a dense model.
        comp_layers (List[str])
            A list of layer names which should be thresholded.
        sparsity (float)
            target sparsity of each tensors in the state_dict
    """
    for name, weight in state_dict.items():
        if name in comp_layers:
            weight.data = sparsify(weight.data, sparsity=sparsity)
    return state_dict


def fft_threshold(state_dict: OrderedDict, comp_layers: List[str], sparsity: float = 0.0) -> OrderedDict:
    """Perform a fourier transform on the input state_dict layerwise. If sparsity is not zero, additionally
    perform a thresholding in the layers of the transformed (to the frequency domain) state_dict.

    Args:
        state_dict (OrderedDict)
            state_dict of a dense model.
        comp_layers (List[str])
            A list of layer names which should be compressed.
        sparsity (float)
            target sparsity of each tensors in the state_dict
    """
    signal_sparse_sd = OrderedDict()  # state_dict with layer tensors as signal sparse tensor (SST)
    for name, weight in state_dict.items():
        if name in comp_layers:
            fft_w = torch.fft.fft(weight.data)  # type: ignore # apply FFT
            fft_w_abs = torch.abs(fft_w)  # get absolute FFT values (make real)
            sps_mask = get_mask(fft_w_abs, sparsity)  # use the normalized real values for getting the topk mask
            signal_sparse_sd[name] = fft_w * sps_mask  # but mask the actual complex FFT values topk
        else:
            signal_sparse_sd[name] = weight
    return signal_sparse_sd


def inverse_fft(fft_state_dict: OrderedDict, comp_layers: List[str]) -> OrderedDict:
    """Perform an inverse fourier transform on the fourier transformed input state_dict.

    Args:
        fft_state_dict (OrderedDict)
            A state_dict with the layer weight tensors transformed by FFT
        comp_layers (List[str])
            A list of layer names which should be compressed.
        sparsity (float)
            target sparsity of each tensors in the state_dict
    """
    ifft_state_dict = OrderedDict()
    for name, weight in fft_state_dict.items():
        if name in comp_layers:
            # IFFT on the sparse complex topk and save only the real values
            ifft_state_dict[name] = torch.fft.ifft(fft_state_dict[name]).real  # type: ignore
        else:
            ifft_state_dict[name] = weight
    return ifft_state_dict


def delta_sparse(state_dict: OrderedDict, recon_state_dict: OrderedDict) -> OrderedDict:
    """Calculate the difference between the weights reconstructed from the FFT compressions and
    subsequent iFFT.

    Args:
        state_dict (OrderedDict)
            A model state_dict with the layer weight tensors of the dense model
        recon_state_dict (OrderedDict)
            A model state_dict with the layer weight tensors reconstructed by IFFT after having
            been transformed with FFT at first.
    """
    delta_sps_sd = OrderedDict()
    for (name, weight), (name, weight_p) in zip(state_dict.items(), recon_state_dict.items()):
        delta_sps_sd[name] = weight - weight_p
    return delta_sps_sd


def recons_from_del_sparse(delta_sps_sd: OrderedDict, inverse_recon_sd: OrderedDict) -> OrderedDict:
    """Reconstruct the layer weights from the Delta Sparse Tensors and the IFFT Reconstructed tensors.
    This function takes both the state_dict of Delta Sparse Tensors (DST) and the IFFT reconstructed
    tensors as inputs. It returns a reconstructed state_dict from the inputs state_dicts.

    Args:
        delta_sps_sd (OrderedDict)
            A model state_dict with the layer weight tensors of the dense model
        recon_state_dict (OrderedDict)
            A model state_dict with the layer weight tensors reconstructed by IFFT after having
            been transformed with FFT at first.
    """
    recon_sps_sd = OrderedDict()
    for (name, weight_s), (name, weight_p) in zip(delta_sps_sd.items(), inverse_recon_sd.items()):
        recon_sps_sd[name] = weight_s + weight_p
    return recon_sps_sd


def compressible_layers(model: nn.Module) -> List[str]:
    """Finds out the compressible layers of a neural network and returns a list of the names of
    those layers.

    Args:
        model (nn.Module)
            A model state_dict with the layer weight tensors of the dense model
        recon_state_dict (OrderedDict)
            A model state_dict with the layer weight tensors reconstructed by IFFT after having
            been transformed with FFT at first.
    """
    comp_layers = list()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            comp_layers.append(name + ".weight")
    return comp_layers
