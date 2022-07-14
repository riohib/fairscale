# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy

import pytest
import torch

import fairscale.experimental.wgit.compression as comp


class PreReq:
    """Sets up the pre-requisites needed for each test"""

    def __init__(self, trials) -> None:
        self.rand_tensor = torch.randn(20, 20)
        self.trials = trials

        # TODO: State_dict keys need to be changed to something more encompassing instead of being ad-hoc
        self.state_dict_keys = [
            "conv1.weight",
            "bn1.weight",
            "bn1.bias",
            "layer1.0.conv1.weight",
            "layer1.0.conv2.weight",
            "layer1.1.bn1.weight",
            "layer1.1.bn1.bias",
            "layer1.1.bn1.running_mean",
            "layer1.1.conv1.weight",
            "layer1.1.bn2.weight",
            "layer1.1.bn2.bias",
            "layer1.1.conv2.weight",
            "layer2.0.conv1.weight",
        ]

        self.comp_layers = [
            "conv1.weight",
            "layer1.0.conv2.weight",
            "layer1.1.conv1.weight",
            "layer1.1.conv2.weight",
            "layer2.0.conv1.weight",
        ]

        # initialize a random state_dict
        self.state_dict = self._get_state_dict()

    def _get_state_dict(self):
        state_dict = OrderedDict()
        for keys in self.state_dict_keys:
            m = torch.randint(10, 50, (1,))
            n = torch.randint(10, 50, (1,))
            state_dict[keys] = torch.randn(m, n)
        return state_dict

    @staticmethod
    def sparsity_level(tensor):
        return 1 - torch.count_nonzero(tensor) / tensor.numel()


@pytest.fixture
def pre_reqs():
    pre_req = PreReq(trials=5)
    return pre_req


def test_sparsify(pre_reqs):
    trials = pre_reqs.trials
    for (m, n, sps) in zip(torch.randint(10, 50, (trials,)), torch.randint(10, 50, (trials,)), torch.rand(trials)):
        sps_tens1 = pre_reqs.sparsity_level(comp.sparsify(torch.randn(m, n), sps))
        sps_tens2 = pre_reqs.sparsity_level(comp.sparsify(torch.randn(m, n), sps))

        assert torch.isclose(
            sps_tens1, sps, atol=1e-2
        )  # for sparsity, we only care about upto 2 decimal places of accuracy
        assert torch.isclose(sps_tens2, sps, atol=1e-2)


def test_get_mask(pre_reqs):
    trials = pre_reqs.trials
    for (m, n, sps) in zip(torch.randint(10, 50, (trials,)), torch.randint(10, 50, (trials,)), torch.rand(trials)):

        sps_tens1 = pre_reqs.sparsity_level(comp.get_mask(torch.randn(m, n), sps))
        sps_tens2 = pre_reqs.sparsity_level(comp.get_mask(torch.randn(m, n), sps))

        assert torch.isclose(sps_tens1, sps, atol=1e-2)
        assert torch.isclose(sps_tens2, sps, atol=1e-2)


def test_layerwise_threshold(pre_reqs):
    for i in range(pre_reqs.trials):
        sps = torch.rand(1)
        state_dict_c = copy.deepcopy(pre_reqs.state_dict)
        sps_state_dict = comp.layerwise_threshold(state_dict_c, pre_reqs.comp_layers, sps)

        for key in pre_reqs.state_dict.keys():
            if key in pre_reqs.comp_layers:
                sps_of_tensor = pre_reqs.sparsity_level(sps_state_dict[key])
                assert torch.isclose(sps_of_tensor, sps, atol=1e-2)


def test_fft_threshold(pre_reqs):
    fft_sd = comp.fft_threshold(pre_reqs.state_dict, pre_reqs.comp_layers, 0.0)
    for key in pre_reqs.state_dict.keys():
        if key in pre_reqs.comp_layers:
            is_eq = torch.all(fft_sd[key] == torch.fft.fft(pre_reqs.state_dict[key]))
            assert is_eq


def test_ifft(pre_reqs):
    fft_sd = comp.fft_threshold(pre_reqs.state_dict, pre_reqs.comp_layers, 0.0)
    ifft_sd = comp.inverse_fft(fft_sd, pre_reqs.comp_layers)
    for key in pre_reqs.state_dict.keys():
        if key in pre_reqs.comp_layers:
            fft_ifft_tensor = torch.fft.ifft(torch.fft.fft(pre_reqs.state_dict[key])).real
            is_eq = torch.all(torch.isclose(ifft_sd[key], fft_ifft_tensor, atol=1e-4))
            assert is_eq


def test_delta(pre_reqs):
    fft_sd = comp.fft_threshold(pre_reqs.state_dict, pre_reqs.comp_layers, 0.0)
    ifft_sd = comp.inverse_fft(fft_sd, pre_reqs.comp_layers)
    delta_sd = comp.delta(pre_reqs.state_dict, ifft_sd)  # Compute the delta_W = W - W'

    for key in delta_sd.keys():
        if key in pre_reqs.comp_layers:
            delta = pre_reqs.state_dict[key] - ifft_sd[key]
            # check if the delta function calculates the correct difference between state_d and reconds ifft_sd
            is_eq = torch.all(torch.isclose(delta_sd[key], delta, atol=1e-4))
            assert is_eq


def test_recons_from_del_sparse(pre_reqs):
    fft_sd = comp.fft_threshold(pre_reqs.state_dict, pre_reqs.comp_layers, 0.0)
    ifft_sd = comp.inverse_fft(fft_sd, pre_reqs.comp_layers)
    delta_sd = comp.delta(pre_reqs.state_dict, ifft_sd)  # Compute the delta_W = W - W'

    # # test with threshold at sparsity zero to verify perfect reconstruction after fft -> ifft -> and delta sps
    # delta_sps_sd = comp.layerwise_threshold(delta_sd, pre_reqs.comp_layers, sparsity = 0.0)
    recons_sd = comp.recons_from_del_sparse(delta_sd, ifft_sd)  # Reconstruct the model SD from delta_W and W'

    for key in delta_sd.keys():
        if key in pre_reqs.comp_layers:
            is_eq = torch.all(torch.isclose(pre_reqs.state_dict[key], recons_sd[key], atol=1e-4))
            assert is_eq
