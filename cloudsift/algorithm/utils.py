# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 02:45:31 2023

@author: Shahir
"""

import torch


def calc_model_size(
        model: torch.nn.Module) -> int:

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return param_size + buffer_size
