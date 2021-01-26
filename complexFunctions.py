#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: spopoff
"""

from torch.nn.functional import relu, max_pool2d, dropout, dropout2d
import torch

def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)

def complex_max_pool2d(input,kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):

    return max_pool2d(input.real, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices).type(torch.complex64) \
           + 1j*max_pool2d(input.imag, kernel_size, stride, padding, dilation,
                      ceil_mode, return_indices).type(torch.complex64)

def complex_dropout(input, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part, 
    # this not a clean solution!
    mask = torch.ones_like(input).type(torch.float32)
    mask = dropout(mask, p, training)*1/(1-p)
    return mask*input


def complex_dropout2d(input, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part, 
    # this not a clean solution!
    mask = torch.ones_like(input).type(torch.float32)
    mask = dropout2d(mask, p, training)*1/(1-p)
    return mask*input
