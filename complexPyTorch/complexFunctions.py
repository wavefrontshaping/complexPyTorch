#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: spopoff
"""

import torch
from torch.nn.functional import (
    avg_pool2d,
    dropout,
    dropout2d,
    interpolate,
    max_pool2d,
    relu,
    sigmoid,
    tanh,
)


from torch.nn.functional import max_pool2d, avg_pool2d, dropout, dropout2d, interpolate
from torch import tanh, relu, sigmoid


def complex_matmul(A, B):
    """
    Performs the matrix product between two complex matrices
    """

    outp_real = torch.matmul(A.real, B.real) - torch.matmul(A.imag, B.imag)
    outp_imag = torch.matmul(A.real, B.imag) + torch.matmul(A.imag, B.real)

    return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)


def complex_avg_pool2d(inp, *args, **kwargs):
    """
    Perform complex average pooling.
    """
    absolute_value_real = avg_pool2d(inp.real, *args, **kwargs)
    absolute_value_imag = avg_pool2d(inp.imag, *args, **kwargs)

    return absolute_value_real.type(torch.complex64) + 1j * absolute_value_imag.type(
        torch.complex64
    )


def complex_normalize(inp):
    """
    Perform complex normalization
    """
    real_value, imag_value = inp.real, inp.imag
    real_norm = (real_value - real_value.mean()) / real_value.std()
    imag_norm = (imag_value - imag_value.mean()) / imag_value.std()
    return real_norm.type(torch.complex64) + 1j * imag_norm.type(torch.complex64)


def complex_relu(inp):
    return relu(inp.real).type(torch.complex64) + 1j * relu(inp.imag).type(
        torch.complex64
    )


def complex_sigmoid(inp):
    return sigmoid(inp.real).type(torch.complex64) + 1j * sigmoid(inp.imag).type(
        torch.complex64
    )


def complex_tanh(inp):
    return tanh(inp.real).type(torch.complex64) + 1j * tanh(inp.imag).type(
        torch.complex64
    )


def complex_opposite(inp):
    return -inp.real.type(torch.complex64) + 1j * (-inp.imag.type(torch.complex64))


def complex_stack(inp, dim):
    inp_real = [x.real for x in inp]
    inp_imag = [x.imag for x in inp]
    return torch.stack(inp_real, dim).type(torch.complex64) + 1j * torch.stack(
        inp_imag, dim
    ).type(torch.complex64)


def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-2)
    ).view_as(indices)
    return output


def complex_upsample(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the real and imaginary part and recombining
    """
    outp_real = interpolate(
        inp.real,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    outp_imag = interpolate(
        inp.imag,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return outp_real.type(torch.complex64) + 1j * outp_imag.type(torch.complex64)


def complex_upsample2(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the amplitude and phase part and recombining
    """
    outp_abs = interpolate(
        inp.abs(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    angle = torch.atan2(inp.imag, inp.real)
    outp_angle = interpolate(
        angle,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return outp_abs * (
        torch.cos(outp_angle).type(torch.complex64)
        + 1j * torch.sin(outp_angle).type(torch.complex64)
    )


def complex_max_pool2d(
    inp,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """
    Perform complex max pooling by selecting on the absolute value on the complex values.
    """
    absolute_value, indices = max_pool2d(
        inp.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    # performs the selection on the absolute values
    absolute_value = absolute_value.type(torch.complex64)
    # retrieve the corresponding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    angle = torch.atan2(inp.imag, inp.real)
    # get only the phase values selected by max pool
    angle = _retrieve_elements_from_indices(angle, indices)
    return absolute_value * (
        torch.cos(angle).type(torch.complex64)
        + 1j * torch.sin(angle).type(torch.complex64)
    )


def complex_dropout(inp, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


def complex_dropout2d(inp, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout2d(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp
