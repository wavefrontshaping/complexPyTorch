#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:30:02 2019

@author: Sebastien M. Popoff


Based on https://openreview.net/forum?id=H1T2hmZAb
"""

import torch
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d
from complexFunctions import complex_relu, complex_max_pool2d
from complexFunctions import complex_dropout, complex_dropout2d

class ComplexSequential(Sequential):
    def forward(self, input_r, input_t):
        for module in self._modules.values():
            input_r, input_t = module(input_r, input_t)
        return input_r, input_t

class ComplexDropout(Module):
    def __init__(self,p=0.5, inplace=False):
        super(ComplexDropout,self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self,input_r,input_i):
        return complex_dropout(input_r,input_i,self.p,self.inplace)

class ComplexDropout2d(Module):
    def __init__(self,p=0.5, inplace=False):
        super(ComplexDropout2d,self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self,input_r,input_i):
        return complex_dropout2d(input_r,input_i,self.p,self.inplace)

class ComplexMaxPool2d(Module):

    def __init__(self,kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super(ComplexMaxPool2d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self,input_r,input_i):
        return complex_max_pool2d(input_r,input_i,kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)

class ComplexReLU(Module):

     def forward(self,input_r,input_i):
         return complex_relu(input_r,input_i)

class ComplexConvTranspose2d(Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran_r = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)


    def forward(self,input_r,input_i):
        return self.conv_tran_r(input_r)-self.conv_tran_i(input_i), \
               self.conv_tran_r(input_i)+self.conv_tran_i(input_r)

class ComplexConv2d(Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,input_r, input_i):
#        assert(input_r.size() == input_i.size())
        return self.conv_r(input_r)-self.conv_i(input_i), \
               self.conv_r(input_i)+self.conv_i(input_r)


class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self,input_r, input_i):
        return self.fc_r(input_r)-self.fc_i(input_i), \
               self.fc_r(input_i)+self.fc_i(input_r)

class NaiveComplexBatchNorm1d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self,input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)

class NaiveComplexBatchNorm2d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm2d, self).__init__()
        self.bn_r = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self,input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)

class NaiveComplexBatchNorm1d(Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self,input_r, input_i):
        return self.bn_r(input_r), self.bn_i(input_i)

class _ComplexBatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,3))
            self.bias = Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features,2))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight[:,:2],1.4142135623730951)
            init.zeros_(self.weight[:,2])
            init.zeros_(self.bias)

class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum


        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])


            mean = torch.stack((mean_r,mean_i),dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r-mean_r[None, :, None, None]
            input_i = input_i-mean_i[None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1./n*input_r.pow(2).sum(dim=[0,2,3])+self.eps
            Cii = 1./n*input_i.pow(2).sum(dim=[0,2,3])+self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0,2,3])

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]#+self.eps

            input_r = input_r-mean[None,:,0,None,None]
            input_i = input_i-mean[None,:,1,None,None]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None,:,None,None]*input_r+Rri[None,:,None,None]*input_i, \
                           Rii[None,:,None,None]*input_i+Rri[None,:,None,None]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0,None,None]*input_r+self.weight[None,:,2,None,None]*input_i+\
                               self.bias[None,:,0,None,None], \
                               self.weight[None,:,2,None,None]*input_r+self.weight[None,:,1,None,None]*input_i+\
                               self.bias[None,:,1,None,None]

        return input_r, input_i


class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 2)
        #self._check_input_dim(input)

        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean(dim=0)
            mean_i = input_i.mean(dim=0)
            mean = torch.stack((mean_r,mean_i),dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            # zero mean values
            input_r = input_r-mean_r[None, :]
            input_i = input_i-mean_i[None, :]


            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = input_r.var(dim=0,unbiased=False)+self.eps
            Cii = input_i.var(dim=0,unbiased=False)+self.eps
            Cri = (input_r.mul(input_i)).mean(dim=0)

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]
            # zero mean values
            input_r = input_r-mean[None,:,0]
            input_i = input_i-mean[None,:,1]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None,:]*input_r+Rri[None,:]*input_i, \
                           Rii[None,:]*input_i+Rri[None,:]*input_r

        if self.affine:
            input_r, input_i = self.weight[None,:,0]*input_r+self.weight[None,:,2]*input_i+\
                               self.bias[None,:,0], \
                               self.weight[None,:,2]*input_r+self.weight[None,:,1]*input_i+\
                               self.bias[None,:,1]

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input_r, input_i
