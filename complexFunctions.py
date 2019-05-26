<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
=======
>>>>>>> pytorch-Complex-Layers/master
"""
Created on Wed Mar 20 19:35:11 2019

@author: spopoff
"""

<<<<<<< HEAD

from torch.nn.functional import relu, max_pool2d

#%%

=======
from torch.nn.functional import relu, max_pool2d

>>>>>>> pytorch-Complex-Layers/master


def complex_relu(input_r,input_i):
#    assert(input_r.size() == input_i.size())
    return relu(input_r), relu(input_i)

<<<<<<< HEAD
def complex_max_pool2d(input_r,input_i,kernel_size, stride=None, padding=0,
                                dilation=1, ceil_mode=False, return_indices=False):
=======
def complex_max_pool(input_r,input_i,kernel_size, stride, padding,
                                dilation, ceil_mode, return_indices):
>>>>>>> pytorch-Complex-Layers/master
    return max_pool2d(input_r, kernel_size, stride, padding, dilation, 
                      ceil_mode, return_indices), \
           max_pool2d(input_i, kernel_size, stride, padding, dilation, 
                      ceil_mode, return_indices)
