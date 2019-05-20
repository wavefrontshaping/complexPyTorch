# complexPyTorch

A high-level toolbox for using complex valued neural networks in PyTorch.

## Complex Valued Networks with PyTorch

Artificial neural networks are mainly used for treating data encoded in real values, such as digitized images or sounds. 
In such systems, using complex-valued tensor would be quite useless. 
However, for physic related topics, in particular when dealing with wave propagation, using complex values is interesting as the physics typically has linear, hence more simple, behavior when considering complex fields. 
complexPyTorch is a simple implementation of complex-valued functions and modules using the high-level API of PyTorch. 
Following [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)], it allows the following layers and functions to be used with complex values:
* Linear
* Conv2d
* MaxPool2d
* Relu (&#8450;Relu)
* BatchNorm1d (Naive and Covariance approach)
* BatchNorm2d (Naive and Covariance approach)



## Syntax and usage

The syntax is supposed to copy the one of the standard real functions and modules from PyTorch. 
The names are the same as in `nn.modules` and `nn.functional` except that they start with `Complex` for Modules, e.g. `ComplexRelu`, `ComplexMaxPool2d` or `complex_` for functions, e.g. `complex_relu`, `complex_max_pool2d`.
The only usage difference is that the forward function takes two tensors, corresponding to real and imaginary parts, and returns two ones too.

## BatchNorm

For all other layers, using the recommendation of [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)], the calculation can be done in a straightforward manner using functions and modules form `nn.modules` and `nn.functional`. 
For instance, the function `complex_relu` in `complexFunctions`, or its associated module `ComplexRelu` in `complexLayers`, simply performs `relu` both on the real and imaginary part and returns the two tensors.
The complex BatchNorm proposed in [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)] requires the calculation of the inverse square root of the covariance matrix.
This is implemented in `ComplexbatchNorm1D` and `ComplexbatchNorm2D` but using the high-level PyTorch API, which is quite slow.
The gain of using this approach, however, can be experimentally marginal compared to the naive approach which consists in simply performing the BatchNorm on both the real and imaginary part, which is available using `NaiveComplexbatchNorm1D` or `NaiveComplexbatchNorm2D`.


## Example

For illustration, here is a small example of a complex model.
Note that in that example, complex values are not particularly useful, it just shows how one can handle complex ANNs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d

batch_size = 64
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

class ComplexNet(nn.Module):
    
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 20, 5, 1)
        self.bn  = ComplexBatchNorm2d(20)
        self.conv2 = ComplexConv2d(20, 50, 5, 1)
        self.fc1 = ComplexLinear(4*4*50, 500)
        self.fc2 = ComplexLinear(500, 10)
             
    def forward(self,x):
        xr = x
        # imaginary part to zero
        xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)
        xr,xi = self.conv1(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = complex_max_pool2d(xr,xi, 2, 2)
        
        
        xr,xi = self.bn(xr,xi)
        xr,xi = self.conv2(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = complex_max_pool2d(xr,xi, 2, 2)
        
        xr = xr.view(-1, 4*4*50)
        xi = xi.view(-1, 4*4*50)
        xr,xi = self.fc1(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = self.fc2(xr,xi)
        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))
        return F.log_softmax(x, dim=1)
    
device = torch.device("cuda:0" )
model = ComplexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item())
            )

# Run training on 50 epochs
for epoch in range(50):
    train(model, device, train_loader, optimizer, epoch)
```
        
## Todo
* Script ComplexBatchNorm for improved efficiency ([jit doc](https://pytorch.org/docs/stable/jit.html))

## Acknowledgments

I want to thank Piotr Bialecki for his invaluable help on the PyTorch forum.
