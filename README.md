# complexPyTorch

A high-level toolbox for using complex valued neural networks in PyTorch.

## Complex Valued Networks with PyTorch

Artificial neural networks are mainly used for treating data encoded in real values, such as numerized images or sounds.
In such systems, using complex valued tensor would be quite useless. 
However, for physic related topics, in particular when dealing with wave propagation, using complex values is interesiting as the physics typically has linear, hence more simple, behahiour when considering complex fields. 
complexPyTorch is a simple implementation of complex valued functions and modules using the high-level API of PyTorch. 
Following [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)], it allows the following layers and functions to be used with complex values:
* Linear
* Conv2D
* MaxPool2d
* Relu (&#8450;Relu)
* BatchNorm1d (Naive and Covariance approach)
* BatchNorm2d (Naive and Covariance approach)



## Synthax and usage

The synthax is supposed to copy the one of the standard real functions and modules from PyTorch. 
The names are the same as in `nn.modules` and `nn.functional` except that they start with `Complex`, e.g. `ComplexRelu`, `ComplexMaxPool2D`...
The only usage difference is that the forward fuction takes two tensors, corresponding to real and imaginary parts, and returns two ones too.

## BatchNorm

For all other layers, using the recommendation of [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)], the claculation can be done in a straighforward manner using functions and modules form `nn.modules` and `nn.functional`. 
For instance, the function `complex_relu` in `complexFunctions`, or its associated module `ComplexRelu` in `complexLayers`, simply performs `relu` both on the real and imagary part and returns the two tensors.
The complex BatchNorm proposed in [[C. Trabelsi et al., International Conference on Learning Representations, (2018)](https://openreview.net/forum?id=H1T2hmZAb)] requires the claulcation of the inverse square root of the covariance matrix.
This is implemented in `ComplexbatchNorm1D` and `ComplexbatchNorm2D` but using the high-level PyTorch API, which is quite slow.
The gain of using this approach, however, can be experimentally marginal compared to the naive approach which consist in simply performing the BatchNorm on both thre real and imaginary part, which is available using `NaiveComplexbatchNorm1D` or `NaiveComplexbatchNorm2D`.


## Example

For illustration, here is a small example of a complex model.
Note that in that example, complex values are not particularly useful, it just shows how one can handle complex ANNs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2D

batch_size = 64
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

class ComplexNet(nn.Module):
    
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.bn  = ComplexBatchNorm2D(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
             
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x,_ = self.bn(x,x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
device = torch.device("cuda:3" )
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
```
        
## Todo
* Script ComplexBatchNorm for improved efficiency ([jit doc](https://pytorch.org/docs/stable/jit.html))
