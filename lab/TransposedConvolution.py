import torch
import torch.nn as nn
from torch.autograd import Variable

a = torch.Tensor([[1,2,3],[3,2,1],[1,2,3]]).view(1, 1, 3, 3) # (N, C_IN, H, W)
k = torch.Tensor([[1,2],[3,4]]).view(1, 1, 2, 2) # (C_OUT, C_IN, K[0], K[1])
print('Input:', a)
print('Kernel:', k)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv_transpose_1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2,
                                                   stride=2, padding=1)
        self.conv_transpose_1.weight = nn.Parameter(k)
        self.conv_transpose_1.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.conv_transpose_1(x)
        return x

net = Net()
print(net)

params = list(net.parameters())
print('\nLearnable parameters:\n', params)

out = net(Variable(a))
print('Output:', out)

print('Or, calculate directly with ConvTranspose2d:')
conv_transpose_1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1)
conv_transpose_1.weight = nn.Parameter(k)
conv_transpose_1.bias = nn.Parameter(torch.zeros(1))
print(conv_transpose_1(Variable(a)))

print('--- Alternative View ---')
b = torch.Tensor([[1,0,2,0,3], [0,0,0,0,0],[3,0,2,0,1],[0,0,0,0,0],[1,0,2,0,3]]).view(1, 1, 5, 5)
print('Input dilated:', b)

print('Direct convolution with Conv2d:')
conv_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)
conv_1.weight = nn.Parameter(k)
conv_1.bias = nn.Parameter(torch.zeros(1))
print(conv_1(Variable(b)))

print('The result is wrong, however if we reverse kernel:')
k = torch.Tensor([[4,3],[2,1]]).view(1, 1, 2, 2) # (C_OUT, C_IN, K[0], K[1])
print("Kernel:", k)
conv_1.weight = nn.Parameter(k)
result = conv_1(Variable(b))
print('Result:', result)
print('This shows that the dilated convolution does not share the same kernel\n'+
      'with the transposed convolution. In particular, they seem to be the reverse\n'+
      'of each other. Anyway, it is the desired shape we wish to achieve.')
