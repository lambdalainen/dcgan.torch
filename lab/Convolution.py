import torch
import torch.nn as nn
from torch.autograd import Variable

a = torch.Tensor([[1,2,3,4],[4,3,2,1],[1,2,3,4],[4,3,2,1]]).view(1, 1, 4, 4) # (N, C_IN, H, W)
k = torch.Tensor([[1,2],[3,4]]).view(1, 1, 2, 2) # (C_OUT, C_IN, K[0], K[1])
print('Input:', a)
print('Kernel:', k)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1)
        self.conv1.weight = nn.Parameter(k)
        self.conv1.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.conv1(x)
        return x

net = Net()
print(net)

params = list(net.parameters())
print('\nLearnable parameters:\n', params)

out = net(Variable(a))
print('Output:', out)

print('Or, calculate directly with Conv2d:')
conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2, padding=1)
conv1.weight = nn.Parameter(k)
conv1.bias = nn.Parameter(torch.zeros(1))
print(conv1(Variable(a)))

print('--- Alternative View ---')
b = torch.Tensor([[0,0,0,0,0,0], [0,1,2,3,4,0],[0,4,3,2,1,0],[0,1,2,3,4,0],[0,4,3,2,1,0],[0,0,0,0,0,0]])
b = b.view(36, 1)
print('Input padded, flattened, shown transposed:', torch.t(b))
# c is very sparse, each row corresponds to 1 patch
c = torch.Tensor([
      [0,0,0,0,0,0],[0,4,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
      [0,0,0,0,0,0],[0,0,3,4,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
      [0,0,0,0,0,0],[0,0,0,0,3,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,2,0,0,0,0],[0,4,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,2,0,0],[0,0,3,4,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,3,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,2,0,0,0,0],[0,0,0,0,0,0],
      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,2,0,0],[0,0,0,0,0,0],
      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,0]])
c = c.view(9, 36)
print('The sparse matrix is of 9x36 (not shown)')
r = torch.matmul(c, b)
print('Multiplication result (shown transposed):', torch.t(r))
print('Transposed multiplication result (reshaped to 6x6):', torch.matmul(torch.t(c), r).view(6,6))
print('(As can be seen, the padding is preserved)')

print('Now, this could be done with a 9x16 * 16x1 way')
b = torch.Tensor([[1,2,3,4],[4,3,2,1],[1,2,3,4],[4,3,2,1]])
b = b.view(16, 1)
print('Input flattened, shown transposed:', torch.t(b))
# c is very sparse, each row corresponds to 1 patch
c = torch.Tensor([
      [4,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
      [0,3,4,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
      [0,0,0,3],[0,0,0,0],[0,0,0,0],[0,0,0,0],
      [0,0,0,0],[2,0,0,0],[4,0,0,0],[0,0,0,0],
      [0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0],
      [0,0,0,0],[0,0,0,1],[0,0,0,3],[0,0,0,0],
      [0,0,0,0],[0,0,0,0],[0,0,0,0],[2,0,0,0],
      [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,1,2,0],
      [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]])
c = c.view(9, 16)
print('The sparse matrix is of 9x16 (not shown)')
r = torch.matmul(c, b)
print('Multiplication result (shown transposed):', torch.t(r))
print('Transposed multiplication result (reshaped to 4x4):', torch.matmul(torch.t(c), r).view(4,4))
print('(As can be seen, the result is the same as above)')
