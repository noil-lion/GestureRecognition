import torch

x = torch.tensor([[[1, 1, 1], [1, 1, 2]], [[2, 1, 1], [2, 1, 2]], [[3, 1, 1], [3, 1, 2]]])
print(x)
print(x.shape)
x = x.permute(0, 2, 1)
print(x)
print(x.shape)