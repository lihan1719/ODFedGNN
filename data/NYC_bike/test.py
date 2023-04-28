import torch
from torch.utils.data import TensorDataset

x_data = torch.randn(480, 2, 1, 1)
y_data = torch.randn(480, 1, 1)
print(x_data)
dataset = TensorDataset(x_data, y_data)
print(dataset)