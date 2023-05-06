import torch
from torch.utils.data import TensorDataset
import torch.nn as nn

x_data = torch.randn(480, 535, 535, 1)
a = nn.Linear(1, 1)
print(a(x_data).shape)