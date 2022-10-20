import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)
        self.liner1 = nn.Linear(64, 128)
        self.liner2 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(x, 0.5)
        x = self.max_pool1(x)
        x = torch.flatten(x)
        x = F.relu(self.liner1(x))
        x = F.softmax(self.liner2(x))
        return x
