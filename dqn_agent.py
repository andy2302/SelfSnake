import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions: int, device, width: int, height: int, size: int):
        super(DQN, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.head = nn.Linear(64 * (width // size - 2) * (height // size - 2), num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.head(x)
