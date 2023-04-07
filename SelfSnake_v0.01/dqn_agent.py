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

        # Calculate the output dimensions of the second convolutional layer
        def conv2d_output_size(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        conv1_output_width = conv2d_output_size(width // size, 3, 1)
        conv1_output_height = conv2d_output_size(height // size, 3, 1)
        conv2_output_width = conv2d_output_size(conv1_output_width, 3, 1)
        conv2_output_height = conv2d_output_size(conv1_output_height, 3, 1)

        self.head = nn.Linear(64 * conv2_output_width * conv2_output_height, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.head(x)
