from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers"""

    def __init__(self, channels: int = 256) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """AlphaZero-style neural network for chess"""

    def __init__(self, num_res_blocks: int = 20, num_channels: int = 256, input_planes: int = 106) -> None:
        super(AlphaZeroNet, self).__init__()

        # Initial convolutional block
        self.conv_input = nn.Conv2d(input_planes, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4864)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial conv block
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


def get_model(device: torch.device) -> AlphaZeroNet:
    model = AlphaZeroNet()
    model.to(device)
    model.eval()
    return model
