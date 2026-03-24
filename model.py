"""
Neural network for Gomoku (policy + value).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config

class GomokuNet(nn.Module):
    def __init__(self):
        super().__init__()
        # U-Net style: encoder-decoder with skip connections
        self.enc1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        # decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(16, 4, kernel_size=3, padding=1)  # 8 (from dec1) + 8 (skip) = 16
        self.final_up = nn.Upsample(size=(config.BOARD_SIZE, config.BOARD_SIZE), mode='bilinear', align_corners=False)

        # output heads
        self.policy_conv = nn.Conv2d(4, 1, kernel_size=1)

        # value head
        self.value_fc1 = nn.Linear(16, 8)
        self.value_fc2 = nn.Linear(8, 1)

    def forward(self, x):
        # encoder
        e1 = F.leaky_relu(self.enc1(x))
        e1 = F.leaky_relu(self.enc2(e1))
        p = self.pool(e1)

        # bottleneck
        b = F.leaky_relu(self.bottleneck(p))

        # value head (from bottleneck)
        b_global = F.adaptive_avg_pool2d(b, (1, 1)).view(-1, 16)
        v = F.leaky_relu(self.value_fc1(b_global))
        v = torch.tanh(self.value_fc2(v))

        # decoder + skip connection
        u = self.up(b)
        u = F.leaky_relu(self.dec1(u))
        u = torch.cat([u, e1[:, :, :14, :14]], dim=1)  # skip connection, crop e1 to 14x14 to match u
        u = F.leaky_relu(self.dec2(u))
        u = self.final_up(u)  # upsample to BOARD_SIZExBOARD_SIZE

        # policy head
        p_out = self.policy_conv(u)
        p_out = p_out.view(-1, config.BOARD_SIZE * config.BOARD_SIZE)
        p_out = F.softmax(p_out, dim=1)

        return p_out, v