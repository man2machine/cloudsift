# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 02:26:00 2022

@author: Shahir
"""

import torch.nn as nn


def conv_layer(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        bias=False):

    layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        bias=bias)

    return layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            activation=nn.LeakyReLU,
            downsample=None,
            batchnorm=False):

        super().__init__()

        self.conv1 = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=not batchnorm)
        self.bn1 = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()
        self.activation = activation(inplace=True)
        self.conv2 = conv_layer(out_channels, out_channels, bias=not batchnorm)
        self.bn2 = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()
        self.downsample = downsample
        self.stride = stride
        self.batchnorm = batchnorm

    def forward(
            self,
            x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        return out


class ResNetBuilder(nn.Module):
    def __init__(
            self):

        super().__init__()

    def _initialize_weights(
            self,
            initialize_linear=False):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(
            self,
            in_channels,
            out_channels,
            num_blocks,
            stride=1,
            block=BasicBlock,
            batchnorm=False):

        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=not batchnorm),
                nn.BatchNorm2d(
                    out_channels * block.expansion) if batchnorm else nn.Identity(),
            )

        layers = []
        layers.append(block(
            in_channels,
            out_channels,
            stride=stride,
            downsample=downsample))
        in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(in_channels, out_channels))

        return nn.Sequential(*layers)
