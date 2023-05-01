# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 02:26:00 2022

@author: Shahir

Based on https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

import abc
from typing import Optional

import torch
import torch.nn as nn


class BaseBlock(nn.Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def EXPANSION(
            self) -> int:

        raise NotImplementedError

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            batchnorm: bool = False) -> None:

        if stride != 1:
            assert downsample is not None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        self.batchnorm = batchnorm

    @abc.abstractmethod
    def _make_block(
            self):

        pass


class BasicBlock(BaseBlock):
    EXPANSION = 1

    def _make_block(
            self):

        self.activation = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=(not self.batchnorm))
        self.bn1 = nn.BatchNorm2d(self.out_channels) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=(not self.batchnorm))
        self.bn2 = nn.BatchNorm2d(self.out_channels) if self.batchnorm else nn.Identity()

    def forward(
            self,
            x: torch.Tensor) -> torch.Tensor:

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
            self) -> None:

        super().__init__()

    def _initialize_weights(
            self) -> None:

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
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            stride: int = 1,
            block: BaseBlock = BasicBlock,
            batchnorm: bool = True) -> nn.Sequential:

        downsample = None
        if stride != 1 or in_channels != out_channels * block.EXPANSION:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * block.EXPANSION,
                    kernel_size=1,
                    stride=stride,
                    bias=(not batchnorm)),
                nn.BatchNorm2d(out_channels * block.EXPANSION) if batchnorm else nn.Identity())

        layers = []
        layers.append(block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            downsample=downsample,
            batchnorm=batchnorm))
        in_channels = out_channels * block.EXPANSION
        for _ in range(1, num_blocks):
            layers.append(block(
                in_channels=in_channels,
                out_channels=out_channels,
                batchnorm=batchnorm))

        return nn.Sequential(*layers)
