# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn

from resnet.model.resnet import res2net50
from block import ConvBlock, RRCNN, UpConv, MF, Transformer


class Decode(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor=2):
        super(Decode, self).__init__()
        self.Up = UpConv(ch_in, ch_out, scale_factor)
        self.RCnn = RRCNN(ch_out * 2, ch_out)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X, P):
        Xi = self.Up(X)
        Xo = self.RCnn(torch.cat((P, Xi), dim=1))
        return Xo


class MMF(nn.Module):

    def __init__(self, in_ch=4, is_pretrained=True):
        super(MMF, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2 * 2, n1 * 4 * 2, n1 * 8 * 2, n1 * 16 * 2]

        self.Conv1 = ConvBlock(in_ch, filters[0])
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = ConvBlock(filters[0], int(filters[1]/2), Drop=0.1)

        res = res2net50(pretrained=is_pretrained)
        self.encoder1 = res.layer1
        self.encoder2 = res.layer2
        self.encoder3 = res.layer3
        self.encoder4 = res.layer4

        self.Middle = Transformer(size=3)

        self.decoder4 = Decode(filters[4], filters[3])
        self.decoder3 = Decode(filters[3], filters[2])
        self.decoder2 = Decode(filters[2], filters[1])
        self.decoder1 = Decode(filters[1], int(filters[1] / 2))
        self.MultiConv = MF()

    def forward(self, X):
        X0 = self.Conv1(X)  # 3*512*512
        M = self.Conv2(X0)
        B0 = self.MaxPool(X0)  # 64*256*256

        B1 = self.encoder1(B0)  # 256*256*256
        B2 = self.encoder2(B1)  # 512*128*128
        B3 = self.encoder3(B2)  # 1024*64*64
        X1 = self.encoder4(B3)  # 2048*32*32

        X2 = self.Middle(X1)  # 2048*32*32

        O3 = self.decoder4(X2, B3)  # 1024*64*64
        O2 = self.decoder3(O3, B2)  # 512*128*128
        O1 = self.decoder2(O2, B1)  # 256*256*256
        O0 = self.decoder1(O1, M)  # 128*512*512

        Vessel, Av = self.MultiConv(O0, O1, O2, O3)

        return Vessel, Av

