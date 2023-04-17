# -*- coding: utf-8 -*-
from __future__ import print_function, division

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class DoubleConv(nn.Module):
    def __init__(self, inputFeature, outputFeature):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inputFeature, outputFeature, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputFeature),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputFeature, outputFeature, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputFeature),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class MultiFeatureDetectionBlock(nn.Module):
    def __init__(self, numFeature, numSplit=4):
        super(MultiFeatureDetectionBlock, self).__init__()
        self.Conv1 = nn.Conv2d(numFeature, numFeature, kernel_size=1)
        self.numSplit = numSplit
        self.Conv_group = nn.ModuleList()
        for _ in range(numSplit - 1):
            self.Conv_group.append(nn.Conv2d(numFeature // numSplit, numFeature // numSplit, kernel_size=3, padding=1))
        self.Conv2 = nn.Conv2d(numFeature, numFeature, kernel_size=1)
        self.bn = nn.BatchNorm2d(numFeature)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.Conv1(x)
        splitFeatures = torch.chunk(x, self.numSplit, dim=1)
        out = [splitFeatures[0]]
        for i, (feature, layer) in enumerate(zip(splitFeatures[1:], self.Conv_group)):
            if i == 0:
                out.append(layer(feature))
            else:
                out.append(layer(feature + out[-1]))
        x = self.Conv2(torch.cat(out, dim=1))

        return self.relu(self.bn(x + residual))


class Recurrent(nn.Module):
    def __init__(self, ch_out):
        super(Recurrent, self).__init__()
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        x0 = self.conv(x)
        x1 = self.conv(residual + x0)
        x2 = self.conv(residual + x1)
        return x2


class Decoder(nn.Module):
    def __init__(self, inputFeature, outputFeature):
        super(Decoder, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inputFeature, outputFeature, kernel_size=3, padding=1)
        )
        self.layer = nn.Sequential(
            nn.Conv2d(2 * outputFeature, outputFeature, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputFeature),
            nn.ReLU(inplace=True),
            Recurrent(outputFeature)
        )

    def forward(self, x, y):
        x = self.up(x)
        return self.layer(torch.cat([x, y], dim=1))


class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

        self.Up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DoubleConv(inputFeature=128, outputFeature=32)
        )

        self.Up2 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            DoubleConv(inputFeature=256, outputFeature=16)
        )

        self.Up3 = nn.Sequential(
            nn.Upsample(scale_factor=8),
            DoubleConv(inputFeature=512, outputFeature=8)
        )

        self.Conv1 = DoubleConv(30, 1)
        self.Conv2 = DoubleConv(30, 1)
        self.Conv3 = DoubleConv(30, 1)
        self.Conv4 = DoubleConv(30, 1)

        self.VesConv = nn.Sequential(
            nn.Conv2d(4, 1, 1)
        )
        self.AvConv = nn.Sequential(
            nn.Conv2d(4, 4, 1)
        )

    def forward(self, O0, O1, O2, O3):
        X0 = O0
        X1 = self.Up1(O1)
        X2 = self.Up2(O2)
        X3 = self.Up3(O3)

        layer0_0, layer1_0, layer2_0, layer3_0 = torch.chunk(X0, 4, dim=1)
        layer0_1, layer1_1, layer2_1, layer3_1 = torch.chunk(X1, 4, dim=1)
        layer0_2, layer1_2, layer2_2, layer3_2 = torch.chunk(X2, 4, dim=1)
        layer0_3, layer1_3, layer2_3, layer3_3 = torch.chunk(X3, 4, dim=1)

        layer0 = self.Conv1(torch.cat([layer0_0, layer0_1, layer0_2, layer0_3], dim=1))
        layer1 = self.Conv2(torch.cat([layer1_0, layer1_1, layer1_2, layer1_3], dim=1))
        layer2 = self.Conv3(torch.cat([layer2_0, layer2_1, layer2_2, layer2_3], dim=1))
        layer3 = self.Conv4(torch.cat([layer3_0, layer3_1, layer3_2, layer3_3], dim=1))

        Av = self.AvConv(torch.cat([layer0, layer1, layer2, layer3], dim=1))
        Vessel = self.VesConv(Av)

        return Vessel, Av


class Net(nn.Module):
    def __init__(self, in_ch=4):
        super(Net, self).__init__()
        n = 64
        filters = [n, n * 2, n * 4, n * 8, n * 16]

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_ch, filters[0], kernel_size=3, padding=1),
            MultiFeatureDetectionBlock(filters[0]),
            MultiFeatureDetectionBlock(filters[0])
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
            MultiFeatureDetectionBlock(filters[1]),
            MultiFeatureDetectionBlock(filters[1])
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            MultiFeatureDetectionBlock(filters[2]),
            MultiFeatureDetectionBlock(filters[2])
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1),
            MultiFeatureDetectionBlock(filters[3]),
            MultiFeatureDetectionBlock(filters[3])
        )

        # self.Middle = DoubleConv(filters[3], filters[4])
        self.Middle = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1),
            Transformer(filters[4], size=3)
        )

        self.decoder4 = Decoder(filters[4], filters[3])
        self.decoder3 = Decoder(filters[3], filters[2])
        self.decoder2 = Decoder(filters[2], filters[1])
        self.decoder1 = Decoder(filters[1], filters[0])

        self.featureFusion = MF()

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.apply(_weights_init)

    def forward(self, x):
        c1 = self.encoder1(x)

        c2 = self.encoder2(self.maxPool(c1))
        c3 = self.encoder3(self.maxPool(c2))
        c4 = self.encoder4(self.maxPool(c3))

        y = self.Middle(self.maxPool(c4))

        f4 = self.decoder4(y, c4)
        f3 = self.decoder3(f4, c3)
        f2 = self.decoder2(f3, c2)
        f1 = self.decoder1(f2, c1)

        Vessel, Av = self.featureFusion(f1, f2, f3, f4)

        Av = self.softmax(Av)
        Vessel = self.sigmoid(Vessel)

        return Vessel, Av


class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, 'dim should be divisible by head_dim'
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 1024))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.hidden_size = 1024
        self.attention_norm = nn.LayerNorm(1024, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(1024, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(dim)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Transformer(nn.Module):
    def __init__(self, dim, size=3):
        super().__init__()
        self.Embedding = Embedding()
        self.layers = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(1024, eps=1e-6)
        for _ in range(size):
            layer = Block(dim)
            self.layers.append(copy.deepcopy(layer))

    def forward(self, x):
        x = self.Embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.encoder_norm(x)
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)

        return x
