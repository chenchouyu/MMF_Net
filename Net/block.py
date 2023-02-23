# -*- coding: utf-8 -*-

from __future__ import print_function, division

import math

import numpy as np
import torch
import copy
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, dim=2, Ks=3, St=1, Pa=1, Drop=False):
        super(ConvBlock, self).__init__()
        self.layers = []
        for i in range(dim):
            self.layers.append(
                nn.Conv2d(ch_in if i == 0 else ch_out, ch_out, kernel_size=Ks, stride=St, padding=Pa, bias=True))
            self.layers.append(nn.BatchNorm2d(ch_out))
            self.layers.append(nn.ReLU())
            if bool(Drop):
                self.layers.append(nn.Dropout(Drop))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Recurrent(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x
        # for i in range(self.t):
        #     if i == 0:
        #         x = self.conv(x)
        #     x = self.conv(residual + x)
        x0 = self.conv(x)
        x1 = self.conv(residual + x0)
        x2 = self.conv(residual + x1)
        return x2


class RRCNN(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent(ch_out, t=t),
            Recurrent(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = self.Conv_1x1(x)
        x1 = self.RCNN(x0)
        return x0 + x1


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, scale_factor=2):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        xo = self.up(x)
        return xo


class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

        self.Up0 = nn.Sequential(
            ConvBlock(ch_in=128, ch_out=64, dim=1),
            ConvBlock(ch_in=64, ch_out=3, Ks=1, Pa=0, dim=1)
        )

        self.Up1 = nn.Sequential(
            UpConv(ch_in=256, ch_out=128, scale_factor=2),
            ConvBlock(ch_in=128, ch_out=64, Ks=1, Pa=0, dim=1),
            ConvBlock(ch_in=64, ch_out=3, Ks=1, Pa=0, dim=1)
        )

        self.Up2 = nn.Sequential(
            UpConv(ch_in=512, ch_out=256, scale_factor=4),
            ConvBlock(ch_in=256, ch_out=128, Ks=1, Pa=0, dim=1),
            ConvBlock(ch_in=128, ch_out=64, Ks=1, Pa=0, dim=1),
            ConvBlock(ch_in=64, ch_out=3, Ks=1, Pa=0, dim=1)
        )

        self.Up3 = nn.Sequential(
            UpConv(ch_in=1024, ch_out=512, scale_factor=8),
            ConvBlock(ch_in=512, ch_out=256, Ks=1, Pa=0, dim=1),
            ConvBlock(ch_in=256, ch_out=128, Ks=1, Pa=0, dim=1),
            ConvBlock(ch_in=128, ch_out=64, Ks=1, Pa=0, dim=1),
            ConvBlock(ch_in=64, ch_out=3, Ks=1, Pa=0, dim=1)
        )

        self.RConv = nn.Conv2d(4, 1, 1)
        self.GConv = nn.Conv2d(4, 1, 1)
        self.BConv = nn.Conv2d(4, 1, 1)

        self.VesConv = nn.Sequential(
            ConvBlock(ch_in=3, ch_out=2, Ks=1, Pa=0),
            nn.ReLU()
        )
        self.AvConv = nn.Sequential(
            ConvBlock(ch_in=3, ch_out=4, Ks=1, Pa=0),
            nn.ReLU()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, O0, O1, O2, O3):
        """
        :param O0: 128*512*512
        :param O1: 256*256*256
        :param O2: 512*128*128
        :param O3: 1024*64*64
        :return:
        """
        X0 = self.Up0(O0)  # 3*512*512
        X1 = self.Up1(O1)
        X2 = self.Up2(O2)
        X3 = self.Up3(O3)

        r0 = X0[:, 0, :, :].unsqueeze(1)
        r1 = X1[:, 0, :, :].unsqueeze(1)
        r2 = X2[:, 0, :, :].unsqueeze(1)
        r3 = X3[:, 0, :, :].unsqueeze(1)

        g0 = X0[:, 1, :, :].unsqueeze(1)
        g1 = X1[:, 1, :, :].unsqueeze(1)
        g2 = X2[:, 1, :, :].unsqueeze(1)
        g3 = X3[:, 1, :, :].unsqueeze(1)

        b0 = X0[:, 2, :, :].unsqueeze(1)
        b1 = X1[:, 2, :, :].unsqueeze(1)
        b2 = X2[:, 2, :, :].unsqueeze(1)
        b3 = X3[:, 2, :, :].unsqueeze(1)

        R = self.RConv(torch.cat([r0, r1, r2, r3], dim=1))
        G = self.GConv(torch.cat([g0, g1, g2, g3], dim=1))
        B = self.BConv(torch.cat([b0, b1, b2, b3], dim=1))

        Av = self.AvConv(torch.cat([R, G, B], dim=1))
        Vessel = self.VesConv(torch.cat([R, G, B], dim=1))

        return Vessel, Av


class Attention(nn.Module):
    def __init__(self, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = 16
        self.attention_head_size = int(1024 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(1024, self.all_head_size)
        self.key = nn.Linear(1024, self.all_head_size)
        self.value = nn.Linear(1024, self.all_head_size)

        self.out = nn.Linear(1024, 1024)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


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
    def __init__(self, vis=False):
        super(Block, self).__init__()
        self.hidden_size = 1024
        self.attention_norm = nn.LayerNorm(1024, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(1024, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention(vis)

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


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(2048, 1024, 1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 1024))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Transformer(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.Embedding = Embedding()
        self.layers = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(1024, eps=1e-6)
        for _ in range(size):
            layer = Block()
            self.layers.append(copy.deepcopy(layer))
        self.conv_more = ConvBlock(1024, 2048)

    def forward(self, x):
        x = self.Embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.encoder_norm(x)
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        return x
