# -*- coding: utf-8 -*-
import torch
from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, cuda_device, weight):
        super(MultiLoss, self).__init__()
        self.vesselCriterion = nn.CrossEntropyLoss().to(cuda_device)

        avWeight = torch.FloatTensor([1, 1, 3, 5]).to(cuda_device)
        self.avCriterion = nn.CrossEntropyLoss(weight=avWeight).to(cuda_device)

        self.vesselWeight, self.avWeight = weight

    def forward(self, vesselResult, avResult, vesselLabel, avLabel):
        vesselLoss = self.vesselCriterion(vesselResult, vesselLabel)
        avLoss = self.avCriterion(avResult, avLabel)

        return self.vesselWeight * vesselLoss + self.avWeight * avLoss
