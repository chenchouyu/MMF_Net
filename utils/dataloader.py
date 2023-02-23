# -*- coding: utf-8 -*-
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import cv2
from PIL import Image
from torchvision import transforms


class EyeBallData(Dataset):
    def __init__(self, config, mode):
        super(EyeBallData, self).__init__()
        self.fourthDimension = config.fourthdimension
        self.mode = mode
        self.path = config.path
        self.patchSize = config.patchSize
        self.nameList = os.listdir(os.path.join(self.path, 'image'))

        self.trans = transforms.ToTensor()

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.path, 'image', self.nameList[item]))
        lab = cv2.imread(os.path.join(self.path, 'label', self.nameList[item]))

        mas = cv2.imread(os.path.join(self.path, 'mask', self.nameList[item]), 0)
        ves = cv2.imread(os.path.join(self.path, 'vessel', self.nameList[item]), 0)

        img, mas, ves = self.trans(img), self.trans(mas), self.trans(ves)

        if self.mode == 'train':
            if self.fourthDimension:
                img = np.concatenate((img, mas), axis=0)

            return torch.Tensor(img), torch.tensor(decomposition(lab)), ves

        elif self.mode == 'validation':

            _, cur_h, cur_w = img.shape
            _, ori_h, ori_w = lab.shape
            curSize, oriSize = (cur_h, cur_w), (ori_h, ori_w)
            imgList = self.splitForValidation(img, mas)
            return imgList, torch.tensor(decomposition(lab)), self.trans(ves), curSize, oriSize, self.nameList[item]

        else:
            raise 'not submitted'

    def __len__(self):
        return len(self.nameList)

    def splitForValidation(self, img, mas):

        _, h, w = img.shape
        s = self.patchSize
        imgList = []

        m, n = w // s, h // s

        for j in range(int(m)):
            for i in range(int(n)):
                image = img[:, i * s: (i + 1) * s, j * s: (j + 1) * s]
                if self.fourthDimension:
                    mask = mas[:, i * s: (i + 1) * s, j * s: (j + 1) * s]
                    image = np.concatenate((image, mask), axis=0)
                imgList.append(torch.Tensor(image))

        return imgList

    def splitForTest(self):
        pass


def decomposition(label_av):
    label = np.zeros_like(label_av[..., 0])

    label[label_av[:, :, 0] == 255] = 2
    label[label_av[:, :, 2] == 255] = 3
    label[label_av[:, :, 1] == 255] = 1

    return label
