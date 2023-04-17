# -*- coding: utf-8 -*-
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

import cv2
from torchvision import transforms


class EyeBallData(Dataset):
    def __init__(self, config, mode):
        super(EyeBallData, self).__init__()
        self.mode = mode

        self.path = os.path.join(config.work_path, 'data', mode)

        self.patchSize = config.patch_size
        self.nameList = os.listdir(os.path.join(self.path, 'image'))

        self.trans = transforms.ToTensor()

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.path, 'image', self.nameList[item]))
        lab = cv2.imread(os.path.join(self.path, 'label', self.nameList[item]))

        mas = cv2.imread(os.path.join(self.path, 'MM', self.nameList[item]), 0)
        ves = cv2.imread(os.path.join(self.path, 'vessel', self.nameList[item]), 0)

        img, mas = self.trans(img), self.trans(mas)

        _, cur_h, cur_w = img.shape
        ori_h, ori_w, _ = lab.shape

        curSize, oriSize = (cur_w, cur_h), (ori_w, ori_h)

        if self.mode == 'train':
            img = np.concatenate((img, mas), axis=0)

            return torch.Tensor(img), torch.tensor(composition(lab)), self.trans(ves)

        elif self.mode == 'validation':
            imgList = self.splitForValidation(img, mas)
            return imgList, torch.tensor(composition(lab)), self.trans(ves), curSize, oriSize, self.nameList[item]

        else:
            point = [0, 64, 128, 196, 256]
            # point = [0, 45, 90, 135, 180]
            imgDict = self.splitForTest(img, mas, point)

            return imgDict, oriSize, curSize, self.nameList[item]

    def __len__(self):
        return len(self.nameList)

    def splitForValidation(self, img, mas):

        _, h, w = img.shape
        s = self.patchSize
        imgList = []

        m = w // s + (0 if w % s == 0 else 1)
        n = h // s + (0 if h % s == 0 else 1)

        tmp_w, tmp_h = (m * s - w) // 2, (n * s - h) // 2

        cropItemImg = torch.zeros((3, n * s, m * s))
        cropItemImg[:, tmp_h: tmp_h + h, tmp_w: tmp_w + w] = img

        cropItemMas = torch.zeros((1, n * s, m * s))
        cropItemMas[:, tmp_h: tmp_h + h, tmp_w: tmp_w + w] = mas

        for j in range(int(m)):
            for i in range(int(n)):
                image = cropItemImg[:, i * s: (i + 1) * s, j * s: (j + 1) * s]
                mask = cropItemMas[:, i * s: (i + 1) * s, j * s: (j + 1) * s]
                image = torch.cat([image, mask], dim=0)
                imgList.append(image)

        return imgList

    def splitForTest(self, item, mas, point):

        _, h, w = item.shape
        s = self.patchSize

        m = w // s + 1
        n = h // s + 1

        pos = [(i, j) for i in point for j in point]

        imgDict = dict()

        for idx, p in enumerate(pos):
            newImage = torch.zeros((3, n * s, m * s))
            newMask = torch.zeros((1, n * s, m * s))

            newImage[:, p[0]: p[0] + h, p[1]: p[1] + w] = item
            newMask[:, p[0]: p[0] + h, p[1]: p[1] + w] = mas

            imgDict[p] = [newImage, newMask]

        resDict = defaultdict(list)

        for k in imgDict.keys():
            for j in range(int(m)):
                for i in range(int(n)):
                    image = imgDict[k][0][:, i * s: (i + 1) * s, j * s: (j + 1) * s]
                    mask = imgDict[k][1][:, i * s: (i + 1) * s, j * s: (j + 1) * s]

                    resDict[k].append(torch.cat([image, mask], dim=0))

        return resDict


def composition(label_av):
    label = np.zeros_like(label_av[..., 0])

    label[label_av[:, :, 0] == 255] = 2
    label[label_av[:, :, 2] == 255] = 3
    label[label_av[:, :, 1] == 255] = 1

    return label
