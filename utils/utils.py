# -*- coding: utf-8 -*-
import math
import os
import shutil
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import cv2
from PIL import Image

from skimage import io
from skimage.util import random_noise
from skimage import img_as_ubyte

from dataloader import composition

warnings.simplefilter('ignore')


def adjust(item, sign):
    if sign == 'Drive':
        size = (512, 512)
    elif sign == 'HRF':
        size = (1536, 1024)
    else:
        size = (1024, 1024)

    for k in item.keys():
        item[k] = item[k].resize(size, Image.ANTIALIAS)
        if k == 'label' or k == 'vessel':
            item[k] = Image.fromarray(np.uint8(np.round(np.array(item[k]) / 255) * 255))

    return item


def preprocess(config):
    # prepare
    dataSavePath = os.path.join(config.work_path, 'data')

    for item in ['train', 'test', 'validation']:
        if os.path.exists(os.path.join(dataSavePath, item)):
            shutil.rmtree(os.path.join(dataSavePath, item))

        _dir = ['image', 'label', 'vessel', 'mask']
        for name in _dir:
            os.makedirs(os.path.join(dataSavePath, item, name), exist_ok=True)

    resources = config.resources
    if type(config.dataset) != list:
        config.dataset = [config.dataset]

    for k in config.dataset:
        trainImgName = os.listdir(os.path.join(resources[k], 'training', 'images'))
        testImgName = os.listdir(os.path.join(resources[k], 'test', 'images'))
        valImageName = testImgName
        with tqdm(total=len(trainImgName) + len(testImgName) + len(valImageName), desc=f'{k}') as bar:
            sigma = 0.05
            patch_size = config.patch_size
            num = config.split_number

            for name in trainImgName:
                image = Image.open(os.path.join(resources[k], 'training', 'images', name))
                label = Image.open(os.path.join(resources[k], 'training', 'label', name))
                vessel = Image.open(os.path.join(resources[k], 'training', 'vessel', name)).convert('L')
                mask = Image.open(os.path.join(resources[k], 'training', 'MM', name)).convert('L')

                data = {'image': image, 'label': label, 'vessel': vessel, 'mask': mask}

                data = adjust(data, k)

                w, h = data['image'].size

                for index in range(num):
                    # random patch

                    x, y = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)

                    for sign in data.keys():
                        tmp = data[sign].crop((x, y, x + patch_size, y + patch_size))
                        if sign == 'label':
                            tmp = np.round(np.array(tmp) / 255) * 255
                            tmp = Image.fromarray(np.uint8(tmp))
                        tmp.save('./data/train/' + sign + '/' + k + name.split('.')[0] + f'_{index}_' + '_orig.png')

                    # Noise
                    x, y = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)

                    for sign in data.keys():
                        tmp = data[sign].crop((x, y, x + patch_size, y + patch_size))
                        if sign == 'label':
                            tmp = np.round(np.array(tmp) / 255) * 255
                            tmp = Image.fromarray(np.uint8(tmp))
                        tmp.save('./data/train/' + sign + '/' + k + name.split('.')[0] + f'_{index}_' + '_noise.png')

                    img = io.imread('./data/train/image/' + k + name.split('.')[0] + f'_{index}_' + "_noise.png")
                    noisy_img = random_noise(img, var=sigma ** 2)
                    io.imsave('./data/train/image/' + k + name.split('.')[0] + f'_{index}_' + "_noise.png",
                              img_as_ubyte(noisy_img))

                    # rotation
                    x, y = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)

                    angle = random.choice(range(90))
                    mid = [0 if random.random() < 0.5 else 1, 0 if random.random() < 0.5 else 1]

                    for sign in data.keys():
                        img = cv2.cvtColor(np.asarray(data[sign]), cv2.COLOR_RGB2BGR)
                        if mid[0]:  # horizontal
                            img = cv2.flip(img, 1)
                        if mid[1]:  # vertical
                            img = cv2.flip(img, 0)

                        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        img = img.rotate(angle)

                        tmp = img.crop((x, y, x + patch_size, y + patch_size))
                        if sign == 'label':
                            tmp = np.round(np.array(tmp) / 255) * 255
                            tmp = Image.fromarray(np.uint8(tmp))
                        tmp.save('./data/train/' + sign + '/' + k + name.split('.')[0] + f'_{index}_' + '_rot.png')

                bar.update(1)

            for idx, name in enumerate(testImgName + valImageName):
                image = Image.open(os.path.join(resources[k], 'test', 'images', name))
                label = Image.open(os.path.join(resources[k], 'test', 'label', name))
                vessel = Image.open(os.path.join(resources[k], 'test', 'vessel', name)).convert('L')
                mask = Image.open(os.path.join(resources[k], 'test', 'MM', name)).convert('L')

                data = {'image': image, 'mask': mask}

                data = adjust(data, k)

                data['label'] = label
                data['vessel'] = vessel

                if idx >= len(testImgName):
                    folderName = 'validation'
                else:
                    folderName = 'test'

                for sign in data.keys():
                    tmp = data[sign]
                    tmp.save(f'./data/{folderName}/' + sign + '/' + k + '_' + name)

                bar.update(1)


class Bce2d(nn.Module):
    def __init__(self, reduction='mean'):
        super(Bce2d, self).__init__()
        self.reduction = reduction

    def forward(self, pred, gt):
        pos = torch.eq(gt, 1).float()
        neg = torch.eq(gt, 0).float()
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg
        alpha_pos = num_neg / num_total
        alpha_neg = num_pos / num_total
        weights = alpha_pos * pos + alpha_neg * neg
        return nn.BCEWithLogitsLoss(weights, reduction=self.reduction)(pred, gt)

