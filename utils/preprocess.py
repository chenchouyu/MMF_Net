# -*- coding: utf-8 -*-
import copy
import os
import shutil
import warnings
import cv2
import random

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.util import random_noise
from skimage import img_as_ubyte

from torch.nn.functional import grid_sample
from torchvision.transforms import ToTensor, ToPILImage

warnings.simplefilter('ignore')


class Process(object):
    def __init__(self, config):
        self.dataPath = os.path.join(config.work_path, 'data')
        self.splitNum = config.split_number
        self.patchSize = config.patch_size
        self.resources = config.resources
        self.dataSets = [config.datasets] if type(config.datasets) != list else config.datasets

        self.MM = config.MM_config

        self.certainData = None
        self.certainDataSet = None
        self.certainName = None

        self.trainImgName = None
        self.valImgName = None
        self.testImgName = None

    def __adjust(self):
        if self.certainDataSet == 'HRF':
            size = (1536, 1024)

            for k in self.certainData.keys():
                self.certainData[k] = self.certainData[k].resize(size)
                if k == 'label' or k == 'vessel':
                    self.certainData[k] = Image.fromarray(np.uint8(np.round(np.array(self.certainData[k]) / 255) * 255))

    def getName(self):

        sign = self.certainDataSet

        self.trainImgName = os.listdir(os.path.join(self.resources[sign], 'training', 'images'))
        self.testImgName = os.listdir(os.path.join(self.resources[sign], 'test', 'images'))
        self.valImgName = self.testImgName

    def run_train(self):
        self.__del(['train', 'validation'])
        for self.certainDataSet in self.dataSets:

            self.getName()

            sign = self.certainDataSet

            with tqdm(total=len(self.trainImgName) + len(self.valImgName), desc=f'{sign}', ncols=60) as bar:

                for name in self.trainImgName:
                    image = Image.open(os.path.join(self.resources[sign], 'training', 'images', name))
                    label = Image.open(os.path.join(self.resources[sign], 'training', 'label', name))
                    vessel = Image.open(os.path.join(self.resources[sign], 'training', 'vessel', name)).convert('L')
                    mask = Image.open(os.path.join(self.resources[sign], 'training', 'mask', name)).convert('RGB')
                    MM = Image.open(os.path.join(f'/data2/chenchouyu/arteryVeinDatasets/MM/{sign}/{self.MM}/train', name)).convert('L')

                    self.certainName = name

                    image = np.array(image)
                    mask = np.array(mask)

                    image = image + mask - 255
                    image[image < 0] = 0
                    image = Image.fromarray(np.uint8(image))

                    self.certainData = {'image': image, 'label': label, 'vessel': vessel, 'MM': MM}
                    self.__adjust()
                    self.__produceTrainingImage()
                    bar.update(1)

                for idx, name in enumerate(self.valImgName):

                    image = Image.open(os.path.join(self.resources[sign], 'test', 'images', name))
                    label = Image.open(os.path.join(self.resources[sign], 'test', 'label', name))
                    vessel = Image.open(os.path.join(self.resources[sign], 'test', 'vessel', name)).convert('L')
                    mask = Image.open(os.path.join(self.resources[sign], 'test', 'mask', name)).convert('RGB')
                    MM = Image.open(os.path.join(f'/data2/chenchouyu/arteryVeinDatasets/MM/{sign}/{self.MM}/test', name)).convert('L')

                    self.certainName = sign + '_' + name

                    image = np.array(image)
                    mask = np.array(mask)

                    image = image + mask - 255
                    image[image < 0] = 0
                    image = Image.fromarray(np.uint8(image))

                    self.certainData = {'image': image, 'MM': MM}

                    self.__adjust()

                    self.certainData['label'] = label
                    self.certainData['vessel'] = vessel

                    for k in self.certainData.keys():
                        img = self.certainData[k]
                        img.save(os.path.join(self.dataPath, 'validation', k, self.certainName))

                    bar.update(1)

    def run_test(self):
        self.__del(['test'])
        for self.certainDataSet in self.dataSets:

            self.getName()

            sign = self.certainDataSet

            with tqdm(total=len(self.testImgName), desc=f'{sign}', ncols=60) as bar:

                for idx, name in enumerate(self.testImgName):

                    image = Image.open(os.path.join(self.resources[sign], 'test', 'images', name))
                    label = Image.open(os.path.join(self.resources[sign], 'test', 'label', name))
                    vessel = Image.open(os.path.join(self.resources[sign], 'test', 'vessel', name)).convert('L')
                    mask = Image.open(os.path.join(self.resources[sign], 'test', 'mask', name)).convert('RGB')
                    MM = Image.open(os.path.join(f'/data2/chenchouyu/arteryVeinDatasets/MM/{sign}/{self.MM}/test', name)).convert('L')

                    self.certainName = sign + '_' + name

                    image = np.array(image)
                    mask = np.array(mask)

                    image = image + mask - 255
                    image[image < 0] = 0
                    image = Image.fromarray(np.uint8(image))

                    self.certainData = {'image': image, 'MM': MM}

                    self.__adjust()

                    self.certainData['label'] = label
                    self.certainData['vessel'] = vessel

                    for k in self.certainData.keys():
                        img = self.certainData[k]
                        img.save(os.path.join(self.dataPath, 'test', k, self.certainName))

                    bar.update(1)

    def __del(self, folderName):

        for item in folderName:
            if os.path.exists(self.dataPath + item):
                shutil.rmtree(self.dataPath + item)

            _dir = ['image', 'label', 'vessel', 'MM']
            for name in _dir:
                os.makedirs(self.dataPath + '/' + item + '/' + name, exist_ok=True)

    def __produceTrainingImage(self):

        for index in range(self.splitNum):

            data = copy.deepcopy(self.certainData)
            w, h = data['image'].size

            data = warp(noise(rotate(data)))
            x, y = np.random.randint(0, w - self.patchSize), np.random.randint(0, h - self.patchSize)

            for k in data.keys():
                tmp = data[k].crop((x, y, x + self.patchSize, y + self.patchSize))
                if k == 'label':
                    tmp = np.round(np.array(tmp) / 255) * 255
                    tmp = Image.fromarray(np.uint8(tmp))
                tmp.save(self.dataPath + '/train/' + k + '/' + self.certainDataSet +
                         self.certainName.split('.')[0] + f'_{index}' + '.png')


def rotate(data):
    if random.random() < 0.6:
        return data

    res = dict()
    angle = random.choice(range(90))
    mid = [0 if random.random() < 0.5 else 1, 0 if random.random() < 0.5 else 1]

    for k in data.keys():
        tmp = cv2.cvtColor(np.asarray(data[k]), cv2.COLOR_RGB2BGR)
        if mid[0]:
            tmp = cv2.flip(tmp, 1)
        if mid[1]:
            tmp = cv2.flip(tmp, 0)

        tmp = Image.fromarray(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        tmp = tmp.rotate(angle)

        if k == 'label' or k == 'vessel':
            tmp = np.round(np.array(tmp) / 255) * 255
            tmp = Image.fromarray(np.uint8(tmp))

        res[k] = tmp
    return res


def noise(data):
    if random.random() < 0.6:
        return data

    res = dict()
    var = np.random.choice([0.002, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025])
    for k in data.keys():
        tmp = data[k]
        if k == 'image':
            tmp = random_noise(np.array(tmp), var=var)
            tmp = Image.fromarray(img_as_ubyte(tmp))
        res[k] = tmp
    return res


def warp(data):
    if random.random() < 0.6:
        return data

    res = dict()
    for k in data.keys():
        img = data[k]
        img = ToTensor()(img).unsqueeze(0)
        N, _, H, W = img.shape
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W)
        yy = yy.view(1, H, W)
        grid = torch.cat((xx, yy), 0).unsqueeze(0).float()
        grid = 2.0 * grid / (H - 1) - 1.0
        grid = grid.permute(0, 2, 3, 1)
        img = grid_sample(img, grid)
        img = ToPILImage()(img.squeeze())

        if k == 'label' or k == 'vessel':
            img = np.round(np.array(img) / 255) * 255
            img = Image.fromarray(np.uint8(img))

        res[k] = img
    return res
