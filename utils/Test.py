# -*- coding: utf-8 -*-
import math

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from Net.MMF import MMF
from utils.dataloader import decomposition
from utils.utils import rgb_to_bgr

import numpy as np
from tqdm import tqdm
import os
import cv2


def test():
    Model = MMF(is_pretrained=False)
    Model = Model.to('cuda')
    # Model = nn.DataParallel(Model, device_ids=[0, 1])
    parm = torch.load('./model/Drive/BestModel0.8976.pkl')
    Model.load_state_dict(parm, False)
    Model.eval()

    testLoad = EyeBallData()
    testLoader = DataLoader(testLoad)

    os.makedirs(os.path.join('./TestResult', 'ves'), exist_ok=True)
    os.makedirs(os.path.join('./TestResult', 'av'), exist_ok=True)

    with tqdm(total=len(testLoader), desc='Process') as bar:
        for item, data in enumerate(testLoader):
            resDict, _, _, w, h, name = data
            # print(name[0])
            res = {}
            for k in resDict.keys():
                outListAv = []
                outListVes = []
                for img in resDict[k]:
                    img = img.to('cuda')
                    predVes, predAv = Model(img)

                    predAv = torch.softmax(predAv, 1)
                    _, predAv = torch.max(predAv.cpu().data, 1)

                    predVes = torch.softmax(predVes, 1)
                    _, predVes = torch.max(predVes.cpu().data, 1)

                    predAv = restore_av(predAv[0])
                    predAv = Image.fromarray(np.uint8(predAv), mode='RGB')

                    predVes = Image.fromarray(np.uint8(predVes[0]))

                    outListAv.append(predAv)
                    outListVes.append(predVes)
                res[k] = [outListAv, outListVes]

            outAv, outVessel = threshold_vessel(res, name)
            outAv = Image.fromarray(np.uint8(restore_av(outAv)))
            outVessel = Image.fromarray(np.uint8(outVessel))

            saveAv = np.uint8(np.round(np.array(outAv.resize((w, h)))))
            saveAv = rgb_to_bgr(saveAv)

            # outVes = threshold_ves(outListVes, idx, pos, w, h, patch_size)
            outVes = np.uint8(np.round(np.array(outVessel.resize((w, h))))*255)
            saveVes = outVes

            cv2.imwrite(os.path.join('./TestResult', 'ves', name[0]), saveVes)
            cv2.imwrite(os.path.join('./TestResult', 'av', name[0]), saveAv)
            bar.update(1)


class EyeBallData(Dataset):
    def __init__(self):
        super(EyeBallData, self).__init__()
        self.fourthDimension = True
        self.path = './data/test'
        self.nameList = os.listdir(os.path.join(self.path, 'image'))

        self.trans = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.path, 'image', self.nameList[item]))
        lab = Image.open(os.path.join(self.path, 'label', self.nameList[item]))

        mas = Image.open(os.path.join(self.path, 'mask', self.nameList[item])).convert('L')
        ves = cv2.imread(os.path.join(self.path, 'vessel', self.nameList[item]), 0)

        w, h = lab.size
        resDict = self.split(img, mas)

        return resDict, torch.tensor(decomposition(lab)), self.trans(ves), w, h, self.nameList[item]

    def __len__(self):
        return len(self.nameList)

    def split(self, img, mas):
        w, h = img.size
        s = 256
        pos = [(i, j) for i in [0, 32, 64, 96, 128, 160, 192, 224] for j in [0, 32, 64, 96, 128, 160, 192, 224]]
        imgDict = {}
        for idx, p in enumerate(pos):
            newImage = Image.new('RGB', (w + 256, h + 256))
            newMask = Image.new('L', (w + 256, h + 256))
            newImage.paste(img, p)
            newMask.paste(mas, p)
            imgDict[p] = [newImage, newMask]
        resDict = {}
        w, h = w + 256, h + 256
        m, n = math.ceil(w / s), math.ceil(h / s)
        for k in imgDict.keys():
            resDict[k] = []
            for i in range(int(m)):
                for j in range(int(n)):
                    image = imgDict[k][0].crop((s * i, s * j, s * i + s, s * j + s))
                    if self.fourthDimension:
                        mask = imgDict[k][1].crop((s * i, s * j, s * i + s, s * j + s))
                        image, mask = np.array(image), np.array(mask)
                        # print(image.shape, mask.shape)
                        mask = mask[:, :, np.newaxis]
                        image = np.concatenate((image, mask), axis=2)
                    image = self.trans(image)
                    resDict[k].append(image)

        return resDict


def threshold_vessel(res, name):
    if 'Drive' in name[0]:
        size = (512 + 256, 512 + 256)
    elif 'HRF' in name[0]:
        size = (1536, 1024)
    else:
        raise EOFError

    newAvImg = Image.new('RGB', size)
    newVesselImg = Image.new('L', size)

    s = 256
    w, h = newAvImg.size
    m, n = math.ceil(w / s), math.ceil(h / s)
    outAv, outVessel = [], []
    for k in res.keys():
        for i, out in enumerate(res[k][0]):
            # print(out[0].size, out[1].size)
            newAvImg.paste(out, (int(i / n) * s, int(i % n) * s))
        for i, out in enumerate(res[k][1]):
            newVesselImg.paste(out, (int(i / n) * s, int(i % n) * s))

        outAv.append(decomposition(newAvImg.crop((k[0], k[1], k[0] + 512, k[1] + 512))))
        outVessel.append(np.round(newVesselImg.crop((k[0], k[1], k[0] + 512, k[1] + 512))))

    av = np.zeros_like(outAv[0])
    for i in range(len(outAv[0])):
        for j in range(len(outAv[0][0])):
            pointSum = [0, 0, 0, 0]
            for tmp in outAv:
                pointSum[tmp[i][j]] += 1
            av[i][j] = pointSum.index(max(pointSum))

    vessel = np.zeros_like(outAv[0])
    for i in range(len(outVessel[0])):
        for j in range(len(outVessel[0][0])):
            pointSum = [0, 0]
            for tmp in outVessel:
                pointSum[tmp[i][j]] += 1
            vessel[i][j] = pointSum.index(max(pointSum))

    return av, vessel


def restore_av(data):
    r = np.zeros_like(data)
    g = np.zeros_like(data)
    b = np.zeros_like(data)

    r[data == 3] = 255
    g[data == 1] = 255
    b[data == 2] = 255

    r, g, b = np.expand_dims(r, 2), np.expand_dims(g, 2), np.expand_dims(b, 2)

    return np.concatenate((r, g, b), axis=2)


if __name__ == '__main__':
    test()
