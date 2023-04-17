# -*- coding: utf-8 -*-
import math

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from Net.MMF import MMF
from utils.dataloader import EyeBallData

import numpy as np
from tqdm import tqdm
import os
import cv2


def test(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    Model = MMF(is_pretrained=False).eval()
    Model = nn.DataParallel(Model)
    parm = torch.load('/data2/chenchouyu/MMF_Net/model_NoTransformer_big/HRF/39_vesIou_0.622_vesAcc_0.968_avAcc_0'
                      '.944_vesSpe_0.974_avSpe_0.956_vesSen_0.878_vesPre_0.684_vesF1_0.765_avF1_0.939_avIou_0'
                      '.888_avSen_0.932_avPre_0.948.pth')
    Model.load_state_dict(parm, False)

    Model = Model.to(config.cuda_device)

    testLoad = EyeBallData(config, 'test')
    testLoader = DataLoader(testLoad, shuffle=False)

    os.makedirs(os.path.join('./TestResult', 'ves'), exist_ok=True)
    os.makedirs(os.path.join('./TestResult', 'av'), exist_ok=True)

    with tqdm(total=len(testLoader), desc='Test', ncols=80) as bar:
        for item, (data, originalSize, currentSize, name) in enumerate(testLoader):
            # resDict, _, _, w, h, name = data
            originalSize = tuple(map(int, originalSize))
            currentSize = tuple(map(int, currentSize))
            res = {}
            for k in data.keys():
                outListAv = []
                outListVes = []
                for img in data[k]:

                    img = img.to(config.cuda_device)
                    predVes, predAv = Model(img)

                    _, predAv = torch.max(predAv.cpu().data, 1)
                    predAv = predAv[0]

                    # _, predVes = torch.max(predVes.cpu().data, 1)
                    # predVes = predVes[0]
                    predVes = predVes[0][0].cpu().detach().numpy()

                    outListAv.append(predAv)
                    outListVes.append(predVes)

                res[k] = [outListAv, outListVes]

            outAv, outVessel = threshold_vessel(res, currentSize)

            saveAv = Image.fromarray(np.uint8(np.round(cv2.resize(restore_av(outAv), originalSize)))).convert('RGB')
            # saveVes = Image.fromarray(np.uint8(np.round(cv2.resize(outVessel * 255, originalSize)))).convert('L')
            saveVes = Image.fromarray(np.uint8(cv2.resize(np.round(outVessel) * 255, originalSize))).convert('L')

            saveAv.save(os.path.join(f'./TestResult', 'av', name[0]))
            saveVes.save(os.path.join(f'./TestResult', 'ves', name[0]))
            bar.update(1)


def threshold_vessel(res, currentSize):
    w, h = currentSize

    s = 256

    m = w // s + 1
    n = h // s + 1

    size = (n * s, m * s)

    outAv, outVessel = [], []
    for k in res.keys():

        newAvImg = np.zeros(size)
        newVesselImg = np.zeros(size)

        for i, out in enumerate(res[k][0]):
            newAvImg[int(i % n) * s: int(i % n) * s + s, int(i / n) * s: int(i / n) * s + s] = out

        for i, out in enumerate(res[k][1]):
            newVesselImg[int(i % n) * s: int(i % n) * s + s, int(i / n) * s: int(i / n) * s + s] = out

        outAv.append(np.round(newAvImg[k[0]: k[0] + h, k[1]: k[1] + w]))
        outVessel.append(np.round(newVesselImg[k[0]: k[0] + h, k[1]: k[1] + w]))

    av = np.zeros_like(outAv[0])
    for i in range(len(outAv[0])):
        for j in range(len(outAv[0][0])):
            pointSum = [0, 0, 0, 0]
            for tmp in outAv:
                try:
                    pointSum[int(tmp[i][j])] += 1
                except:
                    raise
            av[i][j] = pointSum.index(max(pointSum))

    vessel = np.zeros_like(outVessel[0])
    for i in range(len(outVessel[0])):
        for j in range(len(outVessel[0][0])):
            pointSum = [0, 0]
            for tmp in outVessel:
                try:
                    pointSum[int(tmp[i][j])] += 1
                except:
                    raise
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
