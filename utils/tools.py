# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def threshold_vessel(outList, name, mode='av'):
    if 'Drive' in name[0]:
        size = (512, 512)
    elif 'HRF' in name[0]:
        size = (1536, 1024)
    else:
        size = (1024, 1024)

    if mode == 'av':
        newImg = Image.new('RGB', size)
    else:
        newImg = Image.new('L', size)

    s = outList[0].size[0]
    w, h = newImg.size
    m, n = math.ceil(w / s), math.ceil(h / s)

    for i, out in enumerate(outList):
        newImg.paste(out, (int(i / n) * s, int(i % n) * s))

    return newImg


def threshold_ves(outList, idx, pos, w, h, patchSize, axis=0.5):
    c = 1
    newImg = np.zeros((patchSize, patchSize, c))
    for i, out in enumerate(outList):
        x, y = idx[i]
        tmpImg = np.zeros((patchSize, patchSize, c))
        tmpImg[y:y + 256, x:x + 256, :] = out
        newImg += tmpImg
    for x in range(patchSize):
        for y in range(patchSize):
            for z in range(c):
                num = pos[0][y][x]
                if num == 0:
                    continue
                mid = newImg[y][x][z] / num
                if mid > axis:
                    newImg[y][x][z] = 1
                else:
                    newImg[y][x][z] = 0
    newImg = np.round(cv2.resize(newImg, (w, h)))
    return newImg


def rgb_to_bgr(img):
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    return img


def restore_av(data):
    r = np.zeros_like(data)
    g = np.zeros_like(data)
    b = np.zeros_like(data)

    r[data == 3] = 1
    g[data == 1] = 1
    b[data == 2] = 1

    r, g, b = np.expand_dims(r, 2), np.expand_dims(g, 2), np.expand_dims(b, 2)

    return np.concatenate((r, g, b), axis=2)
