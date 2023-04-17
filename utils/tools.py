# -*- coding: utf-8 -*-
import logging
from logging import handlers

import cv2
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def threshold_vessel(outList, size):
    w, h = size
    s = outList['av'][0].shape[0]

    m = w // s + (0 if w % s == 0 else 1)
    n = h // s + (0 if h % s == 0 else 1)

    tmp_w, tmp_h = (m * s - w) // 2, (n * s - h) // 2

    predAv = np.zeros((n * s, m * s))
    predVessel = np.zeros((n * s, m * s))

    for j in range(m):
        for i in range(n):
            predAv[i * s: (i + 1) * s, j * s: (j + 1) * s] = outList['av'][j * n + i]
            predVessel[i * s: (i + 1) * s, j * s: (j + 1) * s] = outList['ves'][j * n + i]

    predAv, predVessel = predAv[tmp_h: tmp_h + h, tmp_w: tmp_w + w], predVessel[tmp_h: tmp_h + h, tmp_w: tmp_w + w]

    return predAv, predVessel


def rgb_to_bgr(img):
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    return img


def restore_av(data):
    r = np.zeros_like(data)
    g = np.zeros_like(data)
    b = np.zeros_like(data)

    r[data == 3] = 255
    g[data == 1] = 255
    b[data == 2] = 255

    r, g, b = np.expand_dims(r, 2), np.expand_dims(g, 2), np.expand_dims(b, 2)

    return np.concatenate((r, g, b), axis=2)


def better_than(a1, a2):
    sign = 0
    for k in a1.keys():
        if a1[k] > a2[k]:
            sign += 1
    if sign > len(a1) // 2:
        return True
    else:
        return False


# logger
class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')

        th.setFormatter(format_str)
        self.logger.addHandler(th)
