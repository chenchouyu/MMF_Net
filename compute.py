# -*- coding: utf-8 -*-
import os
from collections import defaultdict
import json

import numpy as np
from PIL import Image
from utils.metrics_computing import Evaluate

# avLabelPath = 'E:/label/DRIVE_label/av'
# vesselLabelPath = 'E:/label/DRIVE_label/ves'
avLabelPath = 'E:/label/HRF_label/av'
vesselLabelPath = 'E:/label/HRF_label/ves'
# avLabelPath = 'E:/label/IOSTAR_label/av'
# vesselLabelPath = 'E:/label/IOSTAR_label/ves'

# predAvPath = 'E:/project/MMF_Net/utils/TestResult/av'
# predVesPath = 'E:/project/MMF_Net/utils/TestResult/ves'

predAvPath = 'E:/project/MMF_Net/TestResult/av'
predVesPath = 'E:/project/MMF_Net/TestResult/ves'

# predAvPath = 'D:/100'

metrics = defaultdict(float)


def composition(image):
    label_av = np.copy(np.asarray(image))
    label = np.zeros_like(label_av[..., 0])

    label[label_av[:, :, 0] == 255] = 3
    label[label_av[:, :, 2] == 255] = 2
    label[label_av[:, :, 1] == 255] = 1

    return label


def compute_ves(pred):
    label_av = np.copy(np.asarray(pred))
    out = np.zeros_like(label_av[..., 0])

    out[label_av[:, :, 0] == 255] = 255
    out[label_av[:, :, 2] == 255] = 255
    out[label_av[:, :, 1] == 255] = 255

    return out


for name in os.listdir(avLabelPath):
    av = Image.open(os.path.join(avLabelPath, name))
    ves = Image.open(os.path.join(vesselLabelPath, name)).convert('L')

    # name = 'IOSTAR_' + name
    predAv = Image.open(os.path.join(predAvPath, name))
    predVes = Image.open(os.path.join(predVesPath, name)).convert('L')

    # computeVes = np.round(np.array(predVes) / 255)
    # predVes = compute_ves(predAv)
    labelVes = np.round(np.array(ves) / 255)

    labelAv = composition(av)

    currentMetrics = Evaluate(predAv, predVes, labelAv, labelVes)

    metrics.update(
        (k, metrics.get(k, 0) + currentMetrics.get(k, 0)) for k in metrics.keys() | currentMetrics.keys())

metrics = {k: v / len(os.listdir(avLabelPath)) for k, v in metrics.items()}
# df = pd.DataFrame.from_dict(metrics)
# df.to_csv('HRF.csv', index=False)
# with open('./iostar.json', 'w') as f:
#     json.dump(metrics, f)

print(metrics)
