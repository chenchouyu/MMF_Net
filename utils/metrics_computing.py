# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix


def composition(image):
    label_av = np.copy(np.asarray(image))
    label = np.zeros_like(label_av[..., 0])

    label[label_av[:, :, 0] == 255] = 3
    label[label_av[:, :, 2] == 255] = 2
    label[label_av[:, :, 1] == 255] = 1

    return label


def compute_for_av(pred, label):

    pred = pred.flatten()
    label = label.flatten()

    ma = confusion_matrix(label, pred)

    TP, TN, FP, FN = ma[3][3], ma[2][2], ma[3][2], ma[2][3]

    return {'avAcc': (TP + TN) / (TP + TN + FN + FP),
            'avSen': TP / (TP + FN),
            'avSpe': TN / (TN + FP),
            'avF1': (2 * TP) / (2 * TP + FP + FN),
            'avPre': TP / (TP + FP),
            'avIou': TP / (TP + FN + FP)
            }


def compute_for_vessel(pred, label):

    label = np.array(label)

    pred = pred.flatten()
    label = label.flatten()

    ma = confusion_matrix(label, pred)

    TP, TN, FP, FN = ma[1][1], ma[0][0], ma[1][0], ma[0][1]

    return {'vesAcc': (TP + TN) / (TP + TN + FN + FP),
            'vesSen': TP / (TP + FN),
            'vesSpe': TN / (TN + FP),
            'vesF1': (2 * TP) / (2 * TP + FP + FN),
            'vesPre': TP / (TP + FP),
            'vesIou': TP / (TP + FN + FP)
            }


def Evaluate(savePredAv, savePredVessel, lab, ves):
    computeAv = composition(savePredAv)
    computeVes = np.round(np.array(savePredVessel) / 255)

    metricsAv = compute_for_av(computeAv, lab)
    metricsVes = compute_for_vessel(computeVes, ves)

    metrics = metricsAv.copy()
    metrics.update(metricsVes)

    return metrics
