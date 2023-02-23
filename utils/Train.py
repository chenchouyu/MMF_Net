# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optimizer
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Net.MMF import MMF
from utils.dataloader import EyeBallData, decomposition
from utils.loss import MultiLoss
from utils.tools import count_parameters, threshold_vessel, rgb_to_bgr

import time
import os


def train(config):
    Model = MMF()
    Model = Model.to(config.cuda_device)

    print('cuda_use:', torch.cuda.is_available())
    print("Total number of parameters: " + str(np.round(count_parameters(Model) / 1e6, 3)) + 'M')

    # prepare datasets
    trainLoad = EyeBallData(config)
    trainLoader = DataLoader(trainLoad, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    valLoad = EyeBallData(config, is_val=True)
    valLoader = DataLoader(valLoad, shuffle=False)

    # optimizer and loss function
    Optimize = optimizer.Adam(Model.parameters(), lr=config.lr)
    Criterion = MultiLoss(config.cuda_device, [0.4, 0.6])

    # other things
    standardAvSen = 0.8
    standardAvSpe = 0.8
    standardBACC = 0.8

    modelSavePath = os.path.join(config.save_model, date)

    os.makedirs(os.path.join(config.save_model, date), exist_ok=True)

    print('The model began training on {}......'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    for epoch in range(config.epoch):
        printLoss = 0
        Model.train()
        for idx, data in enumerate(trainLoader):
            img, lab, ves = data
            if torch.cuda.is_available():
                img, lab, ves = img.to(config.cuda_device), lab.to(config.cuda_device), ves.to(config.cuda_device)
            img = Variable(img)
            lab, ves = Variable(lab.type(torch.long)), Variable(ves.type(torch.long))
            ves = ves.squeeze()

            Optimize.zero_grad()

            pred_ves, pred_av = Model(img)

            loss = Criterion(pred_ves, pred_av, ves, lab)

            loss.backward()
            Optimize.step()

            # print part
            printLoss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, printLoss))

        # validation part
        if config.isvalidation:
            Model.eval()
            senAv, senVessel, accAv, accVes = 0, 0, 0, 0

            with torch.no_grad():
                for item, data in enumerate(valLoader):

                    outDict = defaultdict(list)
                    imgList, lab, ves, w, h, name = data

                    for img in imgList:
                        img = img.to(config.cuda_device)
                        predVes, predAv = Model(img)

                        _, predAv = torch.max(predAv.cpu().data, 1)
                        predAv = predAv[0]

                        predVessel = predVessel.squeeze().cpu().detach().numpy()

                        outDict['av'].append(predAv)
                        outDict['ves'].append(predVes)

                    outAv = threshold_vessel(outDict, name, size)
                    saveAv = np.uint8(np.round(np.array(outAv.resize((w, h)))) * 255)
                    outAv = decomposition(saveAv)
                    saveAv = rgb_to_bgr(saveAv)

                    # outVes = threshold_ves(outListVes, idx, pos, w, h, patch_size)
                    outVes = threshold_vessel(outListVes, name, mode='ves')
                    outVes = np.uint8(np.round(np.array(outVes.resize((w, h)))))
                    saveVes = outVes * 255

                    lab, ves = lab.squeeze().numpy(), ves.squeeze().numpy()
                    outAv = outAv.squeeze()

                    classes = [2, 3]
                    TP = ((outAv == classes[1]) & (lab == classes[1])).sum()
                    TN = ((outAv == classes[0]) & (lab == classes[0])).sum()
                    FN = ((outAv == classes[0]) & (lab == classes[1])).sum()
                    FP = ((outAv == classes[1]) & (lab == classes[0])).sum()

                    if TP + FN == 0 or TN + FP == 0:
                        senAv.append(0)
                        speAv.append(0)
                    else:
                        senAv.append(TP / (TP + FN))
                        speAv.append(TN / (TN + FP))

                    accVes = []
                    TP = ((outVes == 1) & (ves == 1)).sum()
                    TN = ((outVes == 0) & (ves == 0)).sum()
                    FN = ((outVes == 0) & (ves == 1)).sum()
                    FP = ((outVes == 1) & (ves == 0)).sum()

                    if TP + TN + FN + FP == 0:
                        accVes.append(0)
                    else:
                        accVes.append((TP + TN) / (TP + TN + FN + FP))

                    os.makedirs(os.path.join('./Result', str(epoch + 1), 'ves'), exist_ok=True)
                    os.makedirs(os.path.join('./Result', str(epoch + 1), 'av'), exist_ok=True)
                    cv2.imwrite(os.path.join('./Result', str(epoch + 1), 'ves', name[0]), saveVes)
                    cv2.imwrite(os.path.join('./Result', str(epoch + 1), 'av', name[0]), saveAv)

            senAv, speAv, accVes = np.array(senAv), np.array(speAv), np.array(accVes)
            # senAv, speAv = np.array(senAv), np.array(speAv)


            senAv, speAv, accVes = get_acc(Model, valLoader, epoch)
            senAv, speAv, accVes = np.round(senAv, 4), np.round(speAv, 4), np.round(accVes, 4)
            print("Av sen: %.4f, Av spe: %.4f, Vessel accuracy: %.4f" % (senAv, speAv, accVes))
            with open('res.txt', 'a') as f:
                f.write("[%d] Av sen: %.4f, Av spe: %.4f, Vessel accuracy: %.4f \n" % (epoch + 1, senAv, speAv, accVes))
            # save model
            if senAv + speAv > 2 * standardBACC:
                modelName = 'BestModel' + '.pkl'
                torch.save(Model.state_dict(), os.path.join(modelSavePath, modelName))
                standardBACC = (senAv + speAv)/2
                print('save best model, BACC:', standardBACC)
            elif senAv > standardAvSen:
                modelName = 'BestSenModel' + '.pkl'
                torch.save(Model.state_dict(), os.path.join(modelSavePath, modelName))
                standardAvSen = senAv
                print('save best sen model, Sen:', standardAvSen)
            elif speAv > standardAvSpe:
                modelName = 'BestSpeModel' + '.pkl'
                torch.save(Model.state_dict(), os.path.join(modelSavePath, modelName))
                standardAvSpe = speAv
                print('save best spe model, Sen:', standardAvSpe)

    print('The model completed training on {}......'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

