# -*- coding: utf-8 -*-
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.optim as optimizer
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Net.MMF import MMF
from utils.dataloader import EyeBallData
from utils.loss import MultiLoss
from utils.tools import count_parameters, threshold_vessel, restore_av, better_than, Logger
from utils.metrics_computing import Evaluate

import time
import os

# seed = 3407
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(config):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    Model = MMF()
    Model = nn.DataParallel(Model, device_ids=[1, 2])
    Model = Model.to(config.cuda_device)

    print('cuda_use:', torch.cuda.is_available())
    print("Total number of parameters: " + str(np.round(count_parameters(Model) / 1e6, 3)) + 'M')

    # prepare datasets
    trainLoad = EyeBallData(config, mode='train')
    trainLoader = DataLoader(trainLoad, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    valLoad = EyeBallData(config, mode='validation')
    valLoader = DataLoader(valLoad, shuffle=False)

    # optimizer and loss function
    Optimize = optimizer.Adam(Model.parameters(), lr=config.lr, weight_decay=1e-5)
    Criterion = MultiLoss(config.cuda_device, [0.4, 0.6])

    # other things
    bestMetrics = defaultdict(float)
    dataset = '_'.join(config.datsets) if type(config.datasets) == list else str(config.datasets)

    logger = Logger(f'./log/{dataset}_NoTransformer_big.log')
    modelSavePath = os.path.join(config.work_path, 'model_NoTransformer_big', dataset)
    validationSavePath = os.path.join(config.work_path, 'validation_NoTransformer_big', dataset)
    os.makedirs(modelSavePath, exist_ok=True)

    print('The model began training on {}......'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    for epoch in range(config.epoch):
        vesLoss, avLoss, printLoss = 0, 0, 0
        Model.train()
        os.makedirs(os.path.join(validationSavePath, str(epoch + 1), 'av'), exist_ok=True)
        os.makedirs(os.path.join(validationSavePath, str(epoch + 1), 'ves'), exist_ok=True)

        for idx, data in enumerate(trainLoader):
            if idx == 5 and epoch == 0:
                t = time.time()
            img, lab, ves = data
            if torch.cuda.is_available():
                img, lab, ves = img.to(config.cuda_device), lab.to(config.cuda_device), ves.to(config.cuda_device)
            img = Variable(img)
            lab, ves = Variable(lab.type(torch.long)), Variable(ves.type(torch.float))
            Optimize.zero_grad()

            pred_ves, pred_av = Model(img)

            ves_loss, av_loss, loss = Criterion(pred_ves, pred_av, ves, lab)

            loss.backward()
            Optimize.step()

            vesLoss += ves_loss
            avLoss += av_loss

            # print part
            printLoss += loss.item()
            if idx == 5 and epoch == 0:
                t1 = time.time()
                print('The duration of an epoch is about %.2f minutes' % ((t1 - t) * len(trainLoader) / 60))
                logger.logger.info(
                    'The duration of an epoch is about %.2f minutes' % ((t1 - t) * len(trainLoader) / 60))

        print('[%d] vesLoss: %.3f avLoss: %.3f loss: %.3f' % (epoch + 1, vesLoss, avLoss, printLoss))
        logger.logger.info('[%d] vesLoss: %.3f avLoss: %.3f loss: %.3f' % (epoch + 1, vesLoss, avLoss, printLoss))

        # validation part
        Model.eval()
        metrics = defaultdict(float)

        with torch.no_grad():
            for item, data in enumerate(valLoader):

                outDict = defaultdict(list)
                imgList, lab, ves, curSize, oriSize, name = data

                lab, ves = lab.numpy().squeeze(), ves.numpy().squeeze()

                oriSize = tuple(map(int, oriSize))
                curSize = tuple(map(int, curSize))

                for img in imgList:
                    img = img.to(config.cuda_device)
                    predVes, predAv = Model(img)

                    _, predAv = torch.max(predAv.cpu().data, 1)
                    predAv = predAv[0]

                    # _, predVes = torch.max(predVes.cpu().data, 1)
                    # predVes = predVes[0]
                    predVes = predVes[0][0].cpu().detach().numpy()

                    outDict['av'].append(predAv)
                    outDict['ves'].append(predVes)

                predAv, predVessel = threshold_vessel(outDict, curSize)

                savePredAv = Image.fromarray(np.uint8(np.round(cv2.resize(restore_av(predAv), oriSize)))).convert('RGB')

                savePredVessel = Image.fromarray(np.uint8(cv2.resize(np.round(predVessel) * 255, oriSize))).convert('L')

                savePredVessel.save(os.path.join(validationSavePath, str(epoch + 1), 'ves', name[0]))
                savePredAv.save(os.path.join(validationSavePath, str(epoch + 1), 'av', name[0]))

                currentMetrics = Evaluate(savePredAv, savePredVessel, lab, ves)

                metrics.update(
                    (k, metrics.get(k, 0) + currentMetrics.get(k, 0)) for k in metrics.keys() | currentMetrics.keys())

            metrics = {k: v / len(valLoader) for k, v in metrics.items()}

            if better_than(metrics, bestMetrics):
                bestMetrics = metrics
                modelName = f'{epoch}_' + '_'.join(
                    [str(k) + '_' + str(np.round(v, 3)) for k, v in metrics.items()]) + '.pth'
                torch.save(Model.state_dict(), os.path.join(modelSavePath, modelName))
                print('[%d] validate metrics: Acc %.3f / %.3f F1 %.3f / %.3f, saved.' %
                      (epoch + 1, metrics['avAcc'], metrics['vesAcc'], metrics['avF1'], metrics['vesF1']))
                logger.logger.info('[%d] validate metrics: Acc %.3f / %.3f F1 %.3f / %.3f, saved.' %
                                   (epoch + 1, metrics['avAcc'], metrics['vesAcc'], metrics['avF1'], metrics['vesF1']))
            else:
                print('[%d] validate metrics: Acc %.3f / %.3f F1 %.3f / %.3f.' %
                      (epoch + 1, metrics['avAcc'], metrics['vesAcc'], metrics['avF1'], metrics['vesF1']))
                logger.logger.info('[%d] validate metrics: Acc %.3f / %.3f F1 %.3f / %.3f.' %
                                   (epoch + 1, metrics['avAcc'], metrics['vesAcc'], metrics['avF1'], metrics['vesF1']))
