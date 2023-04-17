# -*- coding: utf-8 -*-

import argparse
from utils.preprocess import Process
from utils.Train import train
from utils.Test import test


if __name__ == '__main__':

    arg = argparse.ArgumentParser()

    # setting of all
    arg.add_argument('--mode', default='test', type=str)
    arg.add_argument('--datasets', default='DRIVE')
    arg.add_argument('--cuda_device', default=0)
    arg.add_argument('--work_path', default='/data2/chenchouyu/MMF_Net')
    arg.add_argument('--num_workers', default=0)
    arg.add_argument('--resources', default={'HRF': "/data2/chenchouyu/arteryVeinDatasets/HRF_AV",
                                             'IOSTAR': "/data2/chenchouyu/arteryVeinDatasets/IOSTAR_AV",
                                             'DRIVE': "/data2/chenchouyu/arteryVeinDatasets/DRIVE_AV",
                                             'LES': "/data2/chenchouyu/arteryVeinDatasets/LES_AV"})

    # setting of preprocessing
    arg.add_argument('--preprocess', default=True, help='"True" for preprocess, "False" for not.')
    arg.add_argument('--patch_size', default=256)
    arg.add_argument('--split_number', default=600, type=int)

    # setting of model
    arg.add_argument('--beta', default=0.1)
    arg.add_argument('--MM_config', default=0.2)

    # setting of training
    arg.add_argument('--mode', default='train', type=str)
    arg.add_argument('--batch_size', default=2)
    arg.add_argument('--epoch', default=100)
    arg.add_argument('--lr', default=0.001)

    arg = arg.parse_args()
    process = Process(arg)

    if arg.mode == 'train':
        if arg.preprocess:
            print('Start pretreatment. ')
            process.run_train()
            print('Pretreatment completed. ')
        train(arg)

    if arg.mode == 'test':
        process.run_test()
        test(arg)
