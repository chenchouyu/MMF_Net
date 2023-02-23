# -*- coding: utf-8 -*-

import argparse
import yaml

from utils.utils import preprocess
from utils.Train import train
from utils.Test import test


if __name__ == '__main__':

    arg = argparse.ArgumentParser()
    # setting of all
    arg.add_argument('--mode', default='train', type=str)
    arg.add_argument('--yamlPath', default='./config.yml', type=str)
    arg.add_argument('--dataset', default='Drive',
                     help="Using ['Drive', 'HRF', 'LES'] to complete the training of multiple data sets. ")

    # setting of test
    arg.add_argument('--testModel', default='Best',
                     help="You can choose 'Best' or 'Final' to test a specific model. ")

    arg = arg.parse_args()

    with open(arg.yamlPath, 'r') as f:
        y = yaml.load(f, Loader=yaml.FullLoader)
    parameter = y[arg.mode]

    config = argparse.ArgumentParser()

    config.add_argument('--dataset', default=arg.dataset)
    if arg.mode == 'test':
        config.add_argument('--testModel', default=arg.testModel)

    for k in parameter.keys():
        config.add_argument('--'+k, default=parameter[k])
    config = config.parse_args()

    if arg.mode == 'train':
        if not config.pretreatment:
            print('Start pretreatment. ')
            preprocess(config)
            y[arg.mode]['pretreatment'] = True
            with open(arg.yaml, 'w', encoding='utf-8') as w_f:
                yaml.dump(y, w_f)
            print('Pretreatment completed. ')
        train(config)

    if arg.mode == 'test':
        test(config)
