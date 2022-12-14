import logging
import logging.handlers
import os
import datetime

import numpy as np


def get_logger(log_path='logs/'):
    """
    :param log_path
    :return: logger instance
    """
    
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logger = logging.getLogger('advision')
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s %(message)s', date_format)
    i = 0
    today = datetime.datetime.now()
    name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    while os.path.exists(os.path.join(log_path, name)):
        i += 1
        name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    
    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
    return logger, os.path.join(log_path, name)


def make_date_dir(path):
    """
    :param path
    :return: os.path.join(path, date_dir)
    """
    if not os.path.exists(path):
        os.mkdir(path)
    i = 0
    today = datetime.datetime.now()
    name = today.strftime('%Y%m%d')+'-'+'%02d' % i
    while os.path.exists(os.path.join(path, name)):
        i += 1
        name = today.strftime('%Y%m%d')+'-'+'%02d' % i
    os.mkdir(os.path.join(path, name))
    return os.path.join(path, name)


def find_latest_dir(path):
    dirs = os.listdir(path)
    dirs_splited = list(map(lambda x:x.split("-"), dirs))
    
    # find latest date
    dirs_date = [int(dir[0]) for dir in dirs_splited]
    dirs_date.sort()
    latest_date = dirs_date[-1]
    
    # find latest num in lastest date
    dirs_num = [int(dir[1]) for dir in dirs_splited if int(dir[0]) == latest_date]
    dirs_num.sort()
    latest_num = dirs_num[-1]
    latest_dir = str(latest_date) + '-' + '%02d' % latest_num

    return os.path.join(path, latest_dir)


def desc(params_dict):
    description = ''
    for k, v in params_dict.items():
        description += k + ":" + str(v) + ", "
    return description


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.best_epoch = 0
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics, epoch):
        if self.best is None:
            self.best = metrics
            self.best_epoch = epoch
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_epoch = epoch
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
