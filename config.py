#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from glob import glob
import os

if not os.path.exists('tensorboard_log'):
    os.mkdir('tensorboard_log')
if not os.path.exists('weights'):
    os.mkdir('weights')
if not os.path.exists('results'):
    os.mkdir('results')

share_config = {'mode': 'training',
                'dataset': 'avenue',
                'img_size': (64, 64),
                'data_root': '/home/chojh21c/ADGW/ViT_MT/datasets/'}  # remember the final '/'


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode=None):
    share_config['mode'] = mode
    assert args.dataset in ('ped2', 'avenue', 'shanghai', 'CalTech'), 'Dataset error.'
    share_config['dataset'] = args.dataset

    if mode == 'train':
        share_config['cropped_data'] = share_config['data_root'] + args.dataset + '_cropped_04'
        share_config['train_data'] = share_config['data_root'] + args.dataset + '/training'
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/testing/'
        share_config['save_prefix'] = 'log'+'checkpoints'
        share_config['save_epoch'] = 1


        share_config['resume'] = glob(f'weights/{args.resume}*')[0] if args.resume else None
        share_config['epoch'] = args.epoch
        share_config['model'] = args.model
        share_config['device'] = 'cuda'
        

        share_config['batch_size'] = args.batch_size
        share_config['input_size'] = args.input_size
        share_config['patch_size'] = args.patch_size
        share_config['nframe'] = args.nframe

        share_config['optimizer'] = args.optimizer
        share_config['scheduler'] = args.scheduler

        share_config['resize_w'] = 64
        share_config['resize_h'] = 64
        share_config['symmetric'] = True
        share_config['verbose'] = 10
        

        share_config['g_lr'] = 1e-3
        share_config['l2'] = 1e-5 
        share_config['init'] = 'normal'

        
        

        

    elif mode == 'test':
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/testing/'
        share_config['trained_model'] = args.trained_model
        share_config['input_size'] = args.input_size
        share_config['confidence_score'] = args.confidence_score
        share_config['bbw'] = args.bbw


    return dict2class(share_config)  # change dict keys to class attributes
