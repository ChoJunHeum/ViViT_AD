import torch
from torchvision.transforms import ToTensor

import argparse
import logging
import warnings
import random
from os.path import join

from torch.utils.data import DataLoader
from einops import rearrange, repeat

from config import update_config

import init_utils
import data_utils
import log_utils
import crop_utils
import ops

parser = argparse.ArgumentParser(description='Advision')

parser.add_argument('--batch_size', '-bs', default=32, type=int)
parser.add_argument('--epoch', '-e', default=10000, type=int )
parser.add_argument('--dataset', '-ds', default='avenue', type=str)
parser.add_argument('--resume', '-r', default=None, type=str)
parser.add_argument('--model', '-m', default='vivit', type=str)

parser.add_argument('--bounding_box_weight', '-bbw', default=0.2, type=float)
parser.add_argument('--confidence_score', '-cs', default=.4, type=float)
parser.add_argument('--input_size', '-is', default=256, type=int)
parser.add_argument('--patch_size', '-ps', default=16, type=int)
parser.add_argument('--nframe', '-nf', default=9, type=int)

parser.add_argument('--save_epoch', '-se', default=5, type=int)
parser.add_argument('--init', '-init', default='normal', type=str,
                                            choices=['original', 'xavier', 'normal'])
parser.add_argument('--optimizer', '-opt', default='adamw', type=str,
                                            choices=['adam', 'adamw'])
parser.add_argument('--scheduler', '-sch', default='cosine', type=str,
                                            choices=['cosine', 'no', 'cosinewr'])
'''
parser.add_argument('--', '--', default=, type=)

'''

def train(cfg):

    # Setting Environment
    train_dataset = data_utils.CropTrainDataset(cfg)
    N = train_dataset.__len__()
    model, opts, schs, info = init_utils.get_model_opts(cfg=cfg, N=N)

    losses = init_utils.get_losses(cfg=cfg)
    
    # configure loggers
    loggers = [logging.getLogger()]
    loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]

    for l in loggers:
        if l.name != 'advision':
            l.handlers=[]
    if cfg.resume is not None:
        logger.info(f'Train resumed from {cfg.resume}')
        logger.info(f"Last lr: {info['last_lr']}")
        logger.info(f"Step count: {info['step']}")
        step = info['step']

    save_prefix = log_utils.make_date_dir(cfg.save_prefix)
    # save_path = join(save_prefix, 'total_model.pt')
    logger.info(f'Model save path: {save_prefix}')

    start_iter = step if cfg.resume else 0
    pil2tensor = ToTensor()

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)

    try:
        global_step = start_iter + 1
        logger.info('Start training')

        for epoch_ in range(cfg.epoch):
            epoch = epoch_ + 1

            epoch_acc = 0
            epoch_count = 0

            epoch_ta_acc = 0
            epoch_ta_count = 0

            epoch_irr_acc = 0
            epoch_irr_count = 0

            epoch_rot_acc = 0
            epoch_rot_count = 0

            for i, frames in enumerate(train_dataloader):
                i += 1

                frames = rearrange(frames, 'b (t c) w h -> b t c w h', c = 3).cuda()

                
                pred_ta, pred_irr, pred_rot = ops.step_train(frames, model, losses,
                                                opts, schs, cfg,epoch=epoch,
                                                global_step=global_step, cur_iter=i)

                acc, ta_acc , irr_acc , rot_acc, count = ops.cal_acc(pred_ta, pred_irr, pred_rot)


                epoch_acc += acc
                epoch_count += count*3

                epoch_ta_acc += ta_acc
                epoch_ta_count += count

                epoch_irr_acc += irr_acc
                epoch_irr_count += count

                epoch_rot_acc += rot_acc
                epoch_rot_count += count


                if i % 100 == 0:
                    print(f'[Train-{epoch}-{i}] Total acc: {1-(epoch_acc/epoch_count):.2f} | ',
                    f'TA acc: {1-(epoch_ta_acc/epoch_ta_count):.2f} | IRR acc: {1-(epoch_irr_acc/epoch_irr_count):.2f} | ',
                    f'ROT acc: {1-(epoch_rot_acc/epoch_rot_count):.2f}')

                    # print(pred_ta, pred_irr, pred_rot)
                    # break

            if epoch % cfg.save_epoch == 0:
                save_path = join(save_prefix, f'total_model_e{epoch}.pt')
                init_utils.save_all(model, opts, schs,
                                    save_path=save_path)

        save_path = join(save_prefix, f'total_model_e{epoch}.pt')                                
        init_utils.save_all(model, opts, schs,
                            save_path=save_path)
    except:
        save_path = join(save_prefix, f'total_model_ex.pt')   
        logger.info('Exception Occurred')
        init_utils.save_all(model, opts, schs,
                            save_path=save_path)
        logger.info(f'[Interrupted] Model_dict saved: {save_path}')
        logger.info('Log path: {}'.format(log_dir))
        logging.exception("Exception message:")

    finally:
        logger.handlers.clear()
        logging.shutdown()  

    return None


if __name__ == "__main__":

    args = parser.parse_args()
    train_cfg = update_config(args, mode='train')
    train_cfg.print_cfg()

    logger, log_dir = log_utils.get_logger('logs/')
    print(log_dir)

    train(train_cfg)