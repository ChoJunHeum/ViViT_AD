import glob
import os
import time
import argparse
import logging
from os.path import join

import numpy as np
# import torch
# from torch.utils.tensorboard import SummaryWriter

from config import Config
# import init_utils
# import data_utils
# import log_utils
# import calc_scores

parser = argparse.ArgumentParser(description='Advision Inference')
# parser.add_argument('--dataset', '-d', default='avenue', type=str)
parser.add_argument('--checkpoint',     '-ck', default=None, type=str, required=True)
parser.add_argument('--generator',      '-g', default='vgg16unet', type=str,
                                            choices=['vgg16unet', 'vgg13unet', 'unet'])
parser.add_argument('--device',         '-device', default='cuda', type=str)
parser.add_argument('--gpu_num',        '-gpu', default='3', type=str)
parser.add_argument('--nframe',         '-nf', default=4, type=int)
parser.add_argument('--resize_h',       '-rsh', default=128, type=int)
parser.add_argument('--resize_w',       '-rsw', default=128, type=int)
parser.add_argument('--max_h',          '-mxh', default=360, type=int)
parser.add_argument('--max_w',          '-mxw', default=640, type=int)
parser.add_argument('--confidence',     '-c', default=.4, type=int)


def main(cfg):
    import torch
    from torchvision.transforms import ToTensor

    import init_utils
    import data_utils
    import ops
    import log_utils
    import calc_scores
    torch.multiprocessing.set_start_method('spawn')   
    yolo_model, generator = init_utils.get_checkpoint_model(cfg)
    
    test_dirs = glob.glob(cfg.te_path)
    test_dirs.sort()
    
    gt_loader = data_utils.Label_loader(cfg, test_dirs)
    gt = gt_loader()
    pil2tensor = ToTensor()
    device = cfg.device
    
    backup = True
    backup_sets = {'psnr_unnorm': [], 
                    'psnr_unnorm_min': [],
                    'size_unnorm': [],
                    'size_norm': []
                    }
    total_scores = []; total_labels = []

    with torch.no_grad():
        for folder_idx, folder in enumerate(test_dirs):
            folder_dataset = data_utils.TestDataset(cfg, folder)

            psnrs_min_folder = []; sizes_min_folder = []

            if backup:
                backup_sets['psnr_unnorm'].append([])
                backup_sets['psnr_unnorm_min'].append([])
                backup_sets['size_unnorm'].append([])
                backup_sets['size_norm'].append([])

            for frame_idx, frames in enumerate(folder_dataset):
                input_frames, target_frame = frames
                last_frame = input_frames[-1]

                areas = ops.detect_nob(last_frame=last_frame,
                                        yolo_model=yolo_model,
                                        cfg=cfg,
                                        confidence=cfg.confidence,
                                        coi=[0])
                
                no_patch = False if len(areas) != 0 else True
                gen_inputs, gen_target = ops.prep_nob(input_frames=input_frames,
                                                        target_frame=target_frame,
                                                        areas=areas,
                                                        pil2tensor=pil2tensor,
                                                        cfg=cfg,
                                                        no_patch=no_patch)
                gen_pred = generator(gen_inputs).detach()
                
                # psnr_frame: psnrs for all patches in one frame
                #           [psnr_patch1, psnr_patch2, ...] in one target_frame
                psnr_frame = calc_scores.psnr_error(gen_pred, gen_target, reduce_mean=False).cpu().numpy()
                
                if no_patch:
                    assert psnr_frame.shape[0] == 1 # one element for the entire frame
                    size_norm = np.array(1.); size_unnorm = np.array(100.)
            
                else:
                    size_norm, size_unnorm = ops.infer_calc_area(areas, np.sqrt)
                psnr_weighted = psnr_frame * size_norm
                
                psnr_min = np.min(psnr_weighted)
                size_min = size_unnorm[np.argmin(psnr_weighted)]
                
                sizes_min_folder.append(size_min)
                psnrs_min_folder.append(psnr_min)

                if backup:
                    backup_sets['psnr_unnorm'][folder_idx].append(psnr_frame)
                    backup_sets['psnr_unnorm_min'][folder_idx].append(np.min(psnr_frame))
                    backup_sets['size_unnorm'][folder_idx].append(size_unnorm)
                    backup_sets['size_norm'][folder_idx].append(size_norm)

            # normalize psnrs_min_folder -> scores
            # get gt label -> gt
            labels_folder = gt[folder_idx][4:]
            total_labels.append(labels_folder)
            
            scores_folder = calc_scores.norm_scores(psnrs_min_folder, sizes=None)
            total_scores.append(scores_folder)
        
            if folder_idx == 0:
                break
        # gather gt, scores in all folders-> total_gt, total_scores
        # calc auc(total_gt, total_scores)
        auc = calc_scores.calc_auc(total_scores, total_labels)
        print(auc)
    return None


if __name__ == "__main__":
    # logger, log_dir = log_utils.get_logger('logs/')
    args = parser.parse_args()
    cfg = Config(args)
    cfg_desc = cfg.print_cfg()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu_num)
    print(f'CUDA_VISIBLE_DEVICES: {cfg.gpu_num}')

    main(cfg)

    # logger.info(cfg_desc)
    
    # main(logger, log_dir, train_cfg)